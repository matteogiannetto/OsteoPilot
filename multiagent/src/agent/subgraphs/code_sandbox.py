# agent/subgraphs/code_sandbox.py
from __future__ import annotations
import ast
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import ulid

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, SystemMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from ..graph_state import SandboxExtendedState
from ..ollama_retry import is_ollama_usage_limit_error
from .subgraph_contract import SubgraphModule
from ..tools.session_management import AttachmentInfo


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE_ROOT = os.environ.get(
    "BIOMED_WORKSPACE_ROOT",
    str(REPO_ROOT),
)

# Maximum number of code-generation / execution attempts per sandbox run.
MAX_SANDBOX_ATTEMPTS = 3
SANDBOX_SUMMARY_PREVIEW_ITEMS = 5
SANDBOX_SUMMARY_FULL_SCALAR_LIST_ITEMS = 50
SANDBOX_SUMMARY_MAX_STRING_CHARS = 500
SANDBOX_SUMMARY_MAX_JSON_CHARS = 8000
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 600


class SandboxSpecModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_type: str = Field(
        default="generic",
        description="Short task family label; this is not the full task contract.",
    )
    target_files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Known input files/directories with path, kind, source_root, and optional metadata.",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class SandboxRepairDecision(BaseModel):
    action: str = Field(pattern="^(regenerate_code|regenerate_spec|give_up)$")
    reason: str = Field(default="")
    repair_message: str = Field(default="")
    withdrawal_message: str = Field(default="")


def _new_attachment_id(prefix: str) -> str:
    """Generate a type-prefixed ULID for attachments, e.g. file-01J5Z8..."""
    return f"{prefix}-{str(ulid.new())}"



class CodeSandboxSubgraph(SubgraphModule):
    key = "code_sandbox"

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        max_repairs: int = MAX_SANDBOX_ATTEMPTS,
        unrestricted: bool = False,
    ) -> None:
        """
        The Ollama-backed agent model is supplied by the parent agent.  Tests may
        provide a substitute model explicitly.

        max_repairs controls how many code-repair cycles are allowed after a
        failed execution.  Pass 0 for single-shot execution (no repair loop).

        unrestricted removes all domain-specific restrictions from the spec
        planner and code generator prompts.  Use for baseline modes where
        code_sandbox is the only available subgraph and must handle all tasks.
        """
        if llm is None:
            raise RuntimeError(
                "CodeSandboxSubgraph requires the Ollama-backed model supplied by BiomedAgent."
            )

        # The null check above establishes this invariant for all graph nodes.
        self.llm: BaseChatModel = llm
        self.max_repairs = max_repairs
        self.unrestricted = unrestricted

    def description(self) -> str:
        if self.unrestricted:
            return (
                "A general-purpose Python 3.11 execution sandbox. "
                "It can implement and run any Python-solvable task, including domain-specific "
                "biomedical workflows such as microscopy preprocessing, segmentation, quantitative "
                "morphometry analysis, and statistical classification. "
                "Use this subgraph for all tasks that require code execution."
            )
        return (
            "A restricted Python execution sandbox for SMALL, SINGLE-PURPOSE utility tasks. "
            "It can run short Python snippets against files in the current session, use "
            "session/data as read-only task input, and inspect the configured read-only workspace data. "
            "Its responsibilities focus on lightweight data access and simple introspection, "
            "NOT domain-specific taks. "
            "Typical responsibilities include: listing files in a folder, filtering by extension, "
            "extracting simple metadata, previewing CSV files, computing basic statistics, "
            "or generating minimal structured outputs (e.g., sandbox_result.json). "
            "When a task is centered on an unknown folder, file sizes, TIFF stack geometry, "
            "or converting a TIFF folder into a NIfTI volume, prefer the dedicated "
            "folder_volume_intake subgraph instead of this sandbox. "
            "It can also register newly discovered or generated files as attachments so that other "
            "subgraphs—especially domain-specific ones—can operate on them. "
            "This subgraph MUST NOT perform heavy or domain-specific workflows (e.g., microscopy "
            "preprocessing, segmentation, image enhancement, or Hotelling T2/KNN classification). "
            "It is intended for atomic utility operations that support or prepare higher-level tasks."
        )

    def capability_text(self) -> str:
        if self.unrestricted:
            return (
                "- code_sandbox (General-Purpose Python Sandbox)\n"
                "  SCOPE: Executes any Python 3.11 code to solve the given task.\n"
                "  CAPABILITIES:\n"
                "    • Biomedical image processing: TIFF/NIfTI I/O, preprocessing, segmentation\n"
                "    • Quantitative analysis: morphometry, statistics, classification\n"
                "    • File discovery, format conversion, CSV/JSON inspection\n"
                "    • Any computation expressible in Python with standard scientific libraries\n"
                "  LIBRARIES: numpy, pandas, nibabel, tifffile, skimage, sklearn, matplotlib, pathlib\n"
                "  LIMITATIONS: No network access, no subprocess, no os.walk.\n"
                "    Write outputs only to the sandbox run directory.\n"
            )
        return (
            "- code_sandbox (Python Sandbox Subgraph)\n"
            "  SCOPE & RESPONSIBILITIES:\n"
            "    • Executes SMALL, RESTRICTED Python scripts to carry out single-purpose utility tasks.\n"
            "    • Operates ONLY on files located in:\n"
            "        - writable session output areas,\n"
            "        - session/data for read-only task inputs,\n"
            "        - the configured read-only workspace root for generic task data.\n"
            "    • Writes only sandbox_result.json and small generated artifacts in the sandbox run directory;\n"
            "      session/data is never an output location.\n"
            "    • Designed for lightweight, preparation-oriented tasks that other subgraphs depend on.\n"
            "\n"
            "  WHAT IT CAN DO:\n"
            "    • List files in a directory (recursive or flat) and return structured results.\n"
            "    • Filter files by extension (e.g., .tif, .tiff, .csv, .png).\n"
            "    • Read CSV files and produce simple human-interpretable outputs:\n"
            "        - head/tail preview tables\n"
            "        - column names, shapes, basic statistics\n"
            "    • Load simple file metadata (size, dtype, image shape via tifffile if necessary).\n"
            "    • Create small, simple derived artifacts (e.g., a downsampled thumbnail, a text report).\n"
            "    • Emit `sandbox_result.json` as a structured result.\n"
            "    • Register discovered or generated files as attachments in the global agent state.\n"
            "\n"
            "  WHAT IT CANNOT DO (STRICT LIMITATIONS):\n"
            "    • MUST NOT perform domain-specific pipelines (e.g., microscopy preprocessing, segmentation,\n"
            "      denoising, flat-field correction, 3D image reconstruction, quantitative analysis).\n"
            "      Those tasks belong to specialized subgraphs such as imaging_preprocessing.\n"
            "    • MUST NOT implement, repair, or bypass Hotelling T2/KNN classification, nearest-neighbor\n"
            "      ranking, weighted voting, or KNN feature-driver analysis. Those tasks must use the\n"
            "      dedicated quantitative_imaging_analysis tool find_similar_patients_by_block_hotelling_t2.\n"
            "    • MUST NOT combine multiple unrelated tasks in one execution.\n"
            "      Each subgraph_task MUST be single-purpose.\n"
            "    • MUST NOT produce complex multi-step workflows—only simple, atomic utility actions.\n"
            "    • MUST NOT interact with files outside the allowed roots, run unsafe imports, or create\n"
            "      network connections.\n"
            "\n"
            "  GOOD EXAMPLES OF APPROPRIATE code_sandbox TASKS:\n"
            "    ✓ \"List all .tif and .tiff files under data/slices and register relevant files as attachments.\"\n"
            "    ✓ \"Show the first 20 rows of sample.csv.\"\n"
            "    ✓ \"Compute basic column stats for this CSV file.\"\n"
            "    ✓ \"Generate a file listing JSON for a folder.\"\n"
            "\n"
            "  BAD EXAMPLES (SHOULD NOT BE ROUTED HERE):\n"
            "    ✗ \"Preprocess or denoise microscopy TIFFs.\" (belongs to imaging_preprocessing)\n"
            "    ✗ \"Segment a region of interest from SR-microCT volume.\" (domain-specific)\n"
            "    ✗ \"Implement custom Hotelling T2 KNN classification.\" (belongs to quantitative_imaging_analysis)\n"
            "    ✗ \"Apply full preprocessing pipelines or multi-stage workflows.\"\n"
            "\n"
            "  DESIGN INTENT:\n"
            "    The sandbox acts as a *utility layer* that other subgraphs rely upon for file discovery,\n"
            "    simple metadata extraction, or preparing attachments. It is intentionally limited to ensure\n"
            "    safety, isolation, and modular task decomposition."
        )


    def capability_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description(),
            "examples": [
                "Show the first 5 rows of data.csv",
                "Compute min/mean/max for each column in data.csv",
                "List all TIFF files under the session path or workspace data and register relevant files as attachments",

            ],
            "result_contract": (
                "Python code writes sandbox_result.json in its working directory. "
                "For file listings, sandbox_result.kind='file_listing' and data.files is a list of files "
                "with fields like filename, path, size_mb, extension, and source_root."
            ),
        }

    def _workspace_root(self, state: SandboxExtendedState) -> str:
        session_value = state.get("session")
        session: dict[str, Any] = dict(session_value) if isinstance(session_value, Mapping) else {}
        root = (
            state.get("workspace_root")
            or (session.get("workspace_root") if isinstance(session, dict) else None)
            or DEFAULT_WORKSPACE_ROOT
        )
        return str(Path(str(root)).expanduser().resolve())

    def _allowed_roots(self, session_path: str, workspace_root: str) -> dict[str, Path]:
        session_root = Path(session_path).expanduser()
        roots = {
            "session_data_readonly": (session_root / "data").resolve(strict=False),
            "session": session_root.resolve(),
            "workspace": Path(workspace_root).expanduser().resolve(),
        }
        return roots

    def _session_data_path(self, session_path: str) -> str:
        return str((Path(session_path).expanduser() / "data").resolve(strict=False))

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _classify_allowed_path(self, path: Path, roots: dict[str, Path]) -> str | None:
        resolved = path.expanduser().resolve(strict=False)
        for name, root in roots.items():
            if self._is_relative_to(resolved, root):
                return name
        return None

    def _resolve_target_path(self, raw_path: Any, roots: dict[str, Path]) -> tuple[str, str | None]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return "", None

        candidate = Path(raw_path.strip()).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            return str(resolved), self._classify_allowed_path(resolved, roots)

        if candidate.parts and candidate.parts[0] == "data":
            resolved = (roots["session"] / candidate).resolve(strict=False)
            if resolved.exists():
                return str(resolved), self._classify_allowed_path(resolved, roots)

        # Prefer an existing path. The session data symlink is checked first so
        # data/... paths are classified as read-only session inputs.
        for root_name in ("session_data_readonly", "workspace", "session"):
            resolved = (roots[root_name] / candidate).resolve(strict=False)
            if resolved.exists():
                return str(resolved), root_name

        resolved = (roots["workspace"] / candidate).resolve(strict=False)
        return str(resolved), self._classify_allowed_path(resolved, roots)

    def _normalize_sandbox_spec(
        self,
        spec: Any,
        session_path: str,
        workspace_root: str,
    ) -> Any:
        if not isinstance(spec, dict):
            return spec

        roots = self._allowed_roots(session_path, workspace_root)
        normalized = dict(spec)
        target_files = normalized.get("target_files")
        if not isinstance(target_files, list):
            return normalized

        normalized_targets = []
        for target in target_files:
            if not isinstance(target, dict):
                continue

            item = dict(target)
            resolved, source_root = self._resolve_target_path(item.get("path"), roots)
            if resolved and source_root:
                item["path"] = resolved
                item["source_root"] = source_root
            elif resolved:
                # Keep the path visible for diagnostics, but mark it unsafe/unknown
                # so generated code has no reason to treat it as an allowed target.
                item["path"] = resolved
                item["source_root"] = "unknown"
            normalized_targets.append(item)

        normalized["target_files"] = normalized_targets
        return normalized

    @staticmethod
    def _message_content_preview(message: Any, limit: int = 1200) -> str:
        content = getattr(message, "content", message)
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

    def _attachment_context_for_spec(self, attachments: Any) -> list[dict[str, Any]]:
        if not isinstance(attachments, list):
            return []

        by_id = {
            a.get("id"): a
            for a in attachments
            if isinstance(a, dict) and a.get("id")
        }

        context: list[dict[str, Any]] = []
        for a in attachments:
            if not isinstance(a, dict):
                continue

            parent_id = a.get("parent")
            parent = by_id.get(parent_id) if parent_id else None
            item = {
                "id": a.get("id"),
                "filename": a.get("filename") or a.get("name"),
                "path": a.get("path"),
                "origin_path": a.get("origin_path"),
                "size_mb": a.get("size_mb"),
                "kind": a.get("kind"),
                "role": a.get("role"),
                "semantic_target": a.get("semantic_target"),
                "origin": a.get("origin"),
                "source": a.get("source"),
                "producer_subgraph": a.get("producer_subgraph"),
                "producer_tool": a.get("producer_tool"),
                "description": a.get("description"),
                "parent_id": parent_id,
                "parent_path": parent.get("path") if isinstance(parent, dict) else None,
                "source_image_path": a.get("source_image_path"),
                "file_format": a.get("file_format"),
                "label_space": a.get("label_space"),
            }
            context.append(
                {
                    key: value
                    for key, value in item.items()
                    if value not in (None, "", [])
                }
            )

        return context

    def _recent_execution_context_for_spec(
        self,
        state: SandboxExtendedState,
        *,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        messages = state.get("internal_messages") or []
        if not isinstance(messages, list):
            return []

        context: list[dict[str, Any]] = []
        for message in messages[-limit:]:
            context.append(
                {
                    "type": type(message).__name__,
                    "name": getattr(message, "name", None),
                    "content": self._message_content_preview(message),
                }
            )
        return context

    @staticmethod
    def _parse_first_json_object(raw: str) -> dict[str, Any] | None:
        if not isinstance(raw, str):
            return None

        start = raw.find("{")
        if start == -1:
            return None

        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        except Exception:
            return None

        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _truncate_text(value: Any, limit: int = 1200) -> str:
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

    @staticmethod
    def _one_line_summary(value: Any, limit: int = 260) -> str:
        text = " ".join(("" if value is None else str(value)).split())
        if not text:
            return "Unknown sandbox failure."
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _compact_value_for_summary(self, value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            return self._truncate_text(value, SANDBOX_SUMMARY_MAX_STRING_CHARS)

        if isinstance(value, list):
            if len(value) <= SANDBOX_SUMMARY_FULL_SCALAR_LIST_ITEMS and all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in value
            ):
                return [
                    self._compact_value_for_summary(item, depth + 1)
                    for item in value
                ]

            preview = [
                self._compact_value_for_summary(item, depth + 1)
                for item in value[:SANDBOX_SUMMARY_PREVIEW_ITEMS]
            ]
            compact: dict[str, Any] = {
                "count": len(value),
                "preview": preview,
            }
            if len(value) > SANDBOX_SUMMARY_PREVIEW_ITEMS:
                compact["omitted_count"] = len(value) - SANDBOX_SUMMARY_PREVIEW_ITEMS
            return compact

        if isinstance(value, dict):
            if depth >= 3:
                keys = list(value.keys())
                compact = {"keys": keys[:20]}
                if len(keys) > 20:
                    compact["omitted_key_count"] = len(keys) - 20
                return compact

            compact_dict: dict[str, Any] = {}
            for key, item in value.items():
                compact_dict[str(key)] = self._compact_value_for_summary(item, depth + 1)
            return compact_dict

        return self._truncate_text(value, SANDBOX_SUMMARY_MAX_STRING_CHARS)

    def _format_compact_payload_message(
        self,
        *,
        summary: str,
        payload: dict[str, Any],
        run_dir: Any,
    ) -> str:
        data_value = payload.get("data")
        data: dict[str, Any] = data_value if isinstance(data_value, dict) else {}
        meta_value = payload.get("meta")
        meta: dict[str, Any] = meta_value if isinstance(meta_value, dict) else {}

        result = data.get("result") if isinstance(data, dict) else None
        compact_result = self._compact_value_for_summary(result)
        compact_payload = {
            "kind": payload.get("kind"),
            "status": data.get("status") if isinstance(data, dict) else None,
            "task_type": meta.get("task_type"),
            "result": compact_result,
        }
        compact_payload = {
            key: value
            for key, value in compact_payload.items()
            if value not in (None, "", [], {})
        }

        preview_json = json.dumps(compact_payload, indent=2, default=str)
        preview_json = self._truncate_text(preview_json, SANDBOX_SUMMARY_MAX_JSON_CHARS)

        result_path = None
        if isinstance(run_dir, str) and run_dir:
            result_path = str(Path(run_dir) / "sandbox_result.json")

        parts = [
            summary,
            "Sandbox produced a structured result. The router should use this compact handoff, not the full raw payload.",
            f"```json\n{preview_json}\n```",
        ]
        if result_path:
            parts.append(f"Full structured result: `{result_path}`")
        return "\n\n".join(parts)

    def _compact_payload_error(self, payload: Any) -> dict[str, Any] | None:
        if not self._sandbox_payload_is_error(payload):
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None

        return {
            "kind": payload.get("kind"),
            "data": {
                key: data.get(key)
                for key in ("status", "error_type", "error_message", "input_file")
                if key in data
            },
            "meta": meta,
        }

    def _failure_entry_from_execution(
        self,
        state: SandboxExtendedState,
        *,
        stage: str,
        result_wrapper: dict[str, Any],
        explicit_error: str | None = None,
    ) -> dict[str, Any] | None:
        attempts = state.get("sandbox_attempts") or 0
        payload = result_wrapper.get("result") if isinstance(result_wrapper, dict) else None
        exit_code = result_wrapper.get("exit_code") if isinstance(result_wrapper, dict) else None
        stdout = result_wrapper.get("stdout") if isinstance(result_wrapper, dict) else ""
        stderr = result_wrapper.get("stderr") if isinstance(result_wrapper, dict) else ""
        run_dir = result_wrapper.get("run_dir") if isinstance(result_wrapper, dict) else None

        compact_payload = self._compact_payload_error(payload)
        if explicit_error:
            summary = explicit_error
            detail_obj: dict[str, Any] = {
                "stage": stage,
                "attempt": attempts,
                "error": explicit_error,
                "exit_code": exit_code,
                "run_dir": run_dir,
            }
        elif compact_payload is not None:
            data = compact_payload.get("data") or {}
            meta = compact_payload.get("meta") or {}
            summary_bits = [
                meta.get("summary"),
                data.get("error_type"),
                data.get("error_message"),
            ]
            summary = ": ".join(str(x) for x in summary_bits if x)
            detail_obj = {
                "stage": stage,
                "attempt": attempts,
                "sandbox_result_error": compact_payload,
                "exit_code": exit_code,
                "run_dir": run_dir,
            }
        elif isinstance(exit_code, int) and exit_code != 0:
            summary = f"Sandbox process exited with code {exit_code}."
            detail_obj = {
                "stage": stage,
                "attempt": attempts,
                "exit_code": exit_code,
                "run_dir": run_dir,
                "stderr": stderr,
                "stdout": stdout,
            }
        elif payload is None and isinstance(exit_code, int):
            summary = "Sandbox did not write a structured sandbox_result.json payload."
            detail_obj = {
                "stage": stage,
                "attempt": attempts,
                "exit_code": exit_code,
                "run_dir": run_dir,
                "stderr": stderr,
                "stdout": stdout,
            }
        else:
            return None

        return {
            "attempt": attempts,
            "stage": stage,
            "summary": self._one_line_summary(summary),
            "detail": self._truncate_text(json.dumps(detail_obj, indent=2, default=str), 12000),
        }

    def _append_failure_entry(
        self,
        state: SandboxExtendedState,
        entry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not entry:
            return {}
        history = list(state.get("sandbox_failure_history") or [])
        history.append(entry)
        return {"sandbox_failure_history": history[-10:]}

    def _failure_context_for_llm(
        self,
        state: SandboxExtendedState,
        *,
        include_code: bool = False,
    ) -> str:
        """
        Compact diagnostic context for repair prompts.
        Keeps enough signal for the LLM without replaying full histories or outputs.
        """
        parts: list[str] = []

        attempts = state.get("sandbox_attempts") or 0
        if attempts:
            parts.append(f"attempts_so_far: {attempts}")

        history = state.get("sandbox_failure_history") or []
        if isinstance(history, list) and history:
            previous = [h for h in history[:-1] if isinstance(h, dict)]
            latest = history[-1] if isinstance(history[-1], dict) else None

            if previous:
                lines = []
                for item in previous[-6:]:
                    attempt = item.get("attempt", "?")
                    stage = item.get("stage", "unknown")
                    summary = self._one_line_summary(item.get("summary"))
                    lines.append(f"- attempt {attempt} ({stage}): {summary}")
                parts.append("previous_error_summaries:\n" + "\n".join(lines))

            if latest:
                parts.append(
                    "latest_error_full:\n"
                    + str(latest.get("detail") or latest.get("summary") or "")
                )

            if include_code:
                code = state.get("sandbox_code") or ""
                if code:
                    parts.append("previous_code_excerpt:\n" + self._truncate_text(code, 1600))

            return "\n\n".join(parts).strip()

        error = state.get("sandbox_error")
        if error:
            parts.append("current_error:\n" + self._truncate_text(error, 1000))

        res = state.get("sandbox_result") or {}
        if isinstance(res, dict):
            exit_code = res.get("exit_code")
            if exit_code is not None:
                parts.append(f"previous_exit_code: {exit_code}")

            payload = res.get("result")
            if isinstance(payload, dict):
                payload_data = payload.get("data")
                if isinstance(payload_data, dict) and payload_data.get("status") == "error":
                    compact_payload = {
                        "kind": payload.get("kind"),
                        "data": {
                            key: payload_data.get(key)
                            for key in ("status", "error_type", "error_message", "input_file")
                            if key in payload_data
                        },
                        "meta": payload.get("meta"),
                    }
                    parts.append(
                        "previous_sandbox_result_error:\n"
                        + self._truncate_text(json.dumps(compact_payload, indent=2, default=str), 1200)
                    )

            stdout = res.get("stdout") or ""
            stderr = res.get("stderr") or ""
            if stderr:
                parts.append("previous_stderr:\n" + self._truncate_text(stderr, 1200))
            if stdout:
                parts.append("previous_stdout:\n" + self._truncate_text(stdout, 800))
            if res.get("run_dir"):
                parts.append(f"previous_run_dir: {res.get('run_dir')}")

        if include_code:
            code = state.get("sandbox_code") or ""
            if code:
                parts.append("previous_code_excerpt:\n" + self._truncate_text(code, 1600))

        return "\n\n".join(parts).strip() or "No previous failure context is available."

    @staticmethod
    def _sandbox_payload_is_error(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        data = payload.get("data")
        return isinstance(data, dict) and data.get("status") == "error"

    # -------------------------------------------------------------------------
    # Helper for local subgraph memory
    # -------------------------------------------------------------------------

    def _append_sandbox_scratchpad(
        self,
        state: SandboxExtendedState,
        *new_msgs: Any,
    ) -> dict[str, Any]:
        """
        Append subgraph-private messages to the private scratchpad.
        This is sandbox self-talk, not parent/router communication.
        """
        history = [
            msg
            for msg in (state.get("private_scratchpad") or [])
            if str(getattr(msg, "name", "") or "").startswith(f"{self.key}_")
        ]
        safe_msgs: list[BaseMessage] = []
        for msg in new_msgs:
            if msg is None:
                continue
            if isinstance(msg, BaseMessage):
                if not str(getattr(msg, "name", "") or "").startswith(f"{self.key}_"):
                    msg.name = f"{self.key}_scratchpad_record"
                safe_msgs.append(msg)
                continue
            if isinstance(msg, BaseModel):
                content = msg.model_dump_json(indent=2)
            else:
                content = self._truncate_text(msg, 4000)
            safe_msgs.append(
                AIMessage(content=content, name=f"{self.key}_scratchpad_record")
            )

        history = list(history) + safe_msgs

        MAX_HISTORY = 30
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        return {"private_scratchpad": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *history]}

    # -------------------------------------------------------------------------
    # Node 1: build_sandbox_spec
    # -------------------------------------------------------------------------

    def build_sandbox_spec(self, state: SandboxExtendedState) -> dict[str, Any]:
        """
        Interpret the router_subgraph_task (if present) + attachments into a JSON spec
        for the sandbox.

        Uses private_scratchpad as subgraph-local context memory.
        """
        session_value = state.get("session")
        session: dict[str, Any] = dict(session_value) if isinstance(session_value, Mapping) else {}
        attachments = state.get("attachments") or []
        session_path_value = session.get("path")
        session_path = session_path_value if isinstance(session_path_value, str) else ""
        workspace_root = self._workspace_root(state)
        session_data_path = self._session_data_path(session_path) if session_path else "N/A"

        sandbox_history = state.get("private_scratchpad") or []
        previous_failure_context = (
            state.get("sandbox_repair_context")
            or self._failure_context_for_llm(state)
        )
        has_previous_failure = previous_failure_context != "No previous failure context is available."

        # Prefer the router-provided subgraph task as the "user request"
        router_task = state.get("router_subgraph_task")
        if router_task:
            user_msg_text = router_task
        else:
            # Fallback: last HUMAN message in the conversation, not last message overall
            user_msg_text = ""
            for m in reversed(state.get("messages") or []):
                if isinstance(m, HumanMessage):
                    user_msg_text = getattr(m, "content", str(m))
                    break

            if not user_msg_text and state.get("messages"):
                # ultra-conservative fallback
                user_msg_text = getattr(state["messages"][-1], "content", str(state["messages"][-1]))

        files_json = json.dumps(
            self._attachment_context_for_spec(attachments),
            indent=2,
            default=str,
        )
        recent_execution_context_json = json.dumps(
            self._recent_execution_context_for_spec(state),
            indent=2,
            default=str,
        )

        _domain_tool_guard = "" if self.unrestricted else """<domain_tool_guard>
Do NOT create a SPEC that implements, repairs, bypasses, or substitutes for Hotelling T2/KNN
classification, nearest-neighbor ranking, weighted disease-class voting, or KNN feature-driver
analysis. Those operations must be handled by the quantitative_imaging_analysis subgraph using
the dedicated find_similar_patients_by_block_hotelling_t2 tool.
For lacunar morphometry feature-extraction tasks, the canonical source of truth is
calculate_lacunae_parameters. Do not create a SPEC that derives lacunar morphometry values,
counts, distributions, percentiles, filters, summaries, surface area, volume filtering, or
morphometry statistics directly from a segmentation mask unless calculate_lacunae_parameters
has already succeeded and produced the canonical per-lacuna morphometry CSV. Do not use
count_lacunae outputs as a substitute for morphometry extraction; count_lacunae counts
segmentation connected components and may not match the canonical morphometry population.
If the sandbox task asks for a custom replacement after calculate_lacunae_parameters failed,
return a SPEC with task_type="unsupported_domain_tool_bypass" and explain that
calculate_lacunae_parameters must be retried or its failure reported.
Allowed: aggregate, validate, summarize, or plot already-produced JSON payload artifacts from
find_similar_patients_by_block_hotelling_t2. Do not recompute Hotelling T2 distances or invent
neighbor lists.
For those KNN payloads, use the actual schema:
- query_patient_id is the payload's query patient; if absent, use query_block_id or the
  query_block_csv basename before "_step".
- class_inference.class_votes is a list of dicts with class_label and weight.
- top_block_matches is a list of neighbors with patient_id, class_label, t2,
  hotelling_t2_distance, and similarity_weight.
For the LOO manifest, patient labels are under manifest["samples"], each with patient_id,
sample_identifier, and class_label. Do not treat top-level manifest metadata as samples.
For ROC/AUC, do not require scikit-learn; compute the rank-based binary AUC or trapezoidal
ROC curve directly from y_true/y_score when sklearn is unavailable.
If the requested sandbox task asks for custom Hotelling/KNN code, return a SPEC with
task_type="unsupported_domain_tool_bypass" and a failure_modes/validation explanation instead of
an algorithm for KNN.
</domain_tool_guard>"""

        # High-level instructions as SystemMessage
        system_msg = SystemMessage(
            content=(
                "You are a planner for a Python sandbox. "
                "You must output ONLY a single JSON object describing a sandbox spec."
                if self.unrestricted else
                "You are a planner for a restricted Python sandbox. "
                "You must output ONLY a single JSON object describing a sandbox spec."
            ),
            name=f"{self.key}_planner_system",
        )

        # Concrete details and schema as HumanMessage
        human_msg = HumanMessage(
            content=f"""
<task>
Convert the user request and available files into a JSON SPEC for a small Python program
that will run in a sandboxed directory inside the current session path.
</task>
<contract>
This SPEC is the primary implementation contract for the coder.
The coder will rely on the SPEC much more than on the original user request.
Therefore the SPEC must carry the planning burden and must be precise, technical, and execution-ready.
Do not leave crucial implementation details implicit when they can be made explicit in structured fields.
The SPEC is an open-ended reasoning/planning object, not a fixed minimal schema.
Use as many task-specific fields as needed to preserve the algorithm, implementation details,
output shape, numerical definitions, validation rules, and assumptions.
</contract>
<filesystem_policy>
SESSION PATH (writable root for this session): {session_path}
READ-ONLY WORKSPACE ROOT: {workspace_root}
SESSION DATA FOLDER (input-only, not for outputs): {session_data_path}

You MUST ONLY reference files and directories that are:
- under the session path, except that session/data is input-only
- under the workspace root

Do NOT invent arbitrary absolute paths outside these allowed roots.
If the user gives a relative path beginning with data/, treat it as a session data read-only input.
For other relative paths, resolve relative to the workspace root before falling back to the session root.
Plan outputs only in the sandbox run directory or another writable non-data session output directory.
</filesystem_policy>
<filesystem_api_policy>
Sandbox code cannot import os or call os.walk. If the user request, router task, or previous
failure context mentions os.walk, encode the intended traversal as pathlib.Path.rglob(...) or
pathlib.Path.iterdir(...) in the SPEC instead. Do not ask generated code to use Path.walk;
it is not portable across supported Python versions.
</filesystem_api_policy>
{_domain_tool_guard}
<available_inputs>
CURRENT ATTACHMENTS:
{files_json}
</available_inputs>
<recent_execution_context>
{recent_execution_context_json}
</recent_execution_context>
<user_request>
{user_msg_text}
</user_request>
<path_provenance_rules>
- CURRENT ATTACHMENTS are the authoritative inventory of files already registered
  in the session for downstream use.
- A path that appears only in the user request is just a user-mentioned path. Do
  not promote it to a stronger semantic role such as prediction, ground truth,
  generated output, or current session output unless the wording explicitly says
  that exact path has that role.
- If the request asks for a file "from current session outputs", select it from
  CURRENT ATTACHMENTS or from explicit produced-file evidence in RECENT EXECUTION
  CONTEXT. Do not substitute a raw/source input path merely because it is related
  to the requested output.
- If no path has enough provenance for a required semantic role, represent that
  uncertainty explicitly in the SPEC using fields such as missing_required_inputs,
  unresolved_roles, validation, or failure_modes. Do not silently guess.
- For each target_files entry, include role/path_source/evidence fields when the
  role matters. Suggested path_source values are attachment, user_request,
  recent_execution_context, or unresolved.
</path_provenance_rules>
<spec_quality_bar>
- The SPEC must describe the task in technical terms, not only as a natural-language paraphrase.
- Encode every important instruction explicitly inside JSON fields whenever possible.
- If the task involves computation, transformation, validation, or structured output,
  represent the full contract explicitly in task-specific top-level fields.
- Include output limits and relevance criteria: identify which results are essential
  for the router/user, which large intermediate details should be counted or previewed,
  and when full raw payloads or full file contents are explicitly requested.
- If the user specified technical constraints, required structure, or correctness conditions,
  encode them explicitly. Do not collapse them into a generic task_type.
- If a detail is genuinely unknown, leave it out rather than inventing it.
- Be conservative about paths, but thorough about the technical contract.
- The coder must be able to implement the task from the SPEC without rediscovering the user's intent.
</spec_quality_bar>
<schema>
Return one JSON object. It is open-ended.
Use these fields when relevant, and add more fields when they make the implementation clearer:
- task_type: short string label only
- target_files: list of objects with path, kind, source_root, and optional notes
- implementation_plan: ordered list of concrete coding steps
- algorithm: exact algorithm, formulas, thresholds, and selection rules
- data_loading: how to discover/read inputs, expected formats, and library hints
- computation: numerical operations and edge-case behavior
- output_contract: required JSON keys, value types, units, precision, and nesting
- output_relevance: which findings must be reported, preview limits for large values,
  and a rule to avoid full file-content or full-payload dumps unless explicitly requested
- validation: checks the script should perform before trusting results
- failure_modes: expected failures the script should catch and report
- constraints: resource/output limits relevant to this task
- params: optional compact machine-readable parameters, not a dumping ground for the whole plan
</schema>
<example_quality_bar>
For a request like:
"Estimate SNR of data/T1_S7step1_Z0 using mean(top 10% intensity voxels) divided by
std(bottom 10% intensity voxels), percentiles over all voxels, return snr/signal_estimate/noise_estimate"
an acceptable SPEC must preserve:
- the directory path and that it is a volume input
- voxel loading/discovery strategy
- percentile definition over all voxels
- top-10% and bottom-10% masks
- signal = mean(top 10%) and noise = std(bottom 10%)
- snr = signal / noise
- exact output keys and float types
It is not acceptable to return only task_type="data_analysis" plus target_files.
</example_quality_bar>
<previous_failure_context>
{previous_failure_context if has_previous_failure else "None. This is the first spec for the current sandbox task."}
</previous_failure_context>
<repair_guidance>
If previous failure context is present, revise the SPEC only when the failure suggests the current SPEC
was ambiguous, missing a crucial requirement, used a wrong/unsafe path, chose the wrong task_type,
or forced a bad assumption. Keep valid parts of the existing task contract.
Do not overfit the SPEC to a coding mistake that can be fixed in Python.
</repair_guidance>
            """.strip(),
            name=f"{self.key}_planner_user",
        )

        history_prefix = list(sandbox_history) if has_previous_failure else []
        messages_for_llm = history_prefix + [system_msg, human_msg]

        raw = ""
        resp: Any = None
        parsed_spec: dict[str, Any] | None = None
        try:
            structured_llm = self.llm.with_structured_output(SandboxSpecModel)
            resp = structured_llm.invoke(messages_for_llm)
            if isinstance(resp, SandboxSpecModel):
                parsed_spec = resp.model_dump()
            elif isinstance(resp, dict):
                parsed_spec = resp
            raw = json.dumps(parsed_spec, indent=2, default=str) if parsed_spec is not None else ""
        except Exception as e:
            if is_ollama_usage_limit_error(e):
                raise
            resp = self.llm.invoke(messages_for_llm)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed_spec = self._parse_first_json_object(raw)

        if parsed_spec is None:
            message = f"Failed to parse sandbox spec JSON. Raw LLM output: {raw}"
            update = {
                "sandbox_error": message,
                "sandbox_spec": None,
                "sandbox_attempts": state.get("sandbox_attempts") or 0,
            }
            update.update(
                self._append_failure_entry(
                    state,
                    {
                        "attempt": state.get("sandbox_attempts") or 0,
                        "stage": "spec",
                        "summary": self._one_line_summary(message),
                        "detail": self._truncate_text(message, 4000),
                    },
                )
            )
            update |= self._append_sandbox_scratchpad(state, system_msg, human_msg, resp)
            return update

        try:
            spec = parsed_spec
            spec = self._normalize_sandbox_spec(spec, session_path, workspace_root)
            error = None
        except Exception as e:
            spec = None
            error = f"Error loading sandbox spec JSON: {e}; raw={raw}"

        update = {
            "sandbox_spec": spec,
            "sandbox_error": error,
            "sandbox_attempts": state.get("sandbox_attempts") or 0,
        }
        if error:
            update.update(
                self._append_failure_entry(
                    state,
                    {
                        "attempt": state.get("sandbox_attempts") or 0,
                        "stage": "spec",
                        "summary": self._one_line_summary(error),
                        "detail": self._truncate_text(error, 4000),
                    },
                )
            )
        update |= self._append_sandbox_scratchpad(state, system_msg, human_msg, resp)
        return update

    # -------------------------------------------------------------------------
    # Node 2: generate_sandbox_code
    # -------------------------------------------------------------------------

    def generate_sandbox_code(self, state: SandboxExtendedState) -> dict[str, Any]:
        """
        Generate Python code based on sandbox_spec.

        Uses private_scratchpad as subgraph-local context memory.
        Includes refinement hints if previous validation or execution failed.
        """
        spec = state.get("sandbox_spec")
        session_value = state.get("session")
        session: dict[str, Any] = dict(session_value) if isinstance(session_value, Mapping) else {}
        if not spec or not session.get("path"):
            attempts = (state.get("sandbox_attempts") or 0) + 1
            message = "Missing sandbox_spec or session.path for code generation."
            update = {
                "sandbox_error": "Missing sandbox_spec or session.path for code generation.",
                "sandbox_attempts": attempts,
                "sandbox_code": "",
            }
            update.update(
                self._append_failure_entry(
                    state,
                    {
                        "attempt": attempts,
                        "stage": "codegen",
                        "summary": self._one_line_summary(message),
                        "detail": message,
                    },
                )
            )
            return update

        spec_json = json.dumps(spec, indent=2)
        session_path = session["path"]
        workspace_root = self._workspace_root(state)
        # Show the logical session data path WITHOUT resolving the symlink.
        # Resolving it produces a path identical to workspace_root/data, which
        # causes the LLM to double-prefix relative "data/..." paths.
        session_data_path = str(Path(session_path).expanduser() / "data")

        _codegen_domain_restrictions = "" if self.unrestricted else (
            "Do not implement Hotelling T2/KNN classification, nearest-neighbor ranking, weighted disease-class\n"
            "voting, or KNN feature-driver analysis in Python from raw data. You may aggregate, validate,\n"
            "summarize, or plot already-produced JSON payload artifacts from find_similar_patients_by_block_hotelling_t2.\n"
            "Do not implement lacunar morphometry feature extraction, surface-area computation, volume filtering,\n"
            "or morphometry statistics directly from segmentation masks. Use only canonical per-lacuna morphometry\n"
            "CSVs already produced by calculate_lacunae_parameters for lightweight post-processing. If the task\n"
            "requests a custom replacement for calculate_lacunae_parameters, write a sandbox_result.json error\n"
            "explaining that calculate_lacunae_parameters must be retried or its failure reported.\n"
            "For those KNN payloads, use query_patient_id/query_block_id/query_block_csv for the query, use\n"
            "class_inference.class_votes as a list of {class_label, weight} dicts, and use top_block_matches\n"
            "entries for neighbor patient_id/class_label/t2/hotelling_t2_distance/similarity_weight. LOO manifest\n"
            "labels are in manifest[\"samples\"], not top-level manifest metadata. Compute ROC/AUC manually if\n"
            "sklearn is unavailable.\n"
            "If the SPEC requests custom KNN work without such payload artifacts, write a sandbox_result.json\n"
            "error explaining that the dedicated find_similar_patients_by_block_hotelling_t2 tool must be used\n"
            "through quantitative_imaging_analysis first.\n"
        )

        failure_context = (
            state.get("sandbox_repair_context")
            or self._failure_context_for_llm(state, include_code=True)
        )
        repair_message = state.get("sandbox_repair_message")
        if isinstance(repair_message, str) and repair_message.strip():
            failure_context = (
                f"repair_plan:\n{repair_message.strip()}\n\n"
                f"{failure_context}"
            )
        has_failure_context = failure_context != "No previous failure context is available."

        system_msg = SystemMessage(
            content="""
You write SAFE, RESTRICTED Python scripts to run in an isolated directory.

You must obey these rules:

- You may NOT use:
  - os, subprocess, shutil, requests, socket, http, urllib
- You may import only: json, csv, math, statistics, pathlib, typing, collections,
  numpy, pandas, nibabel, matplotlib, seaborn, sklearn, and common image libs
  if you want (skimage, tifffile).
- For filesystem/path work, use pathlib.Path. Do not import or use os/os.path;
  os is unnecessary in generated sandbox scripts and is forbidden.
- For recursive directory traversal, use Path.rglob(pattern), for example:
  `for path in root.rglob("*.csv")`. For flat traversal, use Path.iterdir().
  Never use os.walk or Path.walk in generated sandbox code.
- For NIfTI volumes, inspect shape and dtype with nibabel headers/dataobj before
  reading pixels. Large SR-microCT volumes may expand substantially in memory:
  never call get_fdata(), np.asarray(img.dataobj), np.array(img.dataobj), or
  data.reshape(-1) on a whole large NIfTI. Prefer img.dataobj slice/chunk reads.
- For histogram, threshold, Otsu, BV/TV, min/max, mean/std, or percentile work on
  a large NIfTI, stream over Z slices or small chunks and keep only running counts
  or running statistics. If the task permits a representative central Z subset,
  process that subset slice-by-slice and report n_slices_used.
- For Otsu + BV/TV specifically, compute a compact intensity histogram from the
  selected slices/chunks, derive the Otsu threshold from that histogram, then make
  a second streaming pass to count voxels above threshold. Do not build a dense
  whole-volume boolean mask.
- You may READ files only from:
  - The session root path provided in the problem description.
  - The session data folder provided in the problem description, for inputs only.
  - The workspace root path provided in the problem description.
- Never create, modify, move, delete, or write derived outputs under session/data,
  even when the input file was read from there.
- The workspace root is STRICTLY READ-ONLY:
  - Do NOT create, modify, move, or delete any file under the workspace root.
- You must NOT create or modify files except:
  - Reading the specified input files within the allowed roots.
  - Writing 'sandbox_result.json' in the current working directory (the sandbox run directory).
  - Writing requested small derived artifacts, such as PNG/CSV/JSON summaries, in the current working directory.
    If the spec names an output path outside the current working directory, keep only the basename and write it here.

Your script will run in a working directory INSIDE the session path.
You will receive no command line args.

Your script MUST:

1. Read and interpret the SPEC JSON (you will receive it inline in the prompt).
2. Perform the task described in the spec.
3. Write generated artifacts and a JSON file named 'sandbox_result.json' in the current working directory.
4. If execution succeeds, sandbox_result.json must be a dict like:

{
  "kind": "generic_result",
  "data": {
    "status": "ok",
    "result": ...  // task-specific result, JSON-serializable, following the SPEC
  },
  "meta": {
     "task_type": "<from spec>",
     "summary": "<1-line human readable summary>"
  }
}

5. If execution fails for any reason, the script must STILL write sandbox_result.json like:

{
  "kind": "log_only",
  "data": {
    "status": "error",
    "error_type": "<exception class name>",
    "error_message": "<exception message>",
    "input_file": "<resolved primary input path if known>"
  },
  "meta": {
    "task_type": "<from spec>",
    "summary": "<1-line failure summary>"
  }
}

The SPEC is responsible for defining the task-specific technical contract.
If the SPEC defines required fields, structure, invariants, limits, or validation conditions,
your code must satisfy them exactly.
Do not replace missing planning detail with guesses when the SPEC already constrains the task.
Expected input/load/parse/validation errors must be handled inside the script.
Do not let expected data-loading failures escape as uncaught exceptions.
            """.strip(),
            name=f"{self.key}_codegen_system",
        )

        human_msg = HumanMessage(
        content=f"""
<task>
Produce ONLY raw executable Python source code.
Your entire response must be a complete Python script and nothing else.
</task>
<forbidden_output>
Do not return:
- Markdown
- code fences
- JSON
- arrays or wrapper objects such as [{{"type":"text","text":"..."}}]
- explanations
- logs
- summaries
- natural language before or after the code
</forbidden_output>
<execution_contract>
The script will run with no command-line arguments.
The script must always write a file named sandbox_result.json in the current working directory.
Write requested small artifacts, such as PNG/CSV/JSON summaries, in the current working directory too.
If the spec names an output path outside the current working directory, use only that file's basename here.
The script must ALWAYS be produced, even if:
- an input file is missing
- an input file is empty
- an input file is unreadable
- an input file is invalid
- the requested computation cannot be completed exactly
</execution_contract>
<failure_policy>
Do not describe failures outside the script.
Handle failures inside Python code.
If any step fails, the script must catch the exception and write sandbox_result.json with:
{{
  "kind": "log_only",
  "data": {{
    "status": "error",
    "error_type": "<exception class name>",
    "error_message": "<exception message>",
    "input_file": "<resolved path>"
  }},
  "meta": {{
    "task_type": "<from spec>",
    "summary": "<1-line failure summary>"
  }}
}}
</failure_policy>
<success_policy>
If processing succeeds, write sandbox_result.json with:
{{
  "kind": "generic_result",
  "data": {{
    "status": "ok",
    "result": ...  // task-specific result following the SPEC
  }},
  "meta": {{
    "task_type": "<from spec>",
    "summary": "<1-line success summary>"
  }}
}}
</success_policy>
<result_size_policy>
The graph handoff needs concise, relevant results, not raw dumps.
Before writing data.result, decide what information is needed by the router/user to understand
the outcome or choose the next step.
Do not put full file contents, full raw payloads, exhaustive workspace scans, source trees,
directory inventories, full CSV rows, or very large row/list/object dumps inside data.result
unless the SPEC or user request explicitly requires that exact full content.
For large inputs or discovery-style tasks, report counts, schema/columns, important statistics,
the most relevant matches, short previews, warnings, recommendations, and paths to persisted
outputs instead of every item.
If full details may be useful for debugging, write or reference them as a file artifact/path;
keep sandbox_result.json suitable for graph communication.
</result_size_policy>
<library_policy>
You may use:
json, csv, math, statistics, pathlib, typing, collections, numpy, pandas, nibabel,
matplotlib, seaborn, sklearn, skimage, tifffile
Use libraries only when the SPEC requires them.
For filesystem/path work, use pathlib.Path. Do not import or use os/os.path;
os is unnecessary in generated sandbox scripts and is forbidden.
For recursive directory traversal, use Path.rglob(pattern); for flat traversal,
use Path.iterdir(). Never use os.walk or Path.walk.
</library_policy>
<spec>
SPEC JSON:
{spec_json}
</spec>
<environment>
SESSION ROOT PATH: {session_path}
SESSION DATA FOLDER PATH (input-only): {session_data_path}
WORKSPACE ROOT PATH (READ-ONLY): {workspace_root}
</environment>
<execution_notes>
Treat the SPEC as the implementation contract.
Do not assume there is hidden planning context beyond what the SPEC states.
Read the entire SPEC JSON, not only task_type/target_files/params.
Important implementation instructions may appear in top-level fields such as implementation_plan,
algorithm, data_loading, computation, output_contract, validation, failure_modes, or custom fields.
Follow technical details in params literally when they are present, but do not ignore other fields.
{_codegen_domain_restrictions}
CRITICAL PATH RULE: When `target_files[*].path` in the SPEC JSON is already an
absolute path (starts with `/`), use it exactly as given — do NOT re-derive it
by joining SESSION DATA FOLDER PATH or WORKSPACE ROOT PATH with any relative
fragment from the task description. The SPEC paths are pre-resolved; re-joining
them will produce double-prefixed paths (e.g. `.../data/data/...`) that do not exist.
Keep session/data and workspace paths read-only; write the required result
only to the current sandbox run directory.
Catch file loading and parsing errors explicitly.
If the SPEC mentions os.walk, implement the same traversal with pathlib.Path.rglob
or pathlib.Path.iterdir instead.
If a library-specific loading error occurs, handle it and write sandbox_result.json.
Do not invent data when the input cannot be read.
</execution_notes>
<diagnostic_context>
The following is diagnostic context from a previous failed run.
It does NOT change the required response format.
Use it only to improve the Python script. Generate a corrected script when context is present.
{failure_context if has_failure_context else "No prior diagnostic context."}
</diagnostic_context>
<response_prefix_example>
import json
from pathlib import Path
</response_prefix_example>
        """.strip(),
        name=f"{self.key}_codegen_user",
    )

        # IMPORTANT: do *not* prepend sandbox_history here
        messages_for_llm = [system_msg, human_msg]

        resp = self.llm.invoke(messages_for_llm)
        raw_code = getattr(resp, "content", str(resp))

        code = self._extract_code_block(raw_code).strip()
        attempts = (state.get("sandbox_attempts") or 0) + 1

        if not code:
            message = "Sandbox LLM produced empty code instead of a Python script."
            update = {
                "sandbox_code": "",
                "sandbox_error": message,
                "sandbox_attempts": attempts,
            }
            update.update(
                self._append_failure_entry(
                    state,
                    {
                        "attempt": attempts,
                        "stage": "codegen",
                        "summary": self._one_line_summary(message),
                        "detail": message,
                    },
                )
            )
            update |= self._append_sandbox_scratchpad(state, system_msg, human_msg, resp)
            return update

        update = {
            "sandbox_code": code,
            "sandbox_error": None,
            "sandbox_attempts": attempts,
        }
        update |= self._append_sandbox_scratchpad(state, system_msg, human_msg, resp)
        return update


    # -------------------------------------------------------------------------
    # Node 2b: choose repair strategy after validation/execution failure
    # -------------------------------------------------------------------------

    def decide_sandbox_repair(self, state: SandboxExtendedState) -> dict[str, Any]:
        """
        Let the LLM choose whether a failure needs code regeneration or spec regeneration.
        This prevents repeatedly coding against a flawed SPEC when the original plan was wrong.
        """
        attempts = state.get("sandbox_attempts") or 0
        failure_context = self._failure_context_for_llm(state, include_code=True)
        if attempts >= MAX_SANDBOX_ATTEMPTS:
            return {
                "sandbox_repair_action": "give_up",
                "sandbox_repair_context": failure_context,
            }

        spec = state.get("sandbox_spec")
        spec_json = json.dumps(spec, indent=2, default=str) if spec else "null"

        system_msg = SystemMessage(
            content=(
                "You are a repair router for a restricted Python sandbox. "
                "Choose the smallest useful next repair step. "
                "Return ONLY a JSON object."
            ),
            name=f"{self.key}_repair_router_system",
        )
        human_msg = HumanMessage(
            content=f"""
<task>
Decide how to repair the failed sandbox run.
</task>
<actions>
- regenerate_code: use when the SPEC is still valid and the failure is caused by Python syntax,
  imports, missing exception handling, bad parsing/loading code, or an implementation bug.
- regenerate_spec: use when the failure suggests the SPEC is wrong, ambiguous, incomplete,
  points at the wrong/unsafe path, uses the wrong task_type, or requires assumptions the coder
  should not have made.
- give_up: use only if no useful repair is likely within the remaining attempt budget.
</actions>
<repair_message>
If action is regenerate_code or regenerate_spec, include repair_message: a concise natural-language
explanation of why retrying is useful and what will change in the next attempt. Do not include stack
traces. If action is give_up, repair_message should be empty.
</repair_message>
<withdrawal_message>
If action is give_up, include withdrawal_message: a concise natural-language explanation for
the outer router. It should say what failed, why another sandbox retry is unlikely to help,
and what kind of next step or missing input would be useful. Do not include stack traces.
For regenerate_code or regenerate_spec, withdrawal_message should be empty.
</withdrawal_message>
<current_spec>
{spec_json}
</current_spec>
<failure_context>
{failure_context}
</failure_context>
<output_schema>
{{
  "action": "regenerate_code|regenerate_spec|give_up",
  "reason": "brief machine-oriented reason",
  "repair_message": "natural-language retry plan, only when action regenerates something",
  "withdrawal_message": "natural-language router handoff, only when action is give_up"
}}
</output_schema>
            """.strip(),
            name=f"{self.key}_repair_router_user",
        )

        raw = ""
        resp: Any = None
        parsed: dict[str, Any] | None = None
        try:
            structured_llm = self.llm.with_structured_output(SandboxRepairDecision)
            resp = structured_llm.invoke([system_msg, human_msg])
            if isinstance(resp, SandboxRepairDecision):
                parsed = resp.model_dump()
            elif isinstance(resp, dict):
                parsed = resp
            raw = json.dumps(parsed, default=str) if parsed is not None else ""
        except Exception as e:
            if is_ollama_usage_limit_error(e):
                raise
            resp = self.llm.invoke([system_msg, human_msg])
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = self._parse_first_json_object(raw)

        action = "regenerate_code"
        reason = "Defaulted to code regeneration after an unparseable repair decision."
        repair_message = ""
        withdrawal_message = ""
        if isinstance(parsed, dict):
            candidate = parsed.get("action")
            if candidate in {"regenerate_code", "regenerate_spec", "give_up"}:
                action = candidate
                reason = str(parsed.get("reason") or "")
                repair_message = str(parsed.get("repair_message") or "")
                withdrawal_message = str(parsed.get("withdrawal_message") or "")

        update = {
            "sandbox_repair_action": action,
            "sandbox_repair_reason": reason,
            "sandbox_repair_context": failure_context,
        }
        if action in {"regenerate_code", "regenerate_spec"} and repair_message.strip():
            update["sandbox_repair_message"] = repair_message.strip()
        if action == "give_up" and withdrawal_message.strip():
            update["sandbox_withdrawal_summary"] = withdrawal_message.strip()
        update |= self._append_sandbox_scratchpad(state, system_msg, human_msg, resp)
        return update


    # -------------------------------------------------------------------------
    # Node 3: validate_sandbox_code
    # -------------------------------------------------------------------------

    def _validate_ast(self, code: str) -> str | None:
        """
        Return None if code is acceptable, or a string error message if not.
        Very basic AST-based validator.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error in sandbox code: {e}"

        forbidden_names = {
            "os",
            "subprocess",
            "requests",
            "socket",
            "http",
            "urllib",
            "shutil",
        }
        forbidden_calls = {"system", "popen"}

        for node in ast.walk(tree):
            # Forbid dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in forbidden_names:
                        return f"Forbidden import in sandbox code: {name}"

            # Very simple detection of dangerous calls: something.system()
            if isinstance(node, ast.Attribute):
                if node.attr == "walk":
                    return "Unsupported filesystem traversal: use pathlib.Path.rglob(...) or Path.iterdir(), not os.walk or Path.walk."
                if isinstance(node.value, ast.Name) and node.attr in forbidden_calls:
                    return f"Forbidden call: {node.value.id}.{node.attr}"

        return None

    @staticmethod
    def _has_meaningful_script_body(tree: ast.AST) -> bool:
        """
        Reject responses that are parseable Python but not an executable script in practice,
        such as lone string/docstring output, numeric literals, dict/list literals, or wrappers.
        """
        if not isinstance(tree, ast.Module):
            return False

        meaningful_stmt_types = (
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.With,
            ast.AsyncWith,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            ast.Try,
        )

        for node in tree.body:
            if isinstance(node, meaningful_stmt_types):
                return True
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                return True

        return False

    def validate_sandbox_code(self, state: SandboxExtendedState) -> dict[str, Any]:
        raw_code = state.get("sandbox_code") or ""
        code = self._extract_code_block(raw_code).strip()

        # Keep the cleaned version in state, so execute_sandbox_code sees the same thing
        # (we still avoid mutating state directly)
        # DON'T: state["sandbox_code"] = code
        update: dict[str, Any] = {
            "sandbox_code": code
        }

        def fail(message: str) -> dict[str, Any]:
            update["sandbox_error"] = message
            entry = {
                "attempt": state.get("sandbox_attempts") or 0,
                "stage": "validation",
                "summary": self._one_line_summary(message),
                "detail": message,
            }
            update.update(self._append_failure_entry(state, entry))
            return update

        if not code:
            return fail("Sandbox LLM produced empty code instead of a Python script.")

        # Heuristic: if the response looks like a pure JSON / dict spec, treat as error
        # (it's extremely unlikely we *really* want a script that is just a dict literal).
        stripped = code.lstrip()
        if stripped.startswith("{") and '"task_type"' in stripped:
            return fail(
                "Sandbox LLM returned a JSON spec instead of Python code. "
                "Regenerate code that is an executable script, not JSON."
            )

        if stripped.startswith("[") and (
            ('"type"' in stripped and '"text"' in stripped)
            or ("'type'" in stripped and "'text'" in stripped)
        ):
            return fail(
                "Sandbox LLM returned a wrapper/protocol message instead of raw Python code. "
                "Regenerate code that is an executable Python script only."
            )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return fail(f"Syntax error in sandbox code: {e}")

        if not self._has_meaningful_script_body(tree):
            return fail(
                "Sandbox LLM returned text that is syntactically valid Python but not an executable script."
            )

        if "sandbox_result.json" not in code:
            return fail(
                "Sandbox script does not appear to write sandbox_result.json."
            )

        error = self._validate_ast(code)
        if error:
            return fail(error)
        update["sandbox_error"] = None
        return update

    @staticmethod
    def _sandbox_write_guard_prelude(readonly_roots: list[str]) -> str:
        roots_json = json.dumps([str(root) for root in readonly_roots if root])
        return f"""
# --- Framework write guard: session/data is read-only -----------------------
import builtins as __biomed_builtins
import os as __biomed_os
import pathlib as __biomed_pathlib

__BIOMED_READONLY_ROOTS = {roots_json}
__BIOMED_ORIGINAL_OPEN = __biomed_builtins.open

def __biomed_under_readonly_root(path):
    try:
        raw = __biomed_os.fspath(path)
    except TypeError:
        return False
    if not isinstance(raw, str):
        return False
    abs_path = __biomed_os.path.abspath(raw)
    real_path = __biomed_os.path.realpath(raw)
    for root in __BIOMED_READONLY_ROOTS:
        root_abs = __biomed_os.path.abspath(root)
        root_real = __biomed_os.path.realpath(root)
        for candidate in (abs_path, real_path):
            for base in (root_abs, root_real):
                try:
                    if __biomed_os.path.commonpath([base, candidate]) == base:
                        return True
                except ValueError:
                    pass
    return False

def __biomed_is_write_mode(mode):
    return any(flag in str(mode) for flag in ("w", "a", "x", "+"))

def __biomed_guard_open(file, mode="r", *args, **kwargs):
    if __biomed_is_write_mode(mode) and __biomed_under_readonly_root(file):
        raise PermissionError(f"Refusing to write under read-only session data folder: {{file}}")
    return __BIOMED_ORIGINAL_OPEN(file, mode, *args, **kwargs)

__biomed_builtins.open = __biomed_guard_open

__BIOMED_ORIGINAL_PATH_OPEN = __biomed_pathlib.Path.open
def __biomed_guard_path_open(self, mode="r", *args, **kwargs):
    if __biomed_is_write_mode(mode) and __biomed_under_readonly_root(self):
        raise PermissionError(f"Refusing to write under read-only session data folder: {{self}}")
    return __BIOMED_ORIGINAL_PATH_OPEN(self, mode, *args, **kwargs)
__biomed_pathlib.Path.open = __biomed_guard_path_open

for __biomed_name in ("write_text", "write_bytes", "mkdir", "touch", "unlink", "rmdir", "rename", "replace"):
    __biomed_original = getattr(__biomed_pathlib.Path, __biomed_name)
    def __biomed_make_guard(name, original):
        def __biomed_guard(self, *args, **kwargs):
            if __biomed_under_readonly_root(self):
                raise PermissionError(f"Refusing to mutate read-only session data folder via Path.{{name}}: {{self}}")
            if name in ("rename", "replace") and args and __biomed_under_readonly_root(args[0]):
                raise PermissionError(f"Refusing to move output into read-only session data folder: {{args[0]}}")
            return original(self, *args, **kwargs)
        return __biomed_guard
    setattr(__biomed_pathlib.Path, __biomed_name, __biomed_make_guard(__biomed_name, __biomed_original))
# --- End framework write guard ---------------------------------------------
""".lstrip()

    # -------------------------------------------------------------------------
    # Node 4: execute_sandbox_code
    # -------------------------------------------------------------------------

    def execute_sandbox_code(
        self,
        state: SandboxExtendedState,
    ) -> dict[str, Any] | Command[Any]:
        """
        Actually run main.py in a sandbox dir under session.path.

        After execution:
        - Loads sandbox_result.json if present.
        - For kind=='file_listing', converts discovered files into attachments.
        - Also scans the sandbox run directory for any generated files and adds them as attachments.

        Attachments created here:
        - origin: "sandbox"
        - parent:
            - For sandbox-generated files in the run dir: the ID of the first
              matching target file attachment (if any), else None.
            - For file_listing outputs: None (they are root/discovered files).
        """
        session_value = state.get("session")
        session: dict[str, Any] = dict(session_value) if isinstance(session_value, Mapping) else {}
        code = state.get("sandbox_code")
        if not code or not session.get("path"):
            message = "No sandbox_code or session.path for execution."
            entry = {
                "attempt": state.get("sandbox_attempts") or 0,
                "stage": "execution",
                "summary": self._one_line_summary(message),
                "detail": message,
            }
            error_update: dict[str, Any] = {"sandbox_error": message}
            error_update.update(self._append_failure_entry(state, entry))
            return error_update

        session_path = session["path"]
        run_id = uuid.uuid4().hex[:8]  # just a directory name, doesn't have to be ULID
        base_dir = os.path.join(session_path, "sandbox_runs", run_id)
        os.makedirs(base_dir, exist_ok=True)
        mpl_config_dir = os.path.join(base_dir, "matplotlib_config")
        os.makedirs(mpl_config_dir, exist_ok=True)

        script_path = os.path.join(base_dir, "main.py")
        session_data_path = self._session_data_path(session_path)
        guarded_code = self._sandbox_write_guard_prelude(
            [
                os.path.join(session_path, "data"),
                session_data_path,
            ]
        ) + "\n" + code
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(guarded_code)

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = mpl_config_dir

        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            stdout, stderr = proc.communicate(timeout=DEFAULT_SANDBOX_TIMEOUT_SECONDS)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            exit_code = -1

        result_path = os.path.join(base_dir, "sandbox_result.json")
        sandbox_result = None
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    sandbox_result = json.load(f)
            except Exception as e:
                wrapper: dict[str, Any] = {
                    "result": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "run_dir": base_dir,
                }
                message = f"Failed to load sandbox_result.json: {e}"
                load_error_update: dict[str, Any] = {
                    "sandbox_error": message,
                    "sandbox_result": wrapper,
                }
                load_error_update.update(
                    self._append_failure_entry(
                        state,
                        self._failure_entry_from_execution(
                            state,
                            stage="execution",
                            result_wrapper=wrapper,
                            explicit_error=message,
                        ),
                    )
                )
                return Command(
                        update=load_error_update
                    )


        # --- Attachments: start from existing list in state -------------------
        existing_attachments: list[AttachmentInfo] = list(state.get("attachments") or [])
        attachments_by_path: dict[str, AttachmentInfo] = {
            path: attachment
            for attachment in existing_attachments
            if isinstance((path := attachment.get("path")), str)
        }

        # Try to infer a parent attachment from sandbox_spec.target_files
        parent_id: str | None = None
        spec = state.get("sandbox_spec") or {}
        if isinstance(spec, dict):
            target_files = spec.get("target_files") or []
            if isinstance(target_files, list):
                for t in target_files:
                    t_path = t.get("path")
                    if not t_path:
                        continue
                    parent_att = attachments_by_path.get(t_path)
                    if parent_att and parent_att.get("id"):
                        parent_id = parent_att["id"]
                        break

        def _compute_sha256(path: str) -> str:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        if not chunk:
                            break
                        h.update(chunk)
                return h.hexdigest()
            except Exception:
                return ""

        def _is_registerable_generated_path(path: str) -> bool:
            if not path:
                return False
            full = os.path.abspath(path)
            if not os.path.isfile(full):
                return False
            try:
                if os.path.commonpath([os.path.abspath(session_path), full]) != os.path.abspath(session_path):
                    return False
                if os.path.commonpath([os.path.abspath(session_data_path), full]) == os.path.abspath(session_data_path):
                    return False
            except ValueError:
                return False
            return os.path.basename(full) not in ("sandbox_result.json", "main.py")

        def _iter_result_paths(value: Any) -> Iterator[str]:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if isinstance(nested, str) and (
                        key in {"path", "plot_path", "json_path", "output_path", "file_path", "artifact_path"}
                        or key.endswith("_path")
                    ):
                        yield nested
                    else:
                        yield from _iter_result_paths(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from _iter_result_paths(nested)

        generated_attachments: list[AttachmentInfo] = []

        def _append_attachment(full: str) -> None:
            full = os.path.abspath(full)
            if full in attachments_by_path or not _is_registerable_generated_path(full):
                return
            size_mb = os.path.getsize(full) / (1024 * 1024)
            att: AttachmentInfo = {
                "id": _new_attachment_id("file"),
                "filename": os.path.basename(full),
                "path": full,
                "sha256": _compute_sha256(full),
                "size_mb": size_mb,
                "mimetype": None,
                "origin": "sandbox",
                "parent": parent_id,
                "kind": "sandbox_generated",
            }
            new_attachments.append(att)
            generated_attachments.append(att)
            attachments_by_path[full] = att

        new_attachments: list[AttachmentInfo] = []

        # Files generated inside the sandbox run directory (derived images, CSVs, etc.)
        for root, dirs, files in os.walk(base_dir):
            for fn in files:
                if fn in ("sandbox_result.json", "main.py"):
                    continue
                full = os.path.join(root, fn)
                _append_attachment(full)

        if isinstance(sandbox_result, dict):
            for path in _iter_result_paths(sandbox_result):
                _append_attachment(path)

        # Additional attachments from file listings, if any
        if isinstance(sandbox_result, dict) and sandbox_result.get("kind") == "file_listing":
            data_value = sandbox_result.get("data")
            data = data_value if isinstance(data_value, dict) else {}
            files_value = data.get("files")
            files = files_value if isinstance(files_value, list) else []
            for fdesc in files:
                if not isinstance(fdesc, dict):
                    continue
                path = fdesc.get("path")
                if not isinstance(path, str) or not path or not os.path.exists(path):
                    continue
                if path in attachments_by_path:
                    # Already registered as an attachment
                    continue

                filename = fdesc.get("filename") or os.path.basename(path)
                size_mb = fdesc.get("size_mb")
                if size_mb is None:
                    try:
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                    except Exception:
                        size_mb = 0.0
                sha256 = _compute_sha256(path)

                att: AttachmentInfo = {
                    "id": _new_attachment_id("file"),
                    "filename": filename,
                    "path": path,
                    "sha256": sha256,
                    "size_mb": size_mb,
                    "mimetype": None,
                    "origin": "sandbox",
                    "parent": None,             
                    "kind": "file_listing",
                }
                new_attachments.append(att)
                attachments_by_path[path] = att

        all_attachments: list[AttachmentInfo] = existing_attachments + new_attachments

        result_wrapper = {
            "result": sandbox_result,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "run_dir": base_dir,
        }
        execution_update: dict[str, Any] = {
            "sandbox_result": result_wrapper,
            "sandbox_error": None,
        }
        failure_entry = self._failure_entry_from_execution(
            state,
            stage="execution",
            result_wrapper=result_wrapper,
        )
        execution_update.update(self._append_failure_entry(state, failure_entry))
        if all_attachments:
            execution_update["attachments"] = all_attachments
        if generated_attachments:
            execution_update["sandbox_generated"] = generated_attachments

        return Command(update=execution_update)

    # -------------------------------------------------------------------------
    # Node 5: summarize_sandbox_result
    # -------------------------------------------------------------------------

    def summarize_sandbox_result(self, state: SandboxExtendedState) -> dict[str, Any]:
        """
        Turn sandbox_result into a friendly message for the user.

        Special handling for:
        - kind == "file_listing": show a short list of files and mention attachments.

        This node may be reached either:
        - when exit_code == 0 (success), or
        - when we've exhausted MAX_SANDBOX_ATTEMPTS and must surface the final error.
        """
        error = state.get("sandbox_error")
        res = state.get("sandbox_result") or {}

        # Unpack the execution wrapper
        exit_code = res.get("exit_code")
        stdout = res.get("stdout") or ""
        stderr = res.get("stderr") or ""
        run_dir = res.get("run_dir")
        payload = res.get("result")  # this is the JSON written by the sandbox script
        history = state.get("sandbox_failure_history") or []
        latest_failure = history[-1] if isinstance(history, list) and history and isinstance(history[-1], dict) else None

        def withdrawal_update(message: str) -> dict[str, Any]:
            preferred = state.get("sandbox_withdrawal_summary")
            source = preferred if isinstance(preferred, str) and preferred.strip() else message
            compact = self._one_line_summary(source, 500)
            internal = AIMessage(
                content=(
                    "Code sandbox withdrew before completing the routed task.\n\n"
                    f"Issue summary: {compact}\n\n"
                    "The outer router should choose a different next step, ask for missing inputs, "
                    "or avoid repeating the same sandbox attempt unless the issue has changed."
                ),
                name=f"{self.key}_withdrawal_internal",
            )
            return {
                "sandbox_withdrawn": True,
                "sandbox_withdrawal_summary": compact,
                "internal_messages": [internal],
            }

        # If we have no structured payload, fall back to diagnostics
        if not isinstance(payload, dict):
            parts = []

            if error:
                parts.append(f"Sandbox error: {error}")

            parts.append(
                f"Sandbox run finished with exit code "
                f"{exit_code if exit_code is not None else 'unknown'}."
            )

            if stdout:
                parts.append(f"STDOUT:\n{stdout[:2000]}")
            if stderr:
                parts.append(f"STDERR:\n{stderr[:2000]}")

            text = "\n\n".join(parts)
            return withdrawal_update(text)

        # We have a structured payload from sandbox_result.json
        kind = payload.get("kind")
        data = payload.get("data")
        meta = payload.get("meta") or {}
        summary = meta.get("summary") or "Sandbox result:"
        payload_error = self._sandbox_payload_is_error(payload)

        # Start building the message; prepend error if any
        prefix_parts = []
        if error:
            prefix_parts.append(f"Sandbox error (final): {error}")
        if exit_code not in (None, 0):
            prefix_parts.append(f"Final exit code: {exit_code}")

        prefix = "\n\n".join(prefix_parts)
        if prefix:
            summary = prefix + "\n\n" + summary

        # Specialized renderings
        if kind == "file_listing" and isinstance(data, dict):
            files = data.get("files") or []
            total = len(files)
            preview = files[:10]

            lines = []
            for fdesc in preview:
                fname = fdesc.get("filename") or "<unknown>"
                path = fdesc.get("path") or "<no-path>"
                src = fdesc.get("source_root") or "unknown-root"
                lines.append(f"- `{fname}` ({src}): `{path}`")

            listing_md = "\n".join(lines) if lines else "_No files found._"

            if total > len(preview):
                summary += f"\n\nFound {total} file(s). Showing first {len(preview)}."

            msg = (
                f"{summary}\n\n"
                f"{listing_md}\n\n"
                "These files have been registered as attachments when possible, "
                "so other tools and subgraphs can now use them."
            )

        elif kind == "table_preview" and isinstance(data, dict):
            cols = data.get("columns", [])
            rows = data.get("rows", [])

            if cols:
                header = "| " + " | ".join(str(c) for c in cols) + " |"
                sep = "| " + " | ".join("---" for _ in cols) + " |"
                body_lines = []
                for row in rows:
                    body_lines.append("| " + " | ".join(str(v) for v in row) + " |")
                table_md = "\n".join([header, sep, *body_lines])
                msg = f"{summary}\n\n{table_md}"
            else:
                # Fallback if the script didn't provide columns/rows properly
                msg = self._format_compact_payload_message(
                    summary=summary,
                    payload=payload,
                    run_dir=run_dir,
                )
        else:
            msg = self._format_compact_payload_message(
                summary=summary,
                payload=payload,
                run_dir=run_dir,
            )

        if payload_error or error or (isinstance(exit_code, int) and exit_code != 0):
            issue = (
                latest_failure.get("summary")
                if latest_failure and latest_failure.get("summary")
                else msg
            )
            update = withdrawal_update(str(issue))
            update["internal_messages"].append(
                AIMessage(content=msg, name=f"{self.key}_summary_internal")
            )
            return update

        return {
            "sandbox_withdrawn": False,
            "sandbox_withdrawal_summary": "",
            "sandbox_repair_context": "",
            "internal_messages": [AIMessage(content=msg, name=f"{self.key}_summary_internal")],
        }

    
    def _extract_code_block(self, raw: str) -> str:
        """
        Best-effort extraction of plain Python code from a model response.

        - If the response contains Markdown fences (``` or ```python),
          we take the content inside the first fenced block.
        - Otherwise we return the raw string stripped.
        """
        if not raw:
            return ""

        text = raw.strip()

        if "```" not in text:
            # No fences, assume it's already pure code
            return text

        # Find first fence
        first_fence = text.find("```")
        if first_fence == -1:
            return text

        # Skip the opening fence and optional language tag
        after_fence = text[first_fence + 3 :]
        newline_pos = after_fence.find("\n")
        if newline_pos != -1:
            after_fence = after_fence[newline_pos + 1 :]

        # Find closing fence
        closing_fence = after_fence.rfind("```")
        if closing_fence != -1:
            code = after_fence[:closing_fence]
        else:
            code = after_fence

        return code.strip()


    # -------------------------------------------------------------------------
    # Subgraph wiring with refinement logic
    # -------------------------------------------------------------------------

    def attach(
        self,
    ) -> CompiledStateGraph[SandboxExtendedState, None, SandboxExtendedState, SandboxExtendedState]:
        # build internal graph for the sandbox subgraph
        g = StateGraph(SandboxExtendedState)
        g.add_node("build_spec", self.build_sandbox_spec)
        g.add_node("codegen", self.generate_sandbox_code)
        g.add_node("decide_repair", self.decide_sandbox_repair)
        g.add_node("validate", self.validate_sandbox_code)
        g.add_node("execute", self.execute_sandbox_code)
        g.add_node("summarize", self.summarize_sandbox_result)

        g.set_entry_point("build_spec")

        # build_spec is only called once per sandbox run
        g.add_edge("build_spec", "codegen")

        # After validation:
        # - if sandbox_error is set and we still have attempts left -> ask the
        #   repair router whether to regenerate code or regenerate the spec
        # - otherwise -> execute
        def branch_after_validate(state: SandboxExtendedState) -> str:
            error = state.get("sandbox_error")
            attempts = state.get("sandbox_attempts") or 0
            code = (state.get("sandbox_code") or "").strip()

            if error:
                # If we've already tried enough times, or the code is still empty
                # after at least one attempt, give up and surface the error.
                if attempts >= self.max_repairs or (not code and attempts >= 1):
                    return "summarize"
                return "decide_repair"

            # No validation error → safe to try executing the code once
            return "execute"



        g.add_conditional_edges(
            "validate",
            branch_after_validate,
            {
                "codegen": "codegen",
                "decide_repair": "decide_repair",
                "execute": "execute",
                "summarize": "summarize",
            },
        )


        # After execution:
        # - if exit_code == 0 and payload is not an error -> summarize (success path)
        # - if the payload reports an error, or exit_code != 0, and attempts < MAX
        #   -> ask the repair router whether to regenerate code or regenerate the spec
        # - otherwise -> summarize with final error
        def branch_after_execute(state: SandboxExtendedState) -> str:
            res = state.get("sandbox_result") or {}
            attempts = state.get("sandbox_attempts", 0)
            exit_code = res.get("exit_code")
            payload = res.get("result") if isinstance(res, dict) else None
            payload_error = self._sandbox_payload_is_error(payload)

            if payload_error and attempts < self.max_repairs:
                return "decide_repair"
            if payload_error:
                return "summarize"

            if payload is None and attempts < self.max_repairs:
                return "decide_repair"
            if payload is None:
                return "summarize"

            if isinstance(exit_code, int) and exit_code == 0:
                return "summarize"
            if isinstance(exit_code, int) and exit_code != 0 and attempts < self.max_repairs:
                return "decide_repair"
            # No result or max attempts reached → summarize with whatever we have
            return "summarize"

        g.add_conditional_edges(
            "execute",
            branch_after_execute,
            {
                "decide_repair": "decide_repair",
                "summarize": "summarize",
            },
        )

        def branch_after_repair_decision(state: SandboxExtendedState) -> str:
            action = state.get("sandbox_repair_action")
            if action == "regenerate_spec":
                return "build_spec"
            if action == "give_up":
                return "summarize"
            return "codegen"

        g.add_conditional_edges(
            "decide_repair",
            branch_after_repair_decision,
            {
                "build_spec": "build_spec",
                "codegen": "codegen",
                "summarize": "summarize",
            },
        )

        # Linear edges where no branching is needed
        g.add_edge("codegen", "validate")
        g.add_edge("summarize", END)

        return g.compile()
