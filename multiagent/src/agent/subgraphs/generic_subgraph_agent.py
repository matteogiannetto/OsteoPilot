from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
import time
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import ulid
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel

from ..logger_config import get_logger
from ..ollama_retry import is_ollama_usage_limit_error
from ..graph_state import subAgentState, NormalizedToolCall, SubgraphTaskPlan
from ..tools.session_management import AttachmentInfo
from .subgraph_contract import SubgraphModule
from ..agent_execution_recorder import append_tool_event, make_tool_event

logger = get_logger(__name__, level="DEBUG")
MAX_CONSECUTIVE_FAILED_TOOL_BATCHES = 3


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _import_module_from_path(path: str) -> ModuleType:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tool module not found: {p}")
    mod_name = f"_scoped_tool_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import spec from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _new_attachment_id(prefix: str) -> str:
    """Type-prefixed ULID for attachments, e.g. seg-01J... or pre-01J..."""
    return f"{prefix}-{str(ulid.new())}"


_OPENAI_MESSAGE_NAME_RE = re.compile(r"^[^\s<|\\/>]+$")
_OPENAI_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9.]+)s", re.IGNORECASE)


def _sanitize_message_names(messages: list[Any]) -> list[Any]:
    """
    OpenAI chat message names cannot contain spaces or path-like characters.
    LangChain message objects are mutable enough for this narrow cleanup, and
    dropping an invalid name is safer than failing the whole subgraph call.
    """
    cleaned: list[Any] = []
    for message in messages:
        name = getattr(message, "name", None)
        if isinstance(name, str) and name and not _OPENAI_MESSAGE_NAME_RE.match(name):
            try:
                message.name = re.sub(r"[\s<|\\/>]+", "_", name).strip("_") or None
            except Exception:
                pass
        cleaned.append(message)
    return cleaned


def _parse_first_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = stripped.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(stripped[start : index + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

    return None


def _raw_tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        value = tool_call.get("id") or tool_call.get("tool_call_id")
    else:
        value = getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None)
    if isinstance(value, str) and value:
        return value
    return f"missing_tool_call_id_{ulid.new()}"


def _raw_tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function_payload = tool_call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {}
        name = tool_call.get("name") or function_payload.get("name")
    else:
        name = getattr(tool_call, "name", None)
    return name if isinstance(name, str) and name else "<invalid_tool_call>"


def _raw_tool_call_args(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        function_payload = tool_call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {}
        raw_args = tool_call.get("args") or function_payload.get("arguments") or {}
    else:
        raw_args = getattr(tool_call, "args", {})
    if isinstance(raw_args, dict):
        return dict(raw_args)
    return {"raw_arguments": raw_args}


def _tool_message_payload(
    success: bool,
    error: str | None = None,
    output: Any = None,
) -> str:
    payload = {
        "success": success,
        "error": error,
        "output": output,
    }
    return json.dumps(payload, default=str)


def _payload_indicates_success(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False or payload.get("ok") is False:
        return False
    return payload.get("success") is True or payload.get("ok") is True


def _rate_limit_retry_delay(error: Exception) -> float | None:
    """
    Extract a conservative retry delay from OpenAI-style 429 errors.

    The OpenAI SDK already retries, but if it still bubbles a rate-limit error
    to our planner wrapper, immediately starting the next planner attempt tends
    to keep the token window saturated.
    """
    status_code = getattr(error, "status_code", None)
    error_text = str(error)
    if status_code != 429 and "rate_limit" not in error_text.lower():
        return None

    match = _OPENAI_RETRY_AFTER_RE.search(error_text)
    if match:
        try:
            return min(max(float(match.group(1)) + 1.0, 1.0), 65.0)
        except ValueError:
            pass
    return 30.0


# ---------------------------------------------------------------------------
# Generic subgraph agent
# ---------------------------------------------------------------------------

class SubgraphAgent(SubgraphModule):
    """
    Generic plan-execute-summarise subgraph agent.

    All domain-specific content is injected at construction time.
    Instantiate directly or use a factory function (see bottom of file).

    Prompt injection points
    -----------------------
    planner_domain_context
        Prepended to the planner system prompt.
        Should describe what this subgraph is, what inputs it operates on,
        and any high-level scope restrictions the planner must respect.

    executor_domain_context
        Prepended to the executor system prompt.
        Should contain the agent's role identity and any strict scope rules
        (e.g. "do not invent preprocessing steps").

    summarizer_domain_context
        Prepended to the summariser system prompt.
        Should contain the summariser's role identity and any domain-specific
        "do not invent X" clauses (e.g. "do not make modality assumptions").
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *args: Any,
        key: str,
        title: str,
        description: str,
        tool_specs: list[tuple[str, str]],
        planner_domain_context: str,
        executor_domain_context: str,
        summarizer_domain_context: str,
        capabilities: list[str] | None = None,
        strict_tool_loading: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        llm
            LLM instance. Must not be None.
        key
            Unique string identifier for this subgraph (used in logging,
            attachment source tagging, and the LangGraph StateGraph name).
        title
            Human-readable name for this subgraph.
        description
            One-paragraph description of this subgraph's purpose and scope.
        tool_specs
            List of (friendly_name, absolute_path) tuples. Pass "__auto__"
            as the friendly name to register tools under their own .name
            attribute. See _load_scoped_tools for the full contract.
        planner_domain_context
            Domain-specific fragment injected at the top of the planner
            system prompt. See class docstring.
        executor_domain_context
            Domain-specific fragment injected at the top of the executor
            system prompt. See class docstring.
        summarizer_domain_context
            Domain-specific fragment injected at the top of the summariser
            system prompt. See class docstring.
        capabilities
            Optional list of capability strings surfaced by capability_json().

        """
        if llm is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a non-null llm instance."
            )

        # Validate all required string parameters eagerly so failures surface
        # at construction time rather than mid-execution.
        for param_name, param_value in [
            ("key", key),
            ("title", title),
            ("description", description),
            ("planner_domain_context", planner_domain_context),
            ("executor_domain_context", executor_domain_context),
            ("summarizer_domain_context", summarizer_domain_context),
        ]:
            if not isinstance(param_value, str) or not param_value.strip():
                raise ValueError(
                    f"{self.__class__.__name__}: '{param_name}' must be a non-empty string."
                )

        if not isinstance(tool_specs, list) or not tool_specs:
            raise ValueError(
                f"{self.__class__.__name__}: 'tool_specs' must be a non-empty list of (name, path) tuples."
            )

        super().__init__(*args, **kwargs)

        self.key = key
        self.title = title
        self.description_var = description
        self.SCOPED_TOOL_SPECS: list[tuple[str, str]] = tool_specs
        self._planner_domain_context = planner_domain_context
        self._executor_domain_context = executor_domain_context
        self._summarizer_domain_context = summarizer_domain_context
        self._capabilities: list[str] = capabilities or []
        self.strict_tool_loading = strict_tool_loading
        self.tool_loading_errors: list[dict[str, str]] = []

        self.llm = llm
        self._scoped_tools_by_name: dict[str, BaseTool] = {}
        self._load_scoped_tools()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def tools_by_name(self) -> dict[str, BaseTool]:
        return self._scoped_tools_by_name
    
    def description(self) -> str:
        return self.description_var

    def capability_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description_var,
            "capabilities": self._capabilities,
            "tools": sorted(self.tools_by_name.keys()),
            "tool_loading_errors": list(self.tool_loading_errors),
        }

    # ------------------------------------------------------------------
    # Static helpers — pure functions, no instance state required
    # ------------------------------------------------------------------

    @staticmethod
    def _attachment_path(att: Mapping[str, Any]) -> str | None:
        path = att.get("path") or att.get("file_path")
        if isinstance(path, str) and path.strip():
            return path
        return None

    @staticmethod
    def _attachment_name(att: Mapping[str, Any], path: str | None = None) -> str:
        return (
            att.get("filename")
            or att.get("name")
            or (os.path.basename(path) if path else "unnamed")
        )

    @staticmethod
    def _attachment_prefix_from_kind(kind: str | None) -> str:
        """
        Derive a deterministic attachment-id prefix from an attachment kind.
        Keeps only lowercase alphanumeric characters and uses the first 5.
        Falls back to 'file' if kind is missing or empty.
        """
        if not kind:
            return "file"
        cleaned = "".join(ch for ch in str(kind).lower() if ch.isalnum())
        return cleaned[:5] if cleaned else "file"

    @staticmethod
    def _tool_accepts_argument(tool: BaseTool, arg_name: str) -> bool:
        """
        Return True if the tool's Pydantic v2 args_schema declares arg_name.
        This project supports only Pydantic v2 schemas here. Missing args_schema is
        allowed and means the argument cannot be injected. A non-v2 args_schema is a
        tool contract error.
        """
        schema_obj = getattr(tool, "args_schema", None)
        if schema_obj is None:
            return False
        model_fields = getattr(schema_obj, "model_fields", None)
        if not isinstance(model_fields, dict):
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            raise TypeError(
                f"Tool {tool_name!r} exposes an unsupported args_schema. "
                "Expected a Pydantic v2 model with 'model_fields'. "
                f"Got {type(schema_obj).__name__}."
            )
        return arg_name in model_fields

    @staticmethod
    def _apply_orchestrator_tool_scope(
        *,
        tool_name: str,
        tool: BaseTool,
        args_dict: dict[str, Any],
        session_path: str | None,
        workspace_root: Any,
    ) -> dict[str, Any]:
        """
        Inject framework-owned path scope into tool arguments.

        session_path is always owned by the orchestrator because it is the
        active writable session directory. Model-provided session_path values
        are ignored unless session_tree is being asked to inspect a narrower
        path inside the active session tree.

        workspace_root remains configurable/read-only context and is only
        filled when absent.
        """
        scoped_args = dict(args_dict)

        if session_path and SubgraphAgent._tool_accepts_argument(tool, "session_path"):
            is_session_tree = (
                tool_name == "session_tree"
                or getattr(tool, "name", None) == "session_tree"
                or tool.__class__.__name__ == "SessionTreeTool"
            )
            supplied_session_path = scoped_args.get("session_path")
            if is_session_tree and isinstance(supplied_session_path, str) and supplied_session_path.strip():
                session_root = pathlib.Path(session_path).expanduser()
                requested_path = pathlib.Path(supplied_session_path.strip()).expanduser()
                if not requested_path.is_absolute():
                    requested_path = session_root / requested_path

                session_root_abs = os.path.abspath(os.fspath(session_root))
                requested_abs = os.path.abspath(os.fspath(requested_path))
                try:
                    is_inside_session_tree = (
                        os.path.commonpath([session_root_abs, requested_abs]) == session_root_abs
                    )
                except ValueError:
                    is_inside_session_tree = False

                scoped_args["session_path"] = requested_abs if is_inside_session_tree else session_path
            else:
                scoped_args["session_path"] = session_path

        if workspace_root and SubgraphAgent._tool_accepts_argument(tool, "workspace_root"):
            scoped_args.setdefault("workspace_root", workspace_root)

        if session_path:
            SubgraphAgent._reject_session_data_outputs(scoped_args, session_path)

        return scoped_args

    @staticmethod
    def _reject_session_data_outputs(args_dict: dict[str, Any], session_path: str) -> None:
        """
        Block explicit tool outputs under the session-local read-only data folder.

        This keeps session/data usable as an input location while preventing model
        supplied output_path/output_dir values from writing into it.
        """
        session_root = pathlib.Path(session_path).expanduser()
        data_link = session_root / "data"
        data_link_abs = os.path.abspath(os.fspath(data_link))
        data_real_abs = os.path.realpath(os.fspath(data_link))

        def is_inside_data(raw_path: str) -> bool:
            candidate = pathlib.Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = session_root / candidate
            lexical_abs = os.path.abspath(os.fspath(candidate))
            real_abs = os.path.realpath(os.fspath(candidate))
            for root in (data_link_abs, data_real_abs):
                try:
                    if os.path.commonpath([root, lexical_abs]) == root:
                        return True
                    if os.path.commonpath([root, real_abs]) == root:
                        return True
                except ValueError:
                    continue
            return False

        for key in ("output_path", "output_dir"):
            value = args_dict.get(key)
            if isinstance(value, str) and value.strip() and is_inside_data(value.strip()):
                raise ValueError(
                    f"{key} points inside the read-only session data folder: {value!r}. "
                    "Choose a writable session output folder outside `data`."
                )

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------

    def _is_subgraph_generated_attachment(self, att: Mapping[str, Any]) -> bool:
        producer_subgraph = att.get("producer_subgraph")
        if producer_subgraph == self.key:
            return True

        source = att.get("source")
        if isinstance(source, str) and source.startswith(f"{self.key}/"):
            return True

        return False
    
    def _append_subgraph_tool_event(
        self,
        *,
        state: subAgentState,
        execution_trace: dict[str, Any],
        step_index: int,
        tool_name: str,
        tool_args: dict[str, Any],
        success: bool,
        output: Any = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        event = make_tool_event(
            scope="subgraph",
            subgraph_key=self.key,
            step_index=step_index,
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            output=output,
            error=error,
        )
        return append_tool_event(
            {**state, "execution_trace": execution_trace},
            event,
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_message_size_diagnostics(self, label: str, messages: list[Any]) -> None:
        total_chars = 0
        for i, msg in enumerate(messages):
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content)
            total_chars += len(text)
            logger.debug(
                "[%s][%s] msg[%d] type=%s chars=%d",
                label,
                self.key,
                i,
                type(msg).__name__,
                len(text),
            )
        logger.debug(
            "[%s][%s] total_messages=%d total_chars=%d",
            label,
            self.key,
            len(messages),
            total_chars,
        )

    # ------------------------------------------------------------------
    # Planner helpers
    # ------------------------------------------------------------------

    def _planner_build_attachment_brief(
        self,
        attachments: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build a compact, planner-facing view of attachments.
        Keeps parent relationships and size but avoids huge payloads.
        Returns a list of file dicts.
        """
        att_by_id = {
            a.get("id"): a
            for a in attachments
            if isinstance(a, dict) and a.get("id") is not None
        }

        att_brief: list[dict[str, Any]] = []
        for a in attachments:
            if not isinstance(a, dict):
                raise ValueError(f"Invalid attachment (not a dict): {a!r}")

            path = self._attachment_path(a)
            if not path:
                raise ValueError(f"Invalid attachment (no path/file_path): {a!r}")

            name = self._attachment_name(a, path)
            parent_id = a.get("parent")
            parent_path = None
            if parent_id and parent_id in att_by_id:
                parent_path = self._attachment_path(att_by_id[parent_id])

            att_brief.append(
                {
                    "id": a.get("id"),
                    "path": path,
                    "name": name,
                    "size_mb": a.get("size_mb"),
                    "origin": a.get("origin"),
                    "kind": a.get("kind"),
                    "parent_id": parent_id,
                    "parent_path": parent_path,
                }
            )
        return att_brief

    def _planner_build_tool_capabilities(self) -> dict[str, dict[str, Any]]:
        """
        Build a planner-facing description of tools in this subgraph.
        Not stored in scratchpad, used only to build the prompt.
        """
        caps: dict[str, dict[str, Any]] = {}
        for name, tool in self.tools_by_name.items():
            schema_obj = getattr(tool, "args_schema", None)
            arg_names: list[str] = []
            if schema_obj is not None:
                model_fields = getattr(schema_obj, "model_fields", None)
                if not isinstance(model_fields, dict):
                    raise TypeError(
                        f"[PLAN][{self.key}] Tool {name!r} exposes an unsupported "
                        "args_schema. Expected a Pydantic v2 model with 'model_fields'. "
                        f"Got {type(schema_obj).__name__}."
                    )
                arg_names = list(model_fields.keys())

            caps[name] = {
                "description": (getattr(tool, "description", "") or "").strip(),
                "args": arg_names,
            }
        return caps

    def _planner_log_trace(
        self,
        scratchpad: list[Any],
        planner_output: SubgraphTaskPlan,
        raw_reply: Any | None = None,
    ) -> list[Any]:
        
        """Log a compact planner trace into the private scratchpad."""
        
        if raw_reply is not None:
            scratchpad.append(
                AIMessage(
                    content=(
                        "[STRUCTURED OUTPUT]\n"
                        f"{raw_reply}"
                    ) ,  name = f"{self.key}_planner_raw" # letto nelle docs ma non supportato dal codice ???
                )
            )

        note_lines = [
            "[SUBGRAPH PLANNER] Summary and notes:",
            f"- task_status: {planner_output.task_status}",
            f"- task_summary: {planner_output.task_summary.strip()}",
        ]

        if planner_output.steps:
            note_lines.append(
                f"- steps: {len(planner_output.steps)} step(s) planned"
            )

        planner_notes = planner_output.planner_notes.strip()
        if planner_notes:
            note_lines.append(f"- planner_notes: {planner_notes}")

        scratchpad.append(AIMessage(content="\n".join(note_lines),name=f"{self.key}_planner_notes")) # anche qui il campo name
        return scratchpad

    @staticmethod
    def _replace_private_scratchpad(messages: list[Any]) -> list[Any]:
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]

    @staticmethod
    def _new_private_scratchpad_messages(messages: list[Any], start_index: int) -> list[Any]:
        return messages[start_index:]

    # ------------------------------------------------------------------
    # Attachment helpers
    # ------------------------------------------------------------------

    def _add_attachments_for_payload(
        self,
        attachments: list[AttachmentInfo],
        tool_name: str,
        payload: Any,
        tool_args: dict[str, Any] | None = None,
    ) -> tuple[list[AttachmentInfo], list[AttachmentInfo]]:
        """
        Convert a structured tool payload into AttachmentInfo entries.

        Contract:
        - The payload must be a dict.
        - Tool-produced files must be exposed through payload['attachments'].
        - Parent-child relationships are resolved via parent_arg + tool_args.
        - No tool-specific output translation happens in this method.
        """

        def _find_parent_id_from_arg(parent_arg: str | None) -> str | None:
            if not parent_arg or not isinstance(tool_args, dict):
                return None
            parent_path = tool_args.get(parent_arg)
            if not parent_path:
                return None
            parent_att = existing_by_path.get(parent_path)
            if parent_att:
                return parent_att.get("id")
            for a in out:
                if self._attachment_path(a) == parent_path:
                    return a.get("id")
            return None

        def _push_from_spec(spec: dict[str, Any]) -> None:
            path = spec.get("path")
            if not path:
                return
            for a in out:
                if self._attachment_path(a) == path:
                    return

            kind = spec.get("kind") or default_kind or "other"
            description = spec.get("description") or ""
            parent_id = _find_parent_id_from_arg(spec.get("parent_arg"))
            filename = os.path.basename(path)
            prefix = self._attachment_prefix_from_kind(kind)

            att: AttachmentInfo = {
                "id": _new_attachment_id(prefix),
                "filename": filename,
                "path": path,
                "size_mb": os.path.getsize(path) / (1024 * 1024)
                if os.path.exists(path)
                else 0.0,
                "mimetype": None,
                "origin": "agent",
                "parent": parent_id,
                "kind": kind,
                "name": filename,
                "description": description,
                "source": f"{self.key}/{tool_name}",
                "producer_subgraph": self.key,
                "producer_tool": tool_name,
            }
            out.append(att)
            new_attachments.append(att)

        out: list[AttachmentInfo] = list(attachments or [])
        new_attachments: list[AttachmentInfo] = []

        if not isinstance(payload, dict):
            return out, []

        default_kind: str = payload.get("tool_kind") or "other"
        existing_by_path = {
            self._attachment_path(a): a
            for a in out
            if isinstance(a, dict) and self._attachment_path(a)
        }

        att_specs = payload.get("attachments") or []
        if not isinstance(att_specs, list):
            logger.warning(
                "[ATTACHMENTS][%s] Tool %s returned non-list 'attachments'; ignoring payload attachments.",
                self.key,
                tool_name,
            )
            return out , []

        for spec in att_specs:
            if isinstance(spec, dict):
                _push_from_spec(spec)

        return out, new_attachments

    # ------------------------------------------------------------------
    # Tool loading
    # ------------------------------------------------------------------

    def _load_scoped_tools(self) -> None:
        """
        Load tool modules from self.SCOPED_TOOL_SPECS and register only valid
        BaseTool instances.

        Contract: each tool module must define
            EXPORTED_TOOLS: dict[str, BaseTool]
        """
        registry: dict[str, BaseTool] = {}
        self.tool_loading_errors = []
        logger.info("[tools][%s] Loading scoped tools...", self.key)

        def record_tool_loading_error(
            *,
            path: str,
            friendly_name: str,
            stage: str,
            error: str,
        ) -> None:
            item = {
                "subgraph": self.key,
                "friendly_name": friendly_name,
                "path": path,
                "stage": stage,
                "error": error,
            }
            self.tool_loading_errors.append(item)
            logger.warning(
                "[tools][%s] Tool loading problem | friendly_name=%r | path=%s | stage=%s | error=%s",
                self.key,
                friendly_name,
                path,
                stage,
                error,
            )

        def _register_tool(
            registry_key: str,
            tool: BaseTool,
            path: str,
            friendly_name: str,
        ) -> None:
            existing = registry.get(registry_key)
            if existing is not None and existing is not tool:
                record_tool_loading_error(
                    path=path,
                    friendly_name=friendly_name,
                    stage="collision",
                    error=(
                        f"Duplicate tool registry key {registry_key!r}; replacing "
                        f"{existing.__class__.__name__} with {tool.__class__.__name__}."
                    ),
                )
                logger.warning(
                    "[tools][%s] Collision on tool key %r while loading %s. "
                    "Keeping %s and replacing it with %s.",
                    self.key,
                    registry_key,
                    path,
                    existing.__class__.__name__,
                    tool.__class__.__name__,
                )
            registry[registry_key] = tool

        for friendly_name, path in self.SCOPED_TOOL_SPECS:
            path_for_log = str(pathlib.Path(path).expanduser().resolve())
            try:
                module = _import_module_from_path(path)
            except Exception as e:
                record_tool_loading_error(
                    path=path_for_log,
                    friendly_name=friendly_name,
                    stage="import",
                    error=repr(e),
                )
                continue

            if not hasattr(module, "EXPORTED_TOOLS"):
                record_tool_loading_error(
                    path=path_for_log,
                    friendly_name=friendly_name,
                    stage="contract",
                    error="Module does not define EXPORTED_TOOLS.",
                )
                continue

            exported = getattr(module, "EXPORTED_TOOLS")
            if not isinstance(exported, dict):
                record_tool_loading_error(
                    path=path_for_log,
                    friendly_name=friendly_name,
                    stage="contract",
                    error=(
                        "EXPORTED_TOOLS must be dict[str, BaseTool], "
                        f"got {type(exported).__name__}."
                    ),
                )
                continue

            valid_exported: dict[str, BaseTool] = {}
            for exported_name, exported_tool in exported.items():
                if not isinstance(exported_tool, BaseTool):
                    record_tool_loading_error(
                        path=path_for_log,
                        friendly_name=friendly_name,
                        stage="validation",
                        error=(
                            f"Export {exported_name!r} is not a BaseTool; "
                            f"got {type(exported_tool).__name__}."
                        ),
                    )
                    continue
                valid_exported[exported_name] = exported_tool

            if not valid_exported:
                record_tool_loading_error(
                    path=path_for_log,
                    friendly_name=friendly_name,
                    stage="validation",
                    error="No valid BaseTool instances found in EXPORTED_TOOLS.",
                )
                continue

            if friendly_name == "__auto__":
                for exported_name, tool in valid_exported.items():
                    internal_name = getattr(tool, "name", None)
                    if isinstance(internal_name, str) and internal_name:
                        _register_tool(
                            internal_name,
                            tool,
                            path_for_log,
                            friendly_name,
                        )
                        if internal_name != exported_name:
                            _register_tool(
                                exported_name,
                                tool,
                                path_for_log,
                                friendly_name,
                            )
                            logger.debug(
                                "[tools][%s] Registered tool %s under keys %r and %r",
                                self.key,
                                internal_name,
                                internal_name,
                                exported_name,
                            )
                    else:
                        _register_tool(
                            exported_name,
                            tool,
                            path_for_log,
                            friendly_name,
                        )
            else:
                if len(valid_exported) == 1:
                    only_tool = next(iter(valid_exported.values()))
                    _register_tool(
                        friendly_name,
                        only_tool,
                        path_for_log,
                        friendly_name,
                    )
                else:
                    record_tool_loading_error(
                        path=path_for_log,
                        friendly_name=friendly_name,
                        stage="registration",
                        error=(
                            f"Friendly name {friendly_name!r} cannot alias multiple "
                            f"valid tools: {list(valid_exported.keys())}."
                        ),
                    )

            logger.debug(
                "[tools][%s] From %s -> EXPORTED_TOOLS keys: %s",
                self.key,
                path_for_log,
                list(valid_exported.keys()),
            )

        self._scoped_tools_by_name = registry
        logger.info(
            "[tools][%s] Scoped tools ready: %s",
            self.key,
            sorted(registry.keys()),
        )
        if self.tool_loading_errors:
            logger.warning(
                "[tools][%s] Completed with %d tool-loading problem(s). Loaded tools: %s",
                self.key,
                len(self.tool_loading_errors),
                sorted(registry.keys()),
            )
            for err in self.tool_loading_errors:
                logger.warning(
                    "[tools][%s] loading_error path=%s stage=%s error=%s",
                    self.key,
                    err["path"],
                    err["stage"],
                    err["error"],
                )
        if not registry:
            message = (
                f"[tools][{self.key}] No scoped tools were loaded. "
                "This subgraph will not be able to execute tool-backed work."
            )
            logger.error(message)

        if self.strict_tool_loading and self.tool_loading_errors:
            raise RuntimeError(
                f"[tools][{self.key}] Tool loading failed in strict mode:\n"
                + "\n".join(
                    f"- {e['stage']} | {e['path']} | {e['error']}"
                    for e in self.tool_loading_errors
                )
            )

        if self.strict_tool_loading and not registry:
            raise RuntimeError(message)

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def attach(self) -> CompiledStateGraph[subAgentState, None, subAgentState, subAgentState]:
        workflow = StateGraph(
            subAgentState,
        )

        workflow.add_node("plan_and_explain", self.plan_and_explain_node)
        workflow.add_node("agent_step", self.agent_step_node)
        workflow.add_node("summarizer", self.done_node)

        workflow.add_edge(START, "plan_and_explain")
        workflow.add_edge("plan_and_explain", "agent_step")

        workflow.add_conditional_edges(
            "agent_step",
            self._agent_loop_condition,
            {
                "continue": "agent_step",
                "done": "summarizer",
            },
        )
        workflow.add_edge("summarizer", END)

        return workflow.compile(name=f"{self.key}_subgraph")

    # ------------------------------------------------------------------
    # Planner node
    # ------------------------------------------------------------------

    def _run_planner_with_retries(
        self,
        planner_msgs: list[Any],
        max_retries: int = 3,
    ) -> tuple[SubgraphTaskPlan, Any]:
        """
        Run the planner with structured output and strict retries.

        Returns:
            (validated_plan, structured_reply)

        Raises:
            RuntimeError if the planner fails after max_retries attempts.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                structured_llm = self.llm.with_structured_output(SubgraphTaskPlan)
                structured_reply = structured_llm.invoke(_sanitize_message_names(planner_msgs))

                if isinstance(structured_reply, SubgraphTaskPlan):
                    return structured_reply, structured_reply

                plan = SubgraphTaskPlan.model_validate(structured_reply)
                return plan, structured_reply

            except Exception as e:
                if is_ollama_usage_limit_error(e):
                    raise
                last_error = e
                logger.warning(
                    "[PLAN][%s] attempt %d/%d failed: %s",
                    self.key,
                    attempt,
                    max_retries,
                    e,
                )
                retry_delay = _rate_limit_retry_delay(e)
                if retry_delay is not None and attempt < max_retries:
                    logger.warning(
                        "[PLAN][%s] rate limit encountered; sleeping %.1fs before retry",
                        self.key,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"Planner failed after {max_retries} attempts."
        ) from last_error

    def plan_and_explain_node(self, state: subAgentState) -> dict[str, Any]:
        """
        Planner node.

        Produces:
        - private_scratchpad: planner trace messages
        - plan: SubgraphTaskPlan
        - imaging_phase: "planned"
        - imaging_agent_step_count: 0
        - imaging_agent_continue: True
        - imaging_agent_consecutive_failed_tool_batches: 0
        """
        # Start each routed subgraph call with fresh private memory. The executor
        # appends deltas after this so its trace still reaches the summarizer.
        scratchpad: list[Any] = []
        attachments = state.get("attachments") or []
        router_task = state.get("router_subgraph_task")
        workspace_root = state.get("workspace_root")
        session = state.get("session")
        session_path = (
            session.get("path")
            if isinstance(session, dict)
            else getattr(session, "path", None)
        )
        session_data_path = (
            session.get("data_path")
            if isinstance(session, dict)
            else None
        ) or (os.path.join(session_path, "data") if isinstance(session_path, str) else None)

        logger.debug("[PLAN][%s] plan_and_explain_node start", self.key)

        att_brief = self._planner_build_attachment_brief(attachments)
        tool_caps = self._planner_build_tool_capabilities()
        if not tool_caps:
            logger.error(
                "[PLAN][%s] Planner invoked with zero available tools. Routed task=%r",
                self.key,
                router_task,
            )
            tool_caps_text = (
                "NO TOOLS LOADED. This subgraph cannot execute tool-backed operations."
            )
        else:
            tool_caps_text = json.dumps(tool_caps, indent=2)

        # ----------------------------------------------------------------
        # Planner system prompt.
        # self._planner_domain_context is injected at the top; the rest of
        # the prompt is generic and identical across all specialisations.
        # ----------------------------------------------------------------
        
        system = SystemMessage(
            content=(
                f"{self._planner_domain_context}\n\n"
                "Your job is to read:\n"
                "- the high-level routed task\n"
                "- the available attachments\n"
                "- the available tools\n"
                "and produce a structured planning object for the executor.\n\n"
                "File handling:\n"
                "- Treat the session `data` folder as input-only, read-only task data.\n"
                "- It is fine to select files from `data` as primary_inputs, but never plan "
                "a derived file, converted file, plot, mask, report, or other output there.\n"
                "- Put generated files in normal writable session output locations such as "
                "`artifacts`, a tool-specific output folder, or another non-data session subfolder.\n\n"
                "IMPORTANT:\n"
                "- Answer only with a JSON object.\n"
                "- Do not include explanations outside the JSON.\n"
                "- Use real JSON arrays for array fields. Never put arrays inside strings.\n"
                '- CORRECT: "steps": ["step one", "step two"]\n'
                '- INCORRECT: "steps": "[\'step one\', \'step two\']"\n'
                '- CORRECT: "primary_inputs": ["file_a.nii.gz", "file_b.csv"]\n'
                '- INCORRECT: "primary_inputs": "[\'file_a.nii.gz\', \'file_b.csv\']"\n'
                "- The JSON must contain exactly these keys:\n"
                "  {\n"
                '    "task_status": "supported | partially_supported | unsupported_for_this_subgraph",\n'
                '    "task_summary": "Short summary of the routed subtask and relevant files.",\n'
                '    "steps": ["Optional high-level execution steps."],\n'
                '    "primary_inputs": ["Relevant input file paths for this subtask."],\n'
                '    "planner_notes": "Optional short technical notes or strategy. May be empty."\n'
                "  }\n"
                "Set task_status to:\n"
                "- supported: if the routed task can be handled by this subgraph\n"
                "- partially_supported: if the task is semantically in this subgraph's domain, "
                "but only part of it is supported by the available tools\n"
                "- unsupported_for_this_subgraph: if the task falls outside this subgraph's tool-supported scope\n"
                "If task_status is partially_supported, use planner_notes to explicitly name:\n"
                "- the step(s) this subgraph can perform with available tools\n"
                "- the requested step(s) that are not available in this subgraph\n"
                "- whether the unsupported step(s) may be good candidates for code_sandbox if no other subgraph advertises that capability\n"
                "Do not plan tool calls for unsupported steps. Only include executable supported steps in steps.\n"
            ), name=f"{self.key}_planner_system"
        )

        human = HumanMessage(
            content=(
                "The task given to you is:\n"
                f"{router_task or 'N/A'}\n\n"
                "WORKSPACE ROOT (read-only base for generic relative paths, if provided):\n"
                f"{workspace_root or 'N/A'}\n\n"
                "SESSION DATA FOLDER (input-only; not an output destination):\n"
                f"{session_data_path or 'N/A'}\n\n"
                "AVAILABLE ATTACHMENTS (with parent relationships and size in MB):\n"
                f"{json.dumps(att_brief, indent=2)}\n\n"
                "AVAILABLE TOOLS (name -> description / args):\n"
                f"{tool_caps_text}\n\n"
                "YOUR TASK IS TO:\n"
                "Return a JSON object following the schema above. "
                "steps and primary_inputs must be JSON arrays, not strings. "
                "Do not include any text outside the JSON.\n"
            ), name=f"{self.key}_planner_instructions"
        )

        planner_messages = [system, human]
        self._log_message_size_diagnostics("PLANNER_INPUT", planner_messages)
        planner_output, raw_reply = self._run_planner_with_retries(
            planner_messages,
            max_retries=3,
        )

        scratchpad = self._planner_log_trace(
            scratchpad,
            planner_output=planner_output,
            raw_reply=raw_reply,
        )

        logger.debug(
            "[PLAN][%s] plan complete: task_status=%s, primary_inputs=%d, steps=%d",
            self.key,
            planner_output.task_status,
            len(planner_output.primary_inputs),
            len(planner_output.steps),
        )

        return {
            "private_scratchpad": self._replace_private_scratchpad(scratchpad),
            "plan": planner_output.model_dump(),
            "imaging_phase": "planned",
            "imaging_agent_step_count": 0,
            "imaging_agent_continue": True,
            "imaging_agent_consecutive_failed_tool_batches": 0,
            "current_run_produced_attachments": [],
        }

    # ------------------------------------------------------------------
    # Executor node helpers
    # ------------------------------------------------------------------

    def _build_tool_briefs(self) -> dict[str, str]:
        """Build a concise tool-description map for the executor prompt."""
        tool_briefs: dict[str, str] = {}
        for name, tool in self.tools_by_name.items():
            description = getattr(tool, "description", None)
            if not isinstance(description, str):
                logger.warning(
                    "[AGENT][%s] Tool %s does not expose a valid string description.",
                    self.key,
                    name,
                )
                continue
            description = description.strip()
            if description:
                tool_briefs[name] = description
        return tool_briefs

    def _normalize_tool_call(self, tc: Any) -> NormalizedToolCall:
        """
        Normalise the raw tool-call structure emitted by the LLM into a stable shape.
        Supports:
        - dict-style LangChain/OpenAI tool calls
        - object-style tool-call payloads exposing .id, .name, .args
        """
        tool_name = _raw_tool_call_name(tc)
        if tool_name == "<invalid_tool_call>":
            raise ValueError(f"Tool call missing tool name: {tc!r}")
        raw_args = _raw_tool_call_args(tc)
        if "raw_arguments" in raw_args and len(raw_args) == 1:
            raw_value = raw_args["raw_arguments"]
            if isinstance(raw_value, str):
                try:
                    decoded = json.loads(raw_value)
                except Exception as e:
                    raise ValueError(
                        f"Invalid tool-call arguments for {tool_name}: {raw_value!r}"
                    ) from e
                if not isinstance(decoded, dict):
                    raise ValueError(
                        f"Tool-call arguments for {tool_name} must decode to a JSON object, "
                        f"got {type(decoded).__name__}."
                    )
                args_dict = decoded
            else:
                raise ValueError(
                    f"Unsupported argument type for tool {tool_name}: "
                    f"{type(raw_value).__name__}"
                )
        else:
            args_dict = raw_args
        return NormalizedToolCall(
            tool_call_id=_raw_tool_call_id(tc),
            tool_name=tool_name,
            args_dict=args_dict,
        )

    def _get_plan_fields(
        self,
        plan: Any,
    ) -> tuple[str, str, str, list[str], list[str]]:
        """
        Read plan fields stably from either a SubgraphTaskPlan or a dict.
        Returns (task_summary, task_status, planner_notes, steps, primary_inputs).
        """
        if isinstance(plan, SubgraphTaskPlan):
            return (
                plan.task_summary or "",
                plan.task_status or "supported",
                plan.planner_notes or "",
                list(plan.steps or []),
                list(plan.primary_inputs or []),
            )
        if isinstance(plan, dict):
            return (
                plan.get("task_summary") or "",
                plan.get("task_status") or "supported",
                plan.get("planner_notes") or "",
                list(plan.get("steps") or []),
                list(plan.get("primary_inputs") or []),
            )
        return "", "supported", "", [], []

    def _render_attachment_brief_lines(
        self,
        attachment_brief: list[dict[str, Any]],
    ) -> str:
        """Render a compact, readable attachment view for persisted step context."""
        if not attachment_brief:
            return "No attachments are currently registered in session."

        lines: list[str] = []
        for a in attachment_brief:
            if not isinstance(a, dict):
                continue
            path = a.get("path") or "N/A"
            line = (
                f"- {a.get('name') or 'unnamed'}"
                f" | kind={a.get('kind') or 'unknown'}"
                f" | origin={a.get('origin') or 'unknown'}"
                f" | path={path}"
            )
            if a.get("parent_path"):
                line += f" | parent={a['parent_path']}"
            lines.append(line)

        return (
            "\n".join(lines)
            if lines
            else "No attachments are currently registered in session."
        )

    def _build_agent_step_human_message(
        self,
        *,
        step_count: int,
        task_summary: str,
        task_status: str,
        planner_notes: str,
        planned_steps: list[str],
        primary_inputs: list[str],
        attachment_brief: list[dict[str, Any]],
        session_path: str | None = None,
        session_data_path: str | None = None,
        workspace_root: str | None = None,
    ) -> HumanMessage:
        """
        Build the persistent HumanMessage that anchors each executor step.

        This message is NOT stored in the scratchpad. It is rebuilt fresh on
        every step invocation so that the attachment list always reflects the
        current session state (including files produced in previous steps).

        It is prepended to the conversation after the system prompt, giving the
        LLM a stable user-turn that carries the task mandate and the live file
        inventory: [SystemMessage, HumanMessage(this), *scratchpad].
        """
        lines: list[str] = [
            f"[AGENT STEP CONTEXT] step={step_count + 1}",
            f"task_status: {task_status}",
            f"task_summary: {task_summary or 'N/A'}",
        ]

        if planner_notes:
            lines.append(f"planner_notes: {planner_notes}")

        if planned_steps:
            lines.append("planned_steps:")
            lines.extend(f"- {step}" for step in planned_steps)

        if primary_inputs:
            lines.append("primary_inputs:")
            lines.extend(f"- {path}" for path in primary_inputs)

        if workspace_root:
            lines.append(f"workspace_root: {workspace_root}")
        if session_path:
            lines.append(f"session_path: {session_path}")
        if session_data_path:
            lines.append(f"session_data_folder_read_only: {session_data_path}")

        lines.append("attachments_in_session:")
        lines.append(self._render_attachment_brief_lines(attachment_brief))
        lines.append(
            "step_instruction: Review the persisted execution history and decide "
            "whether another tool call is needed now. "
            "If yes, emit the necessary tool call or tool calls. "
            "If not, do not call any tool."
        )

        return HumanMessage(content="\n".join(lines), name=f"{self.key}_executor_context")

    # ------------------------------------------------------------------
    # Executor node
    # ------------------------------------------------------------------

    def agent_step_node(self, state: subAgentState) -> dict[str, Any]:
        """
        Multistep executor node.

        - Uses only private_scratchpad as persisted internal history.
        - Persists one compact HumanMessage per step with operational context.
        - Persists AI decisions, ToolMessages, and explicit status/error events.
        - Continues when at least one tool call succeeds, or while failed tool
          batches remain below the consecutive retry limit.
        - Stops when no tool calls are emitted.
        """
        MAX_STEPS = 10
        
        execution_trace = state.get("execution_trace")
        if not isinstance(execution_trace, dict):
            raise RuntimeError(
                f"[{self.key}] Missing execution_trace in state before subgraph tool execution."
            )
            
            
        current_run_produced_attachments = list(
            state.get("current_run_produced_attachments") or []
        )
        scratchpad = list(state.get("private_scratchpad") or [])
        # This node reads the full current trace for prompting, but returns only
        # newly added messages because private_scratchpad uses add_messages.
        scratchpad_start_index = len(scratchpad)
        attachments = list(state.get("attachments") or [])
        session = state.get("session")
        session_path = (
            session.get("path")
            if isinstance(session, dict)
            else getattr(session, "path", None)
        )
        session_data_path = (
            session.get("data_path")
            if isinstance(session, dict)
            else None
        ) or (os.path.join(session_path, "data") if isinstance(session_path, str) else None)
        workspace_root = state.get("workspace_root")

        plan = state.get("plan")
        (
            task_summary,
            task_status,
            planner_notes,
            planned_steps,
            primary_inputs,
        ) = self._get_plan_fields(plan)

        step_count = int(state.get("imaging_agent_step_count") or 0)
        consecutive_failed_tool_batches = int(
            state.get("imaging_agent_consecutive_failed_tool_batches") or 0
        )

        logger.debug(
            "[AGENT][%s] Step %d starting. attachments=%d",
            self.key,
            step_count,
            len(attachments),
        )
        logger.debug("[AGENT][%s] plan=%s", self.key, plan)
        logger.debug("[AGENT][%s] task_summary=%r", self.key, task_summary)
        logger.debug("[AGENT][%s] task_status=%s", self.key, task_status)

        if step_count >= MAX_STEPS:
            logger.warning("[AGENT][%s] Reached MAX_STEPS, stopping.", self.key)
            scratchpad.append(
                AIMessage(
                    content="[AGENT STATUS] Executor stopped after reaching the internal step limit.",
                    name="framework_message",
                )
            )
            return {
                "private_scratchpad": self._new_private_scratchpad_messages(scratchpad, scratchpad_start_index),
                "attachments": attachments,
                "imaging_agent_step_count": step_count,
                "imaging_agent_continue": False,
                "imaging_agent_consecutive_failed_tool_batches": consecutive_failed_tool_batches,
                "execution_trace": execution_trace,
                "current_run_produced_attachments": current_run_produced_attachments,
            }

        if task_status == "unsupported_for_this_subgraph":
            logger.info(
                "[AGENT][%s] Execution skipped because planner marked task as unsupported.",
                self.key,
            )
            unsupported_task = task_summary.strip() or "No task summary was available."
            scratchpad.append(
                AIMessage(
                    content=(
                        "[AGENT STATUS] Execution not started.\n"
                        "Reason: the routed task is unsupported for this subgraph.\n"
                        f"Task summary: {unsupported_task}\n"
                        f"Planner status: {task_status}\n"
                        "This subgraph only executes tasks that are within the scope of its registered tools."
                    ),
                    name="framework_message",
                )
            )
            return {
                "private_scratchpad": self._new_private_scratchpad_messages(scratchpad, scratchpad_start_index),
                "attachments": attachments,
                "imaging_agent_step_count": step_count + 1,
                "imaging_agent_continue": False,
                "imaging_agent_consecutive_failed_tool_batches": 0,
                "agent_notes": f"Unsupported routed task for this subgraph: {unsupported_task}",
                "execution_trace": execution_trace,
                "current_run_produced_attachments": current_run_produced_attachments,
            }

        seen_tool_ids = set()
        scoped_tools = []
        for tool in self.tools_by_name.values():
            tool_obj_id = id(tool)
            if tool_obj_id in seen_tool_ids:
                continue
            seen_tool_ids.add(tool_obj_id)
            scoped_tools.append(tool)
                
        llm = self.llm.bind_tools(scoped_tools) if scoped_tools else self.llm

        tool_briefs = self._build_tool_briefs()
        attachment_brief = self._planner_build_attachment_brief(attachments)

        # ----------------------------------------------------------------
        # Executor system prompt.
        # Contains only role identity, available tools, and decision rules.
        # Task context and the live file inventory live in the HumanMessage
        # below so that the user turn is always present in the conversation.
        # ----------------------------------------------------------------
        system_message = SystemMessage(
            content=(
                f"{self._executor_domain_context}\n\n"
                "AVAILABLE TOOLS:\n"
                f"{json.dumps(tool_briefs, indent=2)}\n\n"
                "File handling rules:\n"
                "- You may read inputs from the session `data` folder when the task points there.\n"
                "- Do not send any tool output_path or output_dir into `data`; keep generated "
                "artifacts in writable session output folders outside `data`.\n\n"
                "The conversation history below contains planner trace messages, "
                "prior AI decisions, and ToolMessages with tool outputs.\n\n"
                "Repeated-tool-call discipline:\n"
                "- Before emitting a tool call, compare it with the immediately preceding "
                "ToolMessages and AI tool calls in the history.\n"
                "- Do not call the same tool again with the same effective arguments when the "
                "previous call already returned a successful result. An identical repeated call "
                "will return the same information and is not useful progress.\n"
                "- If a prior successful result is insufficient, change the next action: use a "
                "different tool, use meaningfully different arguments, route the unsupported "
                "remainder through the handoff, or stop tool use and let the summarizer report "
                "the limitation.\n"
                "- A similar or repeated call is allowed when something relevant has changed, "
                "such as a new input file, corrected invalid arguments, a different column/"
                "operation, or a narrower follow-up query that can produce new information.\n"
                "- A failed call may still change state by creating files, partial outputs, "
                "attachments, logs, or other artifacts. If the history shows that state changed, "
                "it is valid to make a similar follow-up call that uses the new state or corrects "
                "the failed attempt. Do not treat state-changing failures as identical no-op "
                "repeats.\n\n"
                "If another action is needed and one of the available tools can perform it, "
                "emit one or more tool calls.\n"
                "If no further tool action is needed, do not call any tool.\n"
                "If the task is outside the available tool scope, do not call any tool.\n"
                "For partially_supported plans, execute only the supported steps that are exposed "
                "by available tools. Do not improvise unsupported operations; leave them visible "
                "in the execution history for the router/summarizer handoff."
            ), name= f"{self.key}_executor_system"
        )

        # ----------------------------------------------------------------
        # Persistent HumanMessage.
        # Rebuilt fresh on every step so the attachment list always reflects
        # the current session state, including files produced in prior steps.
        # Not stored in the scratchpad — injected at conversation-assembly
        # time only, giving the LLM a stable user turn that carries both the
        # task mandate and the live file inventory.
        # ----------------------------------------------------------------
        human_message = self._build_agent_step_human_message(
            step_count=step_count,
            task_summary=task_summary,
            task_status=task_status,
            planner_notes=planner_notes,
            planned_steps=planned_steps,
            primary_inputs=primary_inputs,
            attachment_brief=attachment_brief,
            session_path=session_path if isinstance(session_path, str) else None,
            session_data_path=session_data_path if isinstance(session_data_path, str) else None,
            workspace_root=workspace_root if isinstance(workspace_root, str) else None,
        )

        conversation = _sanitize_message_names([system_message, human_message] + scratchpad)

        logger.debug("\n[AGENT][%s] ===== PROMPT START =====", self.key)
        for i, message in enumerate(conversation):
            logger.debug(
                "[%d] %s:\n%s\n",
                i,
                message.__class__.__name__,
                getattr(message, "content", message),
            )
        logger.debug("[AGENT][%s] ===== PROMPT END =====\n", self.key)

        ai_msg = llm.invoke(conversation)
        logger.debug("[AGENT][%s] RAW RESPONSE: %r", self.key, ai_msg)
        scratchpad.append(ai_msg)

        tool_calls = (
            getattr(ai_msg, "tool_calls", None)
            or getattr(ai_msg, "additional_kwargs", {}).get("tool_calls")
        )

        if tool_calls:
            logger.debug("[AGENT][%s] Tool calls detected: %s", self.key, tool_calls)
            successful_tool_runs = 0
            failed_tool_runs = 0

            for tc in tool_calls:
                try:
                    normalized_call = self._normalize_tool_call(tc)
                except Exception as e:
                    err = f"Invalid tool call skipped: {e}"
                    raw_tool_id = _raw_tool_call_id(tc)
                    raw_tool_name = _raw_tool_call_name(tc)
                    raw_tool_args = _raw_tool_call_args(tc)
                    logger.warning("[AGENT][%s] %s", self.key, err)
                    scratchpad.append(
                        ToolMessage(
                            name=raw_tool_name,
                            tool_call_id=raw_tool_id,
                            content=_tool_message_payload(False, err),
                        )
                    )

                    execution_trace = self._append_subgraph_tool_event(
                        state=state,
                        execution_trace=execution_trace,
                        step_index=step_count,
                        tool_name=raw_tool_name,
                        tool_args=raw_tool_args,
                        success=False,
                        output=None,
                        error=err,
                    )

                    failed_tool_runs += 1
                    continue

                args_dict = dict(normalized_call.args_dict)
                requested_tool = self.tools_by_name.get(normalized_call.tool_name)
                if requested_tool is None:
                    err = (
                        f"Requested tool '{normalized_call.tool_name}' is not available "
                        f"in this subgraph."
                    )
                    logger.warning("[AGENT][%s] %s", self.key, err)
                    scratchpad.append(
                        ToolMessage(
                            name=normalized_call.tool_name,
                            tool_call_id=normalized_call.tool_call_id,
                            content=_tool_message_payload(False, err),
                        )
                    )

                    execution_trace = self._append_subgraph_tool_event(
                        state=state,
                        execution_trace=execution_trace,
                        step_index=step_count,
                        tool_name=normalized_call.tool_name,
                        tool_args=args_dict,
                        success=False,
                        output=None,
                        error=err,
                    )

                    failed_tool_runs += 1
                    continue
                
                
                try:
                    args_dict = self._apply_orchestrator_tool_scope(
                        tool_name=normalized_call.tool_name,
                        tool=requested_tool,
                        args_dict=args_dict,
                        session_path=session_path,
                        workspace_root=workspace_root,
                    )
                except Exception as e:
                    err = f"Tool call rejected by filesystem policy: {e}"
                    logger.warning("[AGENT][%s] %s", self.key, err)
                    scratchpad.append(
                        ToolMessage(
                            name=normalized_call.tool_name,
                            tool_call_id=normalized_call.tool_call_id,
                            content=_tool_message_payload(False, err),
                        )
                    )

                    execution_trace = self._append_subgraph_tool_event(
                        state=state,
                        execution_trace=execution_trace,
                        step_index=step_count,
                        tool_name=normalized_call.tool_name,
                        tool_args=args_dict,
                        success=False,
                        output=None,
                        error=err,
                    )

                    failed_tool_runs += 1
                    continue

                logger.info(
                    "[AGENT][%s] Executing tool %s(%s)",
                    self.key,
                    normalized_call.tool_name,
                    args_dict,
                )

                try:
                    result = requested_tool.invoke(args_dict)
                    content = getattr(result, "content", result)

                    if isinstance(content, (dict, list)):
                        payload = content
                    else:
                        try:
                            payload = json.loads(content)
                        except Exception:
                            payload = {"raw": str(content)}

                    payload_success = _payload_indicates_success(payload)
                    payload_error = None
                    if isinstance(payload, dict):
                        payload_error = payload.get("error") or payload.get("message") or payload.get("detail")

                    if payload_success:
                        attachments, new_attachments = self._add_attachments_for_payload(
                            attachments,
                            normalized_call.tool_name,
                            payload,
                            args_dict,
                        )
                        current_run_produced_attachments.extend(new_attachments)
                    

                    scratchpad.append(
                        ToolMessage(
                            name=normalized_call.tool_name,
                            tool_call_id=normalized_call.tool_call_id,
                            content=json.dumps(payload),
                        )
                    )

                    execution_trace = self._append_subgraph_tool_event(
                        state=state,
                        execution_trace=execution_trace,
                        step_index=step_count,
                        tool_name=normalized_call.tool_name,
                        tool_args=args_dict,
                        success=payload_success,
                        output=payload,
                        error=None if payload_success else str(payload_error or "Tool returned success=False."),
                    )

                    if payload_success:
                        successful_tool_runs += 1
                    else:
                        failed_tool_runs += 1
                    logger.debug(
                        "[AGENT][%s] Updated attachments after %s: %s",
                        self.key,
                        normalized_call.tool_name,
                        [self._attachment_path(a) for a in attachments],
                    )

                except Exception as e:
                    err = str(e)
                    logger.exception(
                        "[AGENT][%s] Error while executing tool '%s'",
                        self.key,
                        normalized_call.tool_name,
                    )
                    scratchpad.append(
                        ToolMessage(
                            name=normalized_call.tool_name,
                            tool_call_id=normalized_call.tool_call_id,
                            content=_tool_message_payload(False, err),
                        )
                    )

                    execution_trace = self._append_subgraph_tool_event(
                        state=state,
                        execution_trace=execution_trace,
                        step_index=step_count,
                        tool_name=normalized_call.tool_name,
                        tool_args=args_dict,
                        success=False,
                        output=None,
                        error=err,
                    )

                    failed_tool_runs += 1

            if successful_tool_runs > 0:
                return {
                    "private_scratchpad": self._new_private_scratchpad_messages(scratchpad, scratchpad_start_index),
                    "attachments": attachments,
                    "imaging_agent_step_count": step_count + 1,
                    "imaging_agent_continue": True,
                    "imaging_agent_consecutive_failed_tool_batches": 0,
                    "execution_trace": execution_trace,
                    "current_run_produced_attachments": current_run_produced_attachments,
                }

            consecutive_failed_tool_batches += 1
            should_retry = (
                consecutive_failed_tool_batches < MAX_CONSECUTIVE_FAILED_TOOL_BATCHES
            )
            retry_status = (
                f"Retrying; consecutive failed tool-call batches: "
                f"{consecutive_failed_tool_batches}/{MAX_CONSECUTIVE_FAILED_TOOL_BATCHES}."
                if should_retry
                else (
                    "Executor loop stopped after "
                    f"{MAX_CONSECUTIVE_FAILED_TOOL_BATCHES} consecutive failed "
                    "tool-call batches."
                )
            )
            scratchpad.append(
                AIMessage(
                    content=(
                        "[AGENT STATUS] Tool calls were emitted, but none completed successfully. "
                        f"Successful runs: {successful_tool_runs}. "
                        f"Failed or skipped runs: {failed_tool_runs}. "
                        f"{retry_status}"
                    ),
                    name="framework_message",
                )
            )
            return {
                "private_scratchpad": self._new_private_scratchpad_messages(scratchpad, scratchpad_start_index),
                "attachments": attachments,
                "imaging_agent_step_count": step_count + 1,
                "imaging_agent_continue": should_retry,
                "imaging_agent_consecutive_failed_tool_batches": consecutive_failed_tool_batches,
                "execution_trace": execution_trace,
                "current_run_produced_attachments": current_run_produced_attachments,
            }

        logger.debug("[AGENT][%s] No tool calls emitted. Stopping executor loop.", self.key)
        scratchpad.append(
            AIMessage(
                content="[AGENT STATUS] No further tool calls were emitted. Executor loop stopped.",
                name="framework_message",
            )
        )
        return {
            "private_scratchpad": self._new_private_scratchpad_messages(scratchpad, scratchpad_start_index),
            "attachments": attachments,
            "imaging_agent_step_count": step_count + 1,
            "imaging_agent_continue": False,
            "imaging_agent_consecutive_failed_tool_batches": 0,
            "execution_trace": execution_trace,
            "current_run_produced_attachments": current_run_produced_attachments,
        }

    # ------------------------------------------------------------------
    # Summariser node
    # ------------------------------------------------------------------

    class DoneProducedFile(BaseModel):
        id: str | None = None
        name: str
        path: str | None = None
        description: str = ""
        source: str = ""
        tool_kind: str = "output"
        origin: str | None = None
        parent_id: str | None = None
        parent_path: str | None = None
        size_mb: float | None = None

    def _get_done_run_produced_attachments(
        self,
        state: subAgentState,
    ) -> list[AttachmentInfo]:
        """Return the attachments produced in the current subgraph run."""
        produced_attachments = state.get("current_run_produced_attachments") or []
        if not isinstance(produced_attachments, list):
            logger.warning(
                "[DONE][%s] state['current_run_produced_attachments'] is not a list; ignoring it.",
                self.key,
            )
            return []
        return [a for a in produced_attachments if isinstance(a, dict)]

    def _collect_done_produced_files(
        self,
        produced_attachments: list[AttachmentInfo],
        attachments: list[AttachmentInfo],
    ) -> list[DoneProducedFile]:
        """Collect only files produced in the current subgraph run."""
        att_by_id = {
            a.get("id"): a
            for a in attachments
            if isinstance(a, dict) and a.get("id") is not None
        }

        produced: list[SubgraphAgent.DoneProducedFile] = []

        for a in produced_attachments:
            if not isinstance(a, dict):
                continue

            path = self._attachment_path(a)
            name = self._attachment_name(a, path)

            if not path and not name:
                continue

            parent_id = a.get("parent")
            parent_path = None
            if parent_id and parent_id in att_by_id:
                parent_path = self._attachment_path(att_by_id[parent_id])

            produced.append(
                self.DoneProducedFile(
                    id=a.get("id"),
                    name=name or "unnamed_output",
                    path=path,
                    description=a.get("description") or "",
                    source=a.get("source") or "",
                    tool_kind=a.get("kind") or "output",
                    origin=a.get("origin"),
                    parent_id=parent_id,
                    parent_path=parent_path,
                    size_mb=a.get("size_mb"),
                )
            )

        return produced

    def _render_produced_file_tree(
        self,
        produced_files: list[DoneProducedFile],
    ) -> str:
        """Render a compact parent-child view of produced files."""
        if not produced_files:
            return "No produced files were registered in this run."

        by_parent: dict[str | None, list[SubgraphAgent.DoneProducedFile]] = {}
        by_id: dict[str, SubgraphAgent.DoneProducedFile] = {}

        for produced_file in produced_files:
            if produced_file.id:
                by_id[produced_file.id] = produced_file
            by_parent.setdefault(produced_file.parent_id, []).append(produced_file)

        roots: list[SubgraphAgent.DoneProducedFile] = [
            f for f in produced_files
            if not f.parent_id or f.parent_id not in by_id
        ]

        lines: list[str] = []

        def visit(node: SubgraphAgent.DoneProducedFile, depth: int = 0) -> None:
            indent = "  " * depth
            lines.append(
                f"{indent}- {node.name} | kind={node.tool_kind} | path={node.path or 'N/A'}"
            )
            if node.description:
                lines.append(f"{indent}  description: {node.description}")
            if node.source:
                lines.append(f"{indent}  source: {node.source}")
            for child in by_parent.get(node.id, []):
                visit(child, depth + 1)

        for root in roots:
            visit(root)

        return "\n".join(lines)

    def _render_all_files_inventory(
        self,
        attachments: list[AttachmentInfo],
    ) -> str:
        """Render a compact inventory of all files currently registered in session."""
        if not attachments:
            return "No files are currently registered in the session."

        lines: list[str] = []
        for a in attachments:
            if not isinstance(a, dict):
                continue
            path = self._attachment_path(a) or "N/A"
            name = self._attachment_name(a, None if path == "N/A" else path)
            line = (
                f"- {name}"
                f" | kind={a.get('kind') or 'unknown'}"
                f" | origin={a.get('origin') or 'unknown'}"
                f" | path={path}"
            )
            if a.get("source"):
                line += f" | source={a['source']}"
            lines.append(line)

        return "\n".join(lines)

    def _build_fallback_done_summary(
        self,
        task_summary: str,
        task_status: str,
        planner_notes: str,
        produced_files: list[DoneProducedFile],
        all_files_inventory: str,
    ) -> str:
        """Deterministic fallback summary if the final LLM call fails or returns empty."""
        lines: list[str] = []

        if task_summary:
            lines.append(f"Task summary: {task_summary}")
        if task_status:
            lines.append(f"Planner status: {task_status}")
        if planner_notes:
            lines.append(f"Planner notes: {planner_notes}")

        lines.append("")

        if produced_files:
            lines.append("Files produced in this run:")
            for produced_file in produced_files:
                lines.append(
                    f"- {produced_file.name}\n"
                    f"  Path: {produced_file.path or 'N/A'}\n"
                    f"  Kind: {produced_file.tool_kind}\n"
                    f"  Source: {produced_file.source or 'N/A'}\n"
                    f"  Description: {produced_file.description or 'N/A'}"
                )
                if produced_file.parent_path:
                    lines.append(f"  Parent: {produced_file.parent_path}")
        else:
            lines.append("No produced files were registered in this run.")

        lines.append("")
        lines.append("All registered files:")
        lines.append(all_files_inventory)

        return "\n".join(lines)

    def done_node(self, state: subAgentState) -> dict[str, Any]:
        """
        Final summariser node.

        Design principles:
        - Uses the real private_scratchpad as conversation context.
        - Does not rebuild or duplicate messages.
        - Structures only the produced-files view.
        - Provides a deterministic fallback if the LLM summary fails.
        """
        attachments: list[AttachmentInfo] = state.get("attachments") or []
        scratchpad = list(state.get("private_scratchpad") or [])
        plan = state.get("plan")

        task_summary = ""
        task_status = "unknown"
        planner_notes = ""

        if isinstance(plan, SubgraphTaskPlan):
            task_summary = plan.task_summary or ""
            task_status = plan.task_status or "unknown"
            planner_notes = plan.planner_notes or ""
        elif isinstance(plan, dict):
            task_summary = plan.get("task_summary") or ""
            task_status = plan.get("task_status") or "unknown"
            planner_notes = plan.get("planner_notes") or ""

        produced_attachments = self._get_done_run_produced_attachments(state)
        produced_files = self._collect_done_produced_files(produced_attachments, attachments)
        produced_tree = self._render_produced_file_tree(produced_files)
        all_files_inventory = self._render_all_files_inventory(attachments)

        MAX_SUMMARY_RETRIES = 2
        final_text = ""

        for attempt in range(1, MAX_SUMMARY_RETRIES + 1):
            try:
                # --------------------------------------------------------
                # Summariser system prompt.
                # self._summarizer_domain_context is injected at the top;
                # the rest of the prompt is generic summarisation guidance.
                # --------------------------------------------------------
                summary_system = SystemMessage(
                    content=(
                        f"{self._summarizer_domain_context}\n"
                        "You are given the real persisted execution history of the subgraph as conversation context.\n"
                        "Summarize only what is explicitly visible in that history and in the final file context.\n"
                        "If the user requested complex information such as exact numeric results, arrays, tables, "
                        "structured fields, or multi-part findings, do not reduce it to highlights only; preserve "
                        "the complete relevant information as accurately and usefully as the available context allows.\n"
                        "Do not recommend next steps.\n"
                        "Do not present failed, invalid, or incomplete tool executions as successful results. "
                        "If the execution history contains tool errors, success=false payloads, invalid inputs, "
                        "or missing required outputs with no later successful correction, explicitly state that "
                        "the requested operation was not validly completed and do not report those outputs as meaningful.\n"
                        "If a tool call used contradictory or invalid arguments visible in the history, call that out "
                        "instead of accepting the tool result at face value.\n"
                        "Use clear, factual, technical language.\n"
                        "Do not use JSON in the answer."
                    ) , name= f"{self.key}_summarizer_system"
                )

                summary_user = HumanMessage(
                    content=(
                        "Use the previous conversation context as the execution history.\n\n"
                        f"Task summary: {task_summary or 'N/A'}\n"
                        f"Planner status: {task_status}\n"
                        f"Planner notes: {planner_notes or 'N/A'}\n\n"
                        "Produced files in this session:\n"
                        f"{produced_tree}\n\n"

                        "Write a concise but informative technical summary that:\n"
                        "- explains what happened in the execution history\n"
                        "- mentions the main tool actions and relevant outcomes visible in context\n"
                        "- preserves complete relevant values when the request depends on exact or complex information, rather than returning only highlights\n"
                        "- explicitly reports the files produced in this session\n"
                        "- explicitly reports failed, invalid, or incomplete execution instead of presenting it as a successful result\n"
                        "- if planner status is partially_supported, explicitly states which requested step(s) were not supported by this subgraph and preserves any planner note suggesting code_sandbox for those step(s)\n"
                        "- if planner status is unsupported_for_this_subgraph, explicitly states that no supported tool-backed execution was performed for this subgraph\n"
                        "- stays strictly grounded in the available information"
                    ), name= f"{self.key}_summarizer_user"
                )

                reply = self.llm.invoke(_sanitize_message_names([summary_system, *scratchpad, summary_user]))
                final_text = getattr(reply, "content", str(reply)) or ""

                logger.debug(
                    "[DONE_NODE][%s] LLM summary attempt %d produced text (len=%d).",
                    self.key,
                    attempt,
                    len(final_text),
                )

                if final_text.strip():
                    break

                logger.warning(
                    "[DONE_NODE][%s] LLM summary attempt %d returned empty text, retrying...",
                    self.key,
                    attempt,
                )

            except Exception as e:
                if is_ollama_usage_limit_error(e):
                    raise
                logger.exception(
                    "[DONE_NODE][%s] Error during final summary LLM call on attempt %d: %r",
                    self.key,
                    attempt,
                    e,
                )

        if not final_text.strip():
            logger.warning(
                "[DONE_NODE][%s] Using fallback summary text (LLM unavailable or returned empty).",
                self.key,
            )
            final_text = self._build_fallback_done_summary(
                task_summary=task_summary,
                task_status=task_status,
                planner_notes=planner_notes,
                produced_files=produced_files,
                all_files_inventory=all_files_inventory,
            )

        return {"internal_messages": [AIMessage(content=final_text, name=f"{self.key}_response")]}

    # ------------------------------------------------------------------
    # Loop condition
    # ------------------------------------------------------------------

    def _agent_loop_condition(self, state: subAgentState) -> str:
        cont_raw = state.get("imaging_agent_continue")
        cont = bool(cont_raw)
        logger.debug(
            "[IMAGING_AGENT][%s] _agent_loop_condition called: imaging_agent_continue=%r -> %s",
            self.key,
            cont_raw,
            "continue" if cont else "done",
        )
        return "continue" if cont else "done"
