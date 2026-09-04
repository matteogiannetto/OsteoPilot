#!/usr/bin/env python3
"""Performance evaluator for OsteoBench benchmark questions.

Small user manual
-----------------
Run one or more benchmark questions and receive both pass/fail status and a
13-point rubric score:

    python evaluation/evaluator.py --spec-file osteobench/data_qa.json --human

Useful options:hu
- --task-id <id> can be repeated to run selected questions only.
- --output summary.json writes the complete machine-readable result.
- --render-task-templates resolves placeholders such as {volume_relative_path}
  before execution.
- --preprocess-only prints the rendered spec without running the graph.
- --workspace-root <path> controls where generic relative task paths are resolved.

Score model
-----------
The score is derived from RUBRIC_ROWS below. Add or remove rubric checks there;
the lookup maps and section totals are generated automatically.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field as PydanticField


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIAGENT_ROOT = PROJECT_ROOT / "multiagent"
SRC_ROOT = MULTIAGENT_ROOT / "src"
DEFAULT_WORKSPACE_ROOT = (
    Path(os.environ.get("BIOMED_WORKSPACE_ROOT", str(PROJECT_ROOT)))
    .expanduser()
    .resolve()
)

for import_path in (PROJECT_ROOT, MULTIAGENT_ROOT, SRC_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


if TYPE_CHECKING:
    from agent.ollama_retry import RetryingChatOllama

from agent.time_utils import local_now  # noqa: E402  # Requires the path bootstrap above.
from evaluation.run_summary import build_run_summary_from_state  # noqa: E402


class _SuppressHttpxSuccessfulRequests(logging.Filter):
    _HTTP_200_PATTERN = re.compile(r'HTTP Request: .*"HTTP/\d(?:\.\d)? 200\b')

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "httpx":
            return True
        return self._HTTP_200_PATTERN.search(record.getMessage()) is None


def _install_httpx_success_filter() -> None:
    httpx_logger = logging.getLogger("httpx")
    if getattr(httpx_logger, "_biomed_suppress_success_filter", False):
        return
    httpx_logger.addFilter(_SuppressHttpxSuccessfulRequests())
    httpx_logger._biomed_suppress_success_filter = True  # type: ignore[attr-defined]


_install_httpx_success_filter()


STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"
STATUS_NOT_EVALUATED = "not_evaluated"

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_LIGHT_RED = "\033[91m"
ANSI_DARK_ORANGE = "\033[38;5;208m"
ANSI_YELLOW = "\033[33m"
ANSI_DIM = "\033[2m"


@dataclass(frozen=True)
class RubricCriterion:
    section: str
    code: str
    label: str
    category: str
    max_points: int = 1

    @property
    def id(self) -> str:
        return f"{self.code}_{self.label}"

    @property
    def dimension(self) -> str:
        return self.code[0]


RUBRIC_ROWS = (
    RubricCriterion("routing", "A1", "correct_primary_route", "Correctness"),
    RubricCriterion("routing", "A2", "no_forbidden_routes", "Correctness"),
    RubricCriterion("routing", "A3", "no_redundant_routes", "Efficiency"),
    RubricCriterion("routing", "A4", "required_routes_present", "Completeness"),
    RubricCriterion("subgraph_task", "B1", "task_scope_adherence", "Quality"),
    RubricCriterion("subgraph_task", "B2", "no_redundant_task", "Efficiency"),
    RubricCriterion("tools", "C1", "correct_tools_called", "Correctness"),
    RubricCriterion("tools", "C2", "correct_tool_order", "Correctness"),
    RubricCriterion("tools", "C3", "sandbox_first_attempt", "Efficiency"),
    RubricCriterion("artifacts", "D1", "artifacts_created", "Completeness"),
    RubricCriterion("artifacts", "D2", "artifact_type_correct", "Correctness"),
    RubricCriterion(
        "artifacts",
        "D3",
        "artifact_registered_in_session",
        "Correctness",
    ),
    RubricCriterion("final_output", "E1", "final_output_correct", "Correctness"),
)

RUBRIC_SECTION_ORDER = tuple(
    dict.fromkeys(criterion.section for criterion in RUBRIC_ROWS)
)
RUBRIC_BY_ID = {criterion.id: criterion for criterion in RUBRIC_ROWS}
RUBRIC_BY_SECTION = {
    section: [criterion.id for criterion in RUBRIC_ROWS if criterion.section == section]
    for section in RUBRIC_SECTION_ORDER
}
RUBRIC_MAX_POINTS = sum(criterion.max_points for criterion in RUBRIC_ROWS)

KNOWN_SECTION_KEYS = {
    "dimension_A_routing",
    "dimension_B_task",
    "dimension_C_tools",
    "dimension_D_artifacts",
    "dimension_E_final_output",
}

TASK_DELAY_ENV = "EVAL_TASK_DELAY_SECONDS"
USAGE_LIMIT_PROBE_INTERVAL_ENV = "EVAL_OLLAMA_USAGE_LIMIT_PROBE_INTERVAL_SECONDS"
USAGE_LIMIT_MAX_WAIT_ENV = "EVAL_OLLAMA_USAGE_LIMIT_MAX_WAIT_SECONDS"
DEFAULT_USAGE_LIMIT_PROBE_INTERVAL_SECONDS = 300.0
DEFAULT_USAGE_LIMIT_MAX_WAIT_SECONDS = 6 * 60 * 60.0
DEFAULT_EVALUATOR_RUNS_DIR = MULTIAGENT_ROOT / "evaluator_runs"
OPENAI_JUDGE_RUN_SUMMARY_THRESHOLD_TOKENS = 40_000
OLLAMA_JUDGE_CONTEXT_TOKENS = 256_000
OLLAMA_JUDGE_RUN_SUMMARY_THRESHOLD_TOKENS = 80_000

JUDGE_PASS_THRESHOLD = 3
DIMENSION_B_JUDGE_MAX_ATTEMPTS = 2
OLLAMA_OPTIONS = {
    "num_ctx": OLLAMA_JUDGE_CONTEXT_TOKENS,
    "num_predict": 32768,
    "num_batch": 256,
}
INFRASTRUCTURE_TOOL_NAMES = {"create_session"}


def normalize_score_1_to_5(score: int) -> float:
    return round((score - 1) / 4.0, 4)


class DimensionBJudgeOutput(BaseModel):
    score: int = PydanticField(
        ge=1,
        le=5,
        description="1 = task not achieved at all, 5 = task fully achieved end-to-end",
    )
    rationale: str = PydanticField(
        description=(
            "Detailed grounded explanation for the score, including concrete evidence "
            "from the execution and the main mistakes or missing steps when present."
        )
    )


def get_dimension_b_judge_model_name() -> str:
    model_name = (
        os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("MODEL_NAME")
    )
    if not model_name or not model_name.strip():
        raise RuntimeError(
            "Model not defined. Set EVAL_JUDGE_MODEL or MODEL_NAME before evaluating."
        )
    return model_name.strip()


def dimension_b_judge_uses_openai() -> bool:
    lowered = get_dimension_b_judge_model_name().lower()
    if lowered.startswith("gpt-oss"):
        return False
    return "gpt" in lowered or lowered.startswith(("o1", "o3", "o4"))


def dimension_b_judge_uses_ollama() -> bool:
    return not dimension_b_judge_uses_openai()


def dimension_b_run_summary_threshold_tokens() -> int:
    if dimension_b_judge_uses_ollama():
        return OLLAMA_JUDGE_RUN_SUMMARY_THRESHOLD_TOKENS
    return OPENAI_JUDGE_RUN_SUMMARY_THRESHOLD_TOKENS


def _env_non_negative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring invalid non-negative float value for %s=%r",
            name,
            raw,
        )
        return default
    return max(value, 0.0)


_VALID_GRAPH_NAMES = {
    "full_weak",
    "full_strong",
    "pi_retries_weak",
    "pi_retries_strong",
    "pi_single",
    "llm_only",
    "single_agent_strong",
    "single_agent_weak",
}


def get_compiled_graph_and_llm() -> tuple[Any, Any]:
    graph_name = os.environ.get("BIOMED_GRAPH_NAME", "full_weak").strip()
    if graph_name not in _VALID_GRAPH_NAMES:
        raise ValueError(
            f"Unknown graph name {graph_name!r}. "
            f"Expected one of: {sorted(_VALID_GRAPH_NAMES)}"
        )
    from agent import osteopilot_bootstrap as graph_module

    compiled = getattr(graph_module, f"graph_{graph_name}")
    return compiled, graph_module.llm


def get_compiled_graph() -> Any:
    graph, _llm = get_compiled_graph_and_llm()
    return graph


@lru_cache(maxsize=1)
def get_retrying_chat_ollama_class() -> Any:
    from agent.ollama_retry import RetryingChatOllama

    return RetryingChatOllama


def is_ollama_usage_limit_exception(exc: Exception) -> bool:
    from agent.ollama_retry import is_ollama_usage_limit_error

    return is_ollama_usage_limit_error(exc)


@lru_cache(maxsize=1)
def get_dimension_b_judge_llm() -> Any:
    model_name = get_dimension_b_judge_model_name()

    if dimension_b_judge_uses_openai():
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=512,
            streaming=False,
        )

    RetryingChatOllama = get_retrying_chat_ollama_class()
    base_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("Ollama_API_KEY")
    connection_options: dict[str, Any] = {}
    if base_url:
        connection_options["base_url"] = base_url
    if base_url and api_key:
        connection_options["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {api_key}"}
        }
    return RetryingChatOllama(
        model=model_name,
        temperature=0.0,
        **connection_options,
        **OLLAMA_OPTIONS,
    )


def _ollama_probe_targets() -> list[tuple[str, RetryingChatOllama]]:
    targets: list[tuple[str, RetryingChatOllama]] = []
    seen: set[int] = set()
    RetryingChatOllama = get_retrying_chat_ollama_class()
    _graph, graph_llm = get_compiled_graph_and_llm()

    if isinstance(graph_llm, RetryingChatOllama):
        targets.append(("graph", graph_llm))
        seen.add(id(graph_llm))

    if dimension_b_judge_uses_ollama():
        judge_llm = get_dimension_b_judge_llm()
        if isinstance(judge_llm, RetryingChatOllama) and id(judge_llm) not in seen:
            targets.append(("dimension_b_judge", judge_llm))

    return targets


def _probe_ollama_target(name: str, llm: RetryingChatOllama) -> None:
    response = llm.invoke(
        [
            HumanMessage(
                content="Health probe. Reply with exactly OK.",
                name=f"evaluator_{name}_usage_probe",
            )
        ]
    )
    content = getattr(response, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama {name} probe returned an empty response.")


def wait_for_ollama_usage_limit_recovery(
    *,
    probe_interval_seconds: float | None = None,
    max_wait_seconds: float | None = None,
) -> None:
    interval = (
        DEFAULT_USAGE_LIMIT_PROBE_INTERVAL_SECONDS
        if probe_interval_seconds is None
        else max(float(probe_interval_seconds), 0.0)
    )
    max_wait = (
        DEFAULT_USAGE_LIMIT_MAX_WAIT_SECONDS
        if max_wait_seconds is None
        else max(float(max_wait_seconds), 0.0)
    )
    targets = _ollama_probe_targets()
    if not targets:
        raise RuntimeError(
            "Ollama usage limit was detected, but no RetryingChatOllama "
            "instance was available for recovery probing."
        )

    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            for name, target in targets:
                _probe_ollama_target(name, target)
            return
        except Exception as probe_error:
            elapsed = time.monotonic() - started
            if not is_ollama_usage_limit_exception(probe_error):
                raise RuntimeError(
                    f"Ollama recovery probe failed with a non-quota error: {probe_error}"
                ) from probe_error
            if max_wait > 0 and elapsed >= max_wait:
                raise TimeoutError(
                    "Ollama usage limit did not recover before "
                    f"{max_wait:.1f}s elapsed."
                ) from probe_error
            sleep_for = interval
            if max_wait > 0:
                sleep_for = min(sleep_for, max(max_wait - elapsed, 0.0))
            if sleep_for <= 0:
                sleep_for = 1.0
            print(
                f"[OLLAMA LIMIT] Probe attempt {attempt} still reports usage "
                f"limit; sleeping {sleep_for:.1f}s before probing again.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)


def is_llm_context_length_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code != 400:
        return False

    text_parts = [str(error)]
    response_error = getattr(error, "error", None)
    if isinstance(response_error, str):
        text_parts.append(response_error)
    error_text = " ".join(text_parts).lower()
    return any(
        marker in error_text
        for marker in (
            "prompt is too long",
            "maximum context length",
            "context length",
            "num_ctx",
        )
    )


@dataclass
class ExecutionRecord:
    task_id: str
    final_answer: Any
    messages: list[dict] = field(default_factory=list)
    run_log: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    execution_trace: dict[str, Any] | None = None
    run_summary: dict[str, Any] | None = None
    workspace_dir: str | None = None
    raw_state: dict[str, Any] | None = None


def is_unconstrained(value: Any) -> bool:
    return value in ("", None, [], {})


def result(status: str, details: dict | None = None, reason: str | None = None) -> dict:
    payload: dict[str, Any] = {"status": status, "details": details or {}}
    if reason:
        payload["reason"] = reason
    return payload


def build_section_score(
    criterion_ids: list[str],
    earned_points: dict[str, bool | int],
    *,
    not_applicable: bool = False,
) -> dict[str, Any]:
    criteria: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = {}
    by_dimension: dict[str, dict[str, int]] = {}

    for criterion_id in criterion_ids:
        spec = RUBRIC_BY_ID[criterion_id]
        max_points = spec.max_points
        earned = (
            max_points if not_applicable else int(bool(earned_points.get(criterion_id)))
        )
        item = {
            "id": criterion_id,
            "dimension": spec.dimension,
            "label": spec.label,
            "category": spec.category,
            "earned": earned,
            "max": max_points,
        }
        if not_applicable:
            item["not_applicable"] = True
        criteria.append(item)

        category_totals = by_category.setdefault(spec.category, {"earned": 0, "max": 0})
        category_totals["earned"] += earned
        category_totals["max"] += max_points

        dimension_totals = by_dimension.setdefault(
            spec.dimension, {"earned": 0, "max": 0}
        )
        dimension_totals["earned"] += earned
        dimension_totals["max"] += max_points

    earned_total = sum(item["earned"] for item in criteria)
    max_total = sum(item["max"] for item in criteria)
    return {
        "earned": earned_total,
        "max": max_total,
        "percentage": round(earned_total / max_total, 4) if max_total else 1.0,
        "criteria": criteria,
        "by_category": by_category,
        "by_dimension": by_dimension,
    }


def scored_result(
    status: str,
    criterion_ids: list[str],
    earned_points: dict[str, bool | int],
    details: dict | None = None,
    reason: str | None = None,
    *,
    not_applicable: bool = False,
) -> dict:
    payload = result(status, details=details, reason=reason)
    section_score = build_section_score(
        criterion_ids,
        earned_points,
        not_applicable=not_applicable,
    )
    payload["details"]["points"] = {
        criterion["id"]: criterion["earned"] for criterion in section_score["criteria"]
    }
    payload["details"]["score"] = section_score
    return payload


def aggregate_score(section_results: dict[str, dict]) -> dict[str, Any]:
    criteria_by_id: dict[str, dict[str, Any]] = {}
    for section_key, criterion_ids in RUBRIC_BY_SECTION.items():
        section_result = section_results.get(section_key) or {}
        score = (section_result.get("details") or {}).get("score")
        criteria = score.get("criteria") if isinstance(score, dict) else None
        if isinstance(criteria, list):
            for criterion in criteria:
                if isinstance(criterion, dict) and criterion.get("id"):
                    criteria_by_id[str(criterion["id"])] = criterion
            continue

        for criterion_id in criterion_ids:
            spec = RUBRIC_BY_ID[criterion_id]
            criteria_by_id[criterion_id] = {
                "id": criterion_id,
                "dimension": spec.dimension,
                "label": spec.label,
                "category": spec.category,
                "earned": 0,
                "max": spec.max_points,
            }

    ordered_criteria = [
        criteria_by_id[criterion_id]
        for criterion_id in RUBRIC_BY_ID
        if criterion_id in criteria_by_id
    ]
    earned = sum(int(criterion.get("earned", 0)) for criterion in ordered_criteria)
    max_points = sum(int(criterion.get("max", 0)) for criterion in ordered_criteria)
    return {
        "earned": earned,
        "max": max_points or RUBRIC_MAX_POINTS,
        "percentage": round(earned / max_points, 4) if max_points else 0.0,
        "label": f"{earned}/{max_points or RUBRIC_MAX_POINTS}",
        "criteria": ordered_criteria,
    }


def load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_spec(spec_file: str) -> list[dict]:
    tasks = load_json_file(spec_file)
    if not isinstance(tasks, list):
        raise ValueError("task spec root must be a JSON list")
    return expand_list_valued_task_fields(expand_patient_id_variants(tasks))


def _variant_id_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "patient"


def _path_variant_id_slug(value: str) -> str:
    path = Path(value.strip().rstrip("/"))
    name = path.name or value.strip()
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem or name
    if name.endswith("_original"):
        name = name[: -len("_original")]
    return _variant_id_slug(name)


def _dataset_path_variant_id_slug(value: str) -> str:
    path = Path(value.strip().rstrip("/"))
    if path.parent.name and path.name in {"group_1", "group_2"}:
        return _variant_id_slug(f"{path.parent.name}_{path.name}")
    return _variant_id_slug(path.name or value.strip())


def _task_without_key(task: dict[str, Any], key_to_remove: str) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != key_to_remove}


def _infer_template_patient_id(
    task: dict[str, Any],
    patient_ids: list[str],
) -> str:
    for field_name in ("sample_id", "sample_identifier"):
        value = task.get(field_name)
        if isinstance(value, str):
            for patient_id in patient_ids:
                if patient_id in value:
                    return patient_id

    serialized = json.dumps(
        _task_without_key(task, "patient_id"),
        ensure_ascii=False,
        sort_keys=True,
    )
    scored = [
        (serialized.count(patient_id), patient_id)
        for patient_id in patient_ids
    ]
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return patient_ids[0]


def _replace_patient_id_in_value(
    value: Any,
    template_patient_id: str,
    variant_patient_id: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(template_patient_id, variant_patient_id)
    if isinstance(value, list):
        return [
            _replace_patient_id_in_value(
                item,
                template_patient_id,
                variant_patient_id,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_patient_id_in_value(
                item,
                template_patient_id,
                variant_patient_id,
            )
            for key, item in value.items()
        }
    return value


def expand_patient_id_variants(tasks: list[dict]) -> list[dict]:
    """Expand list-valued patient_id fields into one concrete task per patient."""
    expanded_tasks: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict):
            expanded_tasks.append(task)
            continue

        patient_id_value = task.get("patient_id")
        if not isinstance(patient_id_value, list):
            expanded_tasks.append(task)
            continue

        patient_ids: list[str] = []
        seen_patient_ids: set[str] = set()
        for item in patient_id_value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"task '{task.get('id', '<unknown>')}' patient_id list must "
                    "contain only non-empty strings"
                )
            patient_id = item.strip()
            if patient_id not in seen_patient_ids:
                seen_patient_ids.add(patient_id)
                patient_ids.append(patient_id)

        if not patient_ids:
            raise ValueError(
                f"task '{task.get('id', '<unknown>')}' patient_id list cannot be empty"
            )

        template_patient_id = _infer_template_patient_id(task, patient_ids)
        base_task_id = str(task.get("id", "task"))
        uses_patient_id_placeholder = "{patient_id}" in json.dumps(
            _task_without_key(task, "patient_id"),
            ensure_ascii=False,
        )
        for patient_id in patient_ids:
            variant = copy.deepcopy(task)
            if not uses_patient_id_placeholder:
                variant = _replace_patient_id_in_value(
                    variant,
                    template_patient_id,
                    patient_id,
                )
            variant["patient_id"] = patient_id
            variant["source_task_id"] = base_task_id
            variant["patient_id_template"] = template_patient_id
            if len(patient_ids) > 1:
                variant["id"] = f"{base_task_id}_{_variant_id_slug(patient_id)}"
                if isinstance(variant.get("name"), str) and variant["name"]:
                    variant["name"] = f"{variant['name']} [{patient_id}]"
            expanded_tasks.append(variant)

    return expanded_tasks


def expand_list_valued_task_fields(tasks: list[dict]) -> list[dict]:
    """Expand list-valued input fields into one concrete task per input."""
    list_fields = {
        "volume_relative_path": _path_variant_id_slug,
        "mask_relative_path": _path_variant_id_slug,
        "dataset_relative_path": _dataset_path_variant_id_slug,
    }
    expanded_tasks: list[dict] = []

    for task in tasks:
        variants = [task]
        for field_name, slugger in list_fields.items():
            next_variants: list[dict] = []
            for variant_source in variants:
                if not isinstance(variant_source, dict):
                    next_variants.append(variant_source)
                    continue

                field_value = variant_source.get(field_name)
                if not isinstance(field_value, list):
                    next_variants.append(variant_source)
                    continue

                concrete_values: list[str] = []
                seen_values: set[str] = set()
                for item in field_value:
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(
                            f"task '{variant_source.get('id', '<unknown>')}' "
                            f"{field_name} list must contain only non-empty strings"
                        )
                    concrete_value = item.strip()
                    if concrete_value not in seen_values:
                        seen_values.add(concrete_value)
                        concrete_values.append(concrete_value)

                if not concrete_values:
                    raise ValueError(
                        f"task '{variant_source.get('id', '<unknown>')}' "
                        f"{field_name} list cannot be empty"
                    )

                base_task_id = str(variant_source.get("id", "task"))
                original_source_task_id = str(
                    variant_source.get("source_task_id", base_task_id)
                )
                sample_id_values = variant_source.get("sample_id")
                use_parallel_sample_ids = (
                    isinstance(sample_id_values, list)
                    and len(sample_id_values) == len(field_value)
                    and all(
                        isinstance(item, str) and item.strip()
                        for item in sample_id_values
                    )
                )
                for concrete_value in concrete_values:
                    variant = copy.deepcopy(variant_source)
                    variant[field_name] = concrete_value
                    if use_parallel_sample_ids:
                        original_index = field_value.index(concrete_value)
                        variant["sample_id_template"] = sample_id_values
                        variant["sample_id"] = sample_id_values[original_index].strip()
                    elif field_name == "volume_relative_path" and isinstance(
                        variant.get("sample_id"), str
                    ):
                        variant["sample_id_template"] = variant["sample_id"]
                        variant["sample_id"] = slugger(concrete_value)
                    variant["source_task_id"] = original_source_task_id
                    variant[f"{field_name}_variant"] = concrete_value
                    if len(concrete_values) > 1:
                        variant["id"] = (
                            f"{base_task_id}_{slugger(concrete_value)}"
                        )
                    next_variants.append(variant)
            variants = next_variants
        expanded_tasks.extend(variants)

    return expanded_tasks


PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_task_template_string(text: str, task_context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in task_context:
            return match.group(0)
        value = task_context[key]
        return str(value)

    return PLACEHOLDER_PATTERN.sub(replace, text)


def render_task_template_value(value: Any, task_context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return render_task_template_string(value, task_context)
    if isinstance(value, list):
        return [render_task_template_value(item, task_context) for item in value]
    if isinstance(value, dict):
        return {
            key: render_task_template_value(item, task_context)
            for key, item in value.items()
        }
    return value


def preprocess_task_spec(tasks: list[dict]) -> list[dict]:
    rendered_tasks: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict):
            rendered_tasks.append(task)
            continue
        rendered_task = copy.deepcopy(task)
        for _ in range(5):
            next_rendered_task = render_task_template_value(
                rendered_task,
                rendered_task,
            )
            if next_rendered_task == rendered_task:
                break
            rendered_task = next_rendered_task
        rendered_tasks.append(rendered_task)
    return rendered_tasks


def validate_task_spec(tasks: list[dict]) -> None:
    seen_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task at index {index} must be a JSON object")

        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                f"task at index {index} must contain a non-empty string id"
            )

        if task_id in seen_ids:
            raise ValueError(f"duplicate task id '{task_id}'")
        seen_ids.add(task_id)

        evaluation = task.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(f"task '{task_id}' must contain an evaluation object")

        missing_keys = KNOWN_SECTION_KEYS - set(evaluation.keys())
        if missing_keys:
            raise ValueError(
                f"task '{task_id}' is missing evaluation keys: {sorted(missing_keys)}"
            )

        unknown_keys = set(evaluation.keys()) - KNOWN_SECTION_KEYS
        if unknown_keys:
            raise ValueError(
                f"task '{task_id}' contains unknown evaluation keys: {sorted(unknown_keys)}"
            )

        for section_key in KNOWN_SECTION_KEYS:
            if not isinstance(evaluation[section_key], dict):
                raise ValueError(
                    f"task '{task_id}' evaluation key '{section_key}' must be an object"
                )


def build_task_request(task: dict) -> str:
    parts = [
        task.get("question", ""),
        task.get("additional_instructions", ""),
        task.get("output_instructions", ""),
    ]
    return "\n\n".join(part for part in parts if part)


def message_to_dict(message: Any) -> dict[str, Any]:
    return {
        "type": type(message).__name__,
        "content": getattr(message, "content", ""),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _extract_tool_name_from_id(tool_call_id: str | None) -> str | None:
    if not isinstance(tool_call_id, str) or ":" not in tool_call_id:
        return None
    parts = tool_call_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return None


def _message_lists_from_state(state: dict[str, Any]) -> list[list[Any]]:
    lists: list[list[Any]] = []
    for key in (
        "messages",
        "internal_messages",
        "private_scratchpad",
        "sandbox_messages",
    ):
        value = state.get(key)
        if isinstance(value, list):
            lists.append(value)
    return lists


def extract_tool_calls_from_state(state: dict[str, Any]) -> list[dict]:
    tool_calls: list[dict] = []
    seen: set[tuple[str | None, str | None, str]] = set()

    for message_list in _message_lists_from_state(state):
        for msg in message_list:
            raw_tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    key = (tc.get("id"), name, "ai")
                    if key in seen:
                        continue
                    seen.add(key)
                    tool_calls.append(tc)

            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_name = getattr(msg, "name", None) or _extract_tool_name_from_id(
                tool_call_id
            )
            if tool_name:
                payload = {
                    "name": tool_name,
                    "tool_call_id": tool_call_id,
                }
                key = (tool_call_id, tool_name, "tool")
                if key not in seen:
                    seen.add(key)
                    tool_calls.append(payload)

    execution_trace = state.get("execution_trace")
    if isinstance(execution_trace, dict):
        tool_events = execution_trace.get("tool_events")
        if isinstance(tool_events, list):
            for event in tool_events:
                if not isinstance(event, dict):
                    continue
                tool_name = event.get("tool_name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                key = (event.get("event_id"), tool_name, "execution_trace")
                if key in seen:
                    continue
                seen.add(key)
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "success": event.get("success"),
                        "scope": event.get("scope"),
                        "subgraph_key": event.get("subgraph_key"),
                        "event_id": event.get("event_id"),
                    }
                )

    return tool_calls


def extract_artifacts_from_state(state: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    seen: set[str] = set()

    attachments = state.get("attachments") or []
    if isinstance(attachments, list):
        for att in attachments:
            if not isinstance(att, dict):
                continue
            path = att.get("path")
            origin = att.get("origin")
            kind = att.get("kind")
            producer_subgraph = att.get("producer_subgraph")
            producer_tool = att.get("producer_tool")

            if not isinstance(path, str) or not path:
                continue

            produced = (
                origin in {"agent", "tool", "sandbox"}
                or kind == "sandbox_generated"
                or producer_subgraph is not None
                or producer_tool is not None
            )
            if produced and path not in seen:
                seen.add(path)
                artifacts.append(path)

    sandbox_generated = state.get("sandbox_generated") or []
    if isinstance(sandbox_generated, list):
        for att in sandbox_generated:
            if not isinstance(att, dict):
                continue
            path = att.get("path")
            if isinstance(path, str) and path and path not in seen:
                seen.add(path)
                artifacts.append(path)

    return artifacts


def final_answer_from_state(state: dict[str, Any]) -> Any:
    messages = state.get("messages") or []
    if isinstance(messages, list) and messages:
        return getattr(messages[-1], "content", messages[-1])
    return ""


def execution_record_from_state(
    task_id: str,
    state: dict[str, Any],
    task_request: str | None = None,
) -> ExecutionRecord:
    session = state.get("session") or {}
    workspace_dir = session.get("path") if isinstance(session, dict) else None

    messages = [message_to_dict(msg) for msg in (state.get("messages") or [])]
    run_log = list(state.get("route_history") or [])
    tool_calls = extract_tool_calls_from_state(state)
    artifacts = extract_artifacts_from_state(state)
    run_summary = build_run_summary_from_state(
        final_state=state,
        task=task_request,
        raw_endpoint=None,
        reduction_threshold_tokens=dimension_b_run_summary_threshold_tokens(),
    )

    return ExecutionRecord(
        task_id=task_id,
        final_answer=final_answer_from_state(state),
        messages=messages,
        run_log=run_log,
        tool_calls=tool_calls,
        artifacts=artifacts,
        execution_trace=state.get("execution_trace")
        if isinstance(state.get("execution_trace"), dict)
        else None,
        run_summary=run_summary,
        workspace_dir=workspace_dir,
        raw_state=state,
    )


def build_initial_state(
    task: dict,
    workspace_root: str | Path | None = None,
    execution_trace_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root or DEFAULT_WORKSPACE_ROOT).expanduser().resolve()
    task_id = task["id"]
    task_name = task.get("name")
    state = {
        "messages": [HumanMessage(content=build_task_request(task))],
        "internal_messages": [],
        "route_history": [],
        "test_name": task_id,
        "evaluator": {
            "source": "performance_evaluator",
            "task_id": task_id,
            "task_name": task_name if isinstance(task_name, str) else None,
        },
        "evaluator_task_id": task_id,
        "evaluator_task_name": task_name if isinstance(task_name, str) else "",
        "workspace_root": str(root),
    }
    if execution_trace_base_dir is not None:
        state["execution_trace_base_dir"] = str(Path(execution_trace_base_dir).expanduser().resolve())
    return state


def invoke_graph(compiled_graph: Any, initial_state: dict[str, Any]) -> dict[str, Any]:
    if hasattr(compiled_graph, "invoke"):
        result_state = compiled_graph.invoke(initial_state)
    elif hasattr(compiled_graph, "ainvoke"):
        result_state = asyncio.run(compiled_graph.ainvoke(initial_state))
    else:
        raise TypeError("compiled_graph must expose invoke() or ainvoke()")

    if not isinstance(result_state, dict):
        raise ValueError("graph did not return a state dictionary")

    return result_state


def run_graph_for_task(
    task: dict,
    compiled_graph: Any,
    workspace_root: str | Path | None = None,
    execution_trace_base_dir: str | Path | None = None,
) -> ExecutionRecord:
    state = invoke_graph(
        compiled_graph,
        build_initial_state(
            task,
            workspace_root=workspace_root,
            execution_trace_base_dir=execution_trace_base_dir,
        ),
    )
    return execution_record_from_state(
        task["id"],
        state,
        task_request=build_task_request(task),
    )


def extract_executed_routes(record: ExecutionRecord) -> list[str]:
    routes: list[str] = []
    for entry in record.run_log:
        route = entry.get("route")
        if isinstance(route, str) and route:
            routes.append(route)
    return routes


def extract_routes_from_record(record: ExecutionRecord) -> list[str]:
    routes: list[str] = []
    seen: set[str] = set()
    for route in extract_executed_routes(record):
        if route not in seen:
            seen.add(route)
            routes.append(route)
    return routes


def evaluate_routing(cfg: dict, record: ExecutionRecord) -> dict:
    criterion_ids = RUBRIC_BY_SECTION["routing"]
    expected_primary = cfg.get("expected_primary_route", "")
    necessary_routes = cfg.get("necessary_routes", cfg.get("acceptable_routes", []))
    forbidden_routes = cfg.get("forbidden_routes", [])

    if all(
        is_unconstrained(value)
        for value in (
            expected_primary,
            necessary_routes,
            forbidden_routes,
        )
    ):
        return scored_result(
            STATUS_SKIPPED,
            criterion_ids,
            {},
            reason="no routing constraints",
            not_applicable=True,
        )

    executed_routes = extract_executed_routes(record)
    unique_routes = extract_routes_from_record(record)
    details = {
        "executed_routes": executed_routes,
        "unique_routes": unique_routes,
    }
    failures: list[str] = []
    primary_passed = True
    required_routes_passed = True
    forbidden_routes_passed = True

    if not is_unconstrained(expected_primary):
        if not executed_routes:
            failures.append("no routes detected")
            primary_passed = False
        elif executed_routes[0] != expected_primary:
            failures.append(
                f"expected primary route '{expected_primary}', observed '{executed_routes[0]}'"
            )
            primary_passed = False

    if not is_unconstrained(necessary_routes):
        missing = [route for route in necessary_routes if route not in executed_routes]
        if missing:
            failures.append(f"missing required routes: {missing}")
            required_routes_passed = False

    if not is_unconstrained(forbidden_routes):
        found = [route for route in executed_routes if route in forbidden_routes]
        if found:
            failures.append(f"forbidden routes visited: {found}")
            forbidden_routes_passed = False

    duplicate_routes = sorted(
        {route for route in executed_routes if executed_routes.count(route) > 1}
    )
    details["duplicate_routes"] = duplicate_routes
    redundant_routes_passed = not duplicate_routes
    if duplicate_routes:
        details["soft_failures"] = [
            f"redundant routes selected more than once: {duplicate_routes}"
        ]

    points = {
        "A1_correct_primary_route": primary_passed,
        "A2_no_forbidden_routes": forbidden_routes_passed,
        "A3_no_redundant_routes": redundant_routes_passed,
        "A4_required_routes_present": required_routes_passed,
    }
    if failures:
        details["failures"] = failures
        return scored_result(STATUS_FAILED, criterion_ids, points, details=details)
    return scored_result(STATUS_PASSED, criterion_ids, points, details=details)


def task_spec_for_judge(task_spec: dict | None) -> dict[str, Any]:
    if not isinstance(task_spec, dict):
        return {}
    return {
        "question": task_spec.get("question", ""),
        "additional_instructions": task_spec.get("additional_instructions", ""),
        "output_instructions": task_spec.get("output_instructions", ""),
    }


def legacy_dimension_b_evidence(record: ExecutionRecord) -> dict[str, Any]:
    evidence = {
        "final_answer": record.final_answer,
        "executed_routes": extract_executed_routes(record),
        "subgraph_task_requests": extract_subgraph_task_requests(record),
        "tool_calls": record.tool_calls,
        "artifacts": record.artifacts,
    }
    state = record.raw_state if isinstance(record.raw_state, dict) else {}
    sandbox_result = state.get("sandbox_result")
    if isinstance(sandbox_result, dict):
        evidence["sandbox_execution"] = {
            "evidence_kind": "sandbox_execution",
            "note": (
                "code_sandbox executes generated Python directly; empty "
                "tool_event_ids/tool_names mean no nested graph tools were called, "
                "not that code execution was absent."
            ),
            "attempts": state.get("sandbox_attempts"),
            "exit_code": sandbox_result.get("exit_code"),
            "run_dir": sandbox_result.get("run_dir"),
            "result": sandbox_result.get("result"),
            "stdout": sandbox_result.get("stdout"),
            "stderr": sandbox_result.get("stderr"),
            "sandbox_error": state.get("sandbox_error"),
            "withdrawn": state.get("sandbox_withdrawn"),
            "withdrawal_summary": state.get("sandbox_withdrawal_summary"),
            "code_present": bool(str(state.get("sandbox_code") or "").strip()),
            "spec": state.get("sandbox_spec"),
        }
    return evidence


def build_legacy_dimension_b_judge_payload(
    *,
    execution_record: ExecutionRecord,
    expected_task: str,
    rationale_for_llm_judge: str,
    task_spec: dict | None,
    run_summary_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "payload_format": "legacy",
        "task_id": task_spec["id"] if isinstance(task_spec, dict) else None,
        "task_spec": task_spec_for_judge(task_spec),
        "expected_task": expected_task,
        "rationale_for_llm_judge": rationale_for_llm_judge,
        "run_summary": run_summary_override
        or execution_record.run_summary
        or legacy_dimension_b_evidence(execution_record),
    }


def build_run_summary_dimension_b_judge_payload(
    *,
    execution_record: ExecutionRecord,
    expected_task: str,
    rationale_for_llm_judge: str,
    task_spec: dict | None,
    run_summary_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_summary = run_summary_override or execution_record.run_summary
    if run_summary is None:
        run_summary = legacy_dimension_b_evidence(execution_record)

    return {
        "payload_format": "run_summary",
        "task_id": task_spec["id"] if isinstance(task_spec, dict) else None,
        "task_spec": task_spec_for_judge(task_spec),
        "expected_task": expected_task,
        "rationale_for_llm_judge": rationale_for_llm_judge,
        "run_summary": run_summary,
    }


def build_dimension_b_judge_payload(
    *,
    execution_record: ExecutionRecord,
    expected_task: str,
    rationale_for_llm_judge: str,
    task_spec: dict | None,
    run_summary_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dimension_b_judge_uses_ollama():
        return build_run_summary_dimension_b_judge_payload(
            execution_record=execution_record,
            expected_task=expected_task,
            rationale_for_llm_judge=rationale_for_llm_judge,
            task_spec=task_spec,
            run_summary_override=run_summary_override,
        )
    return build_legacy_dimension_b_judge_payload(
        execution_record=execution_record,
        expected_task=expected_task,
        rationale_for_llm_judge=rationale_for_llm_judge,
        task_spec=task_spec,
        run_summary_override=run_summary_override,
    )


def build_context_reduced_run_summary(
    execution_record: ExecutionRecord,
) -> dict[str, Any]:
    if not isinstance(execution_record.raw_state, dict):
        return execution_record.run_summary or legacy_dimension_b_evidence(
            execution_record
        )

    task_text = None
    existing_summary = execution_record.run_summary
    if isinstance(existing_summary, dict):
        run = existing_summary.get("run")
        if isinstance(run, dict) and isinstance(run.get("task"), str):
            task_text = run["task"]

    return build_run_summary_from_state(
        final_state=execution_record.raw_state,
        task=task_text,
        raw_endpoint=None,
        reduction_threshold_tokens=1,
    )


def build_context_reduced_dimension_b_judge_payload(
    *,
    execution_record: ExecutionRecord,
    expected_task: str,
    rationale_for_llm_judge: str,
    task_spec: dict | None,
) -> dict[str, Any]:
    return build_dimension_b_judge_payload(
        execution_record=execution_record,
        expected_task=expected_task,
        rationale_for_llm_judge=rationale_for_llm_judge,
        task_spec=task_spec,
        run_summary_override=build_context_reduced_run_summary(execution_record),
    )


def default_judge_subgraph_task(
    execution_record: ExecutionRecord,
    expected_task: str = "",
    expected_task_complexity: str = "",
    rationale_for_llm_judge: str = "",
    task_spec: dict | None = None,
    **_: Any,
) -> dict:
    effective_expected_task = expected_task or expected_task_complexity or ""

    try:
        llm = get_dimension_b_judge_llm()
        judge_payload = build_dimension_b_judge_payload(
            execution_record=execution_record,
            expected_task=effective_expected_task,
            rationale_for_llm_judge=rationale_for_llm_judge,
            task_spec=task_spec,
        )

        def build_human_message(payload: dict[str, Any]) -> HumanMessage:
            return HumanMessage(
                content=(
                    "Evaluate the following execution for dimension_B_task.\n"
                    "Return structured output only.\n\n"
                    f"{json.dumps(payload, indent=2, default=str)}"
                )
            )

        system = SystemMessage(
            content=(
                "You are grading dimension_B_task.\n"
                "Your job is to judge whether the whole graph performed what was asked.\n"
                "Use only the evidence provided.\n"
                "Do not judge formatting unless it affects task completion.\n\n"
                "If run_summary is present, treat it as the primary evidence:\n"
                "it contains the graph execution, conversation, routes, subgraphs, tools,\n"
                "artifacts, final answer, and debug metadata.\n\n"
                "Important code_sandbox evidence rule:\n"
                "code_sandbox is an execution subgraph, not a normal nested tool-calling subgraph.\n"
                "A completed code_sandbox route may legitimately have empty tool_event_ids or\n"
                "tool_names for nested graph tools. Do not penalize code_sandbox solely because\n"
                "no ordinary tool calls are listed. For code_sandbox, judge grounding from\n"
                "sandbox-specific evidence instead: the subgraph sandbox_execution block,\n"
                "synthetic code_sandbox_execute_python evidence events, sandbox_result.json\n"
                "payloads, exit_code, stdout, stderr, run_dir, generated artifacts or\n"
                "registered attachments, and code_sandbox_summary_internal or withdrawal\n"
                "messages. Treat sandbox output as ungrounded only when the sandbox result is\n"
                "missing, failed, inconsistent with the final answer, or asserted only in prose\n"
                "without structured sandbox execution evidence.\n\n"
                "When writing the rationale, be specific and detailed enough for manual debugging.\n"
                "Name the concrete evidence you used: routes selected, subgraph tasks, tools called\n"
                "or sandbox execution outputs, artifacts created or missing, final-answer content,\n"
                "and any unsupported, redundant,\n"
                "wrong, missing, or hallucinated step. Explain why those observations justify the score.\n"
                "If the task failed or partially failed, identify the most important failure points and\n"
                "what the graph should have done instead. Avoid vague statements like 'incomplete' unless\n"
                "you immediately ground them in observed execution details.\n\n"
                "Score rubric:\n"
                "1 = the task was not meaningfully completed or with a major failure \n"
                "2 = only minimal or incorrect progress, important gaps remain \n"
                "3 = partial completion, with only minor gaps \n"
                "4 = partial completion with small allucination\n"
                "5 = fully completed end-to-end\n"
            )
        )

        human = build_human_message(judge_payload)

        structured_llm = llm.with_structured_output(DimensionBJudgeOutput)

        verdict = None
        last_judge_error: Exception | None = None
        context_compaction_used = False
        attempt = 0
        max_attempts = DIMENSION_B_JUDGE_MAX_ATTEMPTS
        while attempt < max_attempts:
            attempt += 1
            try:
                verdict = structured_llm.invoke([system, human])
                if verdict is None:
                    raise ValueError("judge returned no structured verdict")
                if getattr(verdict, "score", None) is None:
                    raise ValueError("judge returned a structured verdict without score")
                break
            except Exception as judge_error:
                if is_ollama_usage_limit_exception(judge_error):
                    raise
                last_judge_error = judge_error
                verdict = None
                if (
                    is_llm_context_length_error(judge_error)
                    and not context_compaction_used
                ):
                    judge_payload = build_context_reduced_dimension_b_judge_payload(
                        execution_record=execution_record,
                        expected_task=effective_expected_task,
                        rationale_for_llm_judge=rationale_for_llm_judge,
                        task_spec=task_spec,
                    )
                    context_compaction_used = True
                    max_attempts += 1
                    human = build_human_message(judge_payload)
                    logging.getLogger(__name__).warning(
                        "Dimension-B judge prompt exceeded context; retrying with "
                        "forced-reduced run summary for task %s.",
                        execution_record.task_id,
                    )
                    continue
                if attempt >= max_attempts:
                    normalized_score = normalize_score_1_to_5(1)
                    return {
                        "status": STATUS_FAILED,
                        "reason": (
                            "dimension_B_task judge failed to produce a valid "
                            f"structured verdict after {attempt} attempts: "
                            f"{judge_error}"
                        ),
                        "details": {
                            "raw_score_1_to_5": 1,
                            "normalized_score_0_to_1": normalized_score,
                            "pass_threshold_1_to_5": JUDGE_PASS_THRESHOLD,
                            "rationale": (
                                "The evaluator could not obtain a valid structured "
                                "Dimension-B judge verdict. This is treated as a failed "
                                "Dimension-B criterion rather than an evaluator error so "
                                "the rest of the task remains scoreable."
                            ),
                            "expected_task": effective_expected_task,
                            "rationale_for_llm_judge": rationale_for_llm_judge,
                            "judge_payload_format": judge_payload.get("payload_format"),
                            "judge_attempts": attempt,
                            "judge_error": str(last_judge_error),
                            "judge_context_compaction_used": context_compaction_used,
                        },
                    }

        raw_score = int(verdict.score)
        normalized_score = normalize_score_1_to_5(raw_score)

        return {
            "status": STATUS_PASSED
            if raw_score >= JUDGE_PASS_THRESHOLD
            else STATUS_FAILED,
            "details": {
                "raw_score_1_to_5": raw_score,
                "normalized_score_0_to_1": normalized_score,
                "pass_threshold_1_to_5": JUDGE_PASS_THRESHOLD,
                "rationale": verdict.rationale,
                "expected_task": effective_expected_task,
                "rationale_for_llm_judge": rationale_for_llm_judge,
                "judge_payload_format": judge_payload.get("payload_format"),
                "judge_attempts": attempt,
                "judge_context_compaction_used": context_compaction_used,
            },
        }

    except Exception as e:
        if is_ollama_usage_limit_exception(e):
            raise
        return {
            "status": STATUS_ERROR,
            "reason": f"dimension_B_task judge failed: {e}",
            "details": {
                "expected_task": effective_expected_task,
                "rationale_for_llm_judge": rationale_for_llm_judge,
            },
        }


def extract_subgraph_task_requests(record: ExecutionRecord) -> list[dict[str, str]]:
    requests: list[dict[str, str]] = []
    for entry in record.run_log:
        route = entry.get("route")
        subgraph_task = entry.get("subgraph_task")
        if (
            isinstance(route, str)
            and route
            and isinstance(subgraph_task, str)
            and subgraph_task.strip()
        ):
            requests.append(
                {
                    "route": route,
                    "subgraph_task": subgraph_task.strip(),
                }
            )
    return requests


def normalize_judge_payload(judged: dict) -> dict:
    if "status" in judged:
        return {
            "status": judged["status"],
            "details": judged.get("details", {}),
            **({"reason": judged["reason"]} if "reason" in judged else {}),
        }

    if "passed" in judged:
        return result(
            STATUS_PASSED if judged["passed"] else STATUS_FAILED,
            details=judged.get("details", {}),
        )

    raise ValueError("judge returned unsupported payload")


def evaluate_subgraph_task(
    cfg: dict,
    task: dict,
    record: ExecutionRecord,
) -> dict:
    criterion_ids = RUBRIC_BY_SECTION["subgraph_task"]
    expected_task = cfg.get("expected_task", "")
    rationale_for_llm_judge = cfg.get("rationale_for_llm_judge", "")

    if all(
        is_unconstrained(value) for value in (expected_task, rationale_for_llm_judge)
    ):
        return scored_result(
            STATUS_SKIPPED,
            criterion_ids,
            {},
            reason="no subgraph_task constraints",
            not_applicable=True,
        )

    task_requests = extract_subgraph_task_requests(record)
    seen_requests: set[tuple[str, str]] = set()
    duplicate_requests: list[dict[str, str]] = []
    for request in task_requests:
        key = (request["route"], request["subgraph_task"])
        if key in seen_requests:
            duplicate_requests.append(request)
        else:
            seen_requests.add(key)

    redundancy_passed = not duplicate_requests
    details: dict[str, Any] = {
        "task_requests": task_requests,
        "redundant_task_requests": duplicate_requests,
        "redundancy_point": 1 if redundancy_passed else 0,
    }

    judge = default_judge_subgraph_task
    judged = judge(
        execution_record=record,
        expected_task=expected_task,
        rationale_for_llm_judge=rationale_for_llm_judge,
        task_spec=task,
    )
    normalized_judge = normalize_judge_payload(judged)
    details["llm_judge"] = normalized_judge

    judge_status = normalized_judge["status"]
    points = {
        "B1_task_scope_adherence": judge_status == STATUS_PASSED,
        "B2_no_redundant_task": redundancy_passed,
    }
    if judge_status == STATUS_ERROR:
        return scored_result(STATUS_ERROR, criterion_ids, points, details=details)
    if judge_status == STATUS_FAILED or not redundancy_passed:
        return scored_result(STATUS_FAILED, criterion_ids, points, details=details)
    if judge_status == STATUS_NOT_EVALUATED:
        return scored_result(
            STATUS_NOT_EVALUATED, criterion_ids, points, details=details
        )
    return scored_result(STATUS_PASSED, criterion_ids, points, details=details)


def normalize_tool_name(tool_call: dict) -> str | None:
    return (
        tool_call.get("name")
        or tool_call.get("tool_name")
        or tool_call.get("function", {}).get("name")
    )


def canonical_tool_name(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    if not normalized:
        return None
    if normalized.startswith("main_graph_tool_"):
        normalized = normalized.removeprefix("main_graph_tool_")
    return normalized


def canonical_tool_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    canonical: list[str] = []
    for value in values:
        name = canonical_tool_name(value)
        if name:
            canonical.append(name)
    return canonical


def tool_call_was_successful(tool_call: dict) -> bool:
    return tool_call.get("success") is not False


def tool_calls_for_evaluation(record: ExecutionRecord) -> list[dict]:
    executed_calls = [
        tool_call for tool_call in record.tool_calls if "success" in tool_call
    ]
    return executed_calls or record.tool_calls


def contains_subsequence(items: list[str], subseq: list[str]) -> bool:
    if not subseq:
        return True
    idx = 0
    for item in items:
        if item == subseq[idx]:
            idx += 1
            if idx == len(subseq):
                return True
    return False


def evaluate_tools(cfg: dict, record: ExecutionRecord) -> dict:
    criterion_ids = RUBRIC_BY_SECTION["tools"]
    requires_sandbox = cfg.get("requires_sandbox", "")
    expected_tools = canonical_tool_list(cfg.get("expected_tools", []))
    forbidden_tools = canonical_tool_list(cfg.get("forbidden_tools", []))
    enforce_expected_tool_set = bool(cfg.get("enforce_expected_tool_set", False))
    required_tool_sequence = canonical_tool_list(
        cfg.get(
            "required_tool_calls_in_order",
            cfg.get("required_tool_sequence", []),
        )
    )

    if all(
        is_unconstrained(value)
        for value in (
            requires_sandbox,
            expected_tools,
            forbidden_tools,
            required_tool_sequence,
        )
    ):
        return scored_result(
            STATUS_SKIPPED,
            criterion_ids,
            {},
            reason="no tool constraints",
            not_applicable=True,
        )

    raw_observed_tools = []
    raw_successful_tools = []
    observed_tools = []
    successful_tools = []
    ignored_tools = []
    for tool_call in tool_calls_for_evaluation(record):
        raw_name = normalize_tool_name(tool_call)
        name = canonical_tool_name(raw_name)
        if not name:
            continue
        raw_observed_tools.append(raw_name)
        if name in INFRASTRUCTURE_TOOL_NAMES:
            ignored_tools.append(name)
            continue
        observed_tools.append(name)
        if tool_call_was_successful(tool_call):
            raw_successful_tools.append(raw_name)
            successful_tools.append(name)

    executed_routes = extract_executed_routes(record)
    sandbox_used = "code_sandbox" in executed_routes
    sandbox_attempts = 0
    if isinstance(record.raw_state, dict):
        try:
            sandbox_attempts = int(record.raw_state.get("sandbox_attempts") or 0)
        except (TypeError, ValueError):
            sandbox_attempts = 0

    forbidden_hits = [tool for tool in observed_tools if tool in forbidden_tools]

    missing_expected: list[str] = []
    unexpected_tools: list[str] = []
    if not is_unconstrained(expected_tools):
        missing_expected = [
            tool for tool in expected_tools if tool not in successful_tools
        ]
        unexpected_tools = [
            tool for tool in observed_tools if tool not in expected_tools
        ]

    c1_passed = (
        not forbidden_hits
        and not missing_expected
        and (not enforce_expected_tool_set or not unexpected_tools)
    )

    if is_unconstrained(required_tool_sequence):
        c2_passed = True
    else:
        c2_passed = contains_subsequence(successful_tools, required_tool_sequence)

    if requires_sandbox is True:
        c3_passed = sandbox_used and sandbox_attempts <= 1
    elif requires_sandbox is False:
        c3_passed = not sandbox_used
    else:
        c3_passed = True

    has_tool_constraints = not all(
        is_unconstrained(value)
        for value in (expected_tools, forbidden_tools, required_tool_sequence)
    )
    section_passed = (c1_passed and c2_passed) if has_tool_constraints else c3_passed

    details = {
        "raw_observed_tools": raw_observed_tools,
        "raw_successful_tools": raw_successful_tools,
        "observed_tools": observed_tools,
        "successful_tools": successful_tools,
        "ignored_infrastructure_tools": ignored_tools,
        "executed_routes": executed_routes,
        "points": {
            "C1_correct_tools_called": 1 if c1_passed else 0,
            "C2_correct_tool_order": 1 if c2_passed else 0,
            "C3_sandbox_first_attempt": 1 if c3_passed else 0,
        },
        "C1": {
            "passed": c1_passed,
            "expected_tools": expected_tools,
            "missing_expected_tools": missing_expected,
            "forbidden_tools": forbidden_tools,
            "unexpected_tools": unexpected_tools,
            "forbidden_hits": forbidden_hits,
            "enforce_expected_tool_set": enforce_expected_tool_set,
        },
        "C2": {
            "passed": c2_passed,
            "required_tool_calls_in_order": required_tool_sequence,
        },
        "C3": {
            "passed": c3_passed,
            "requires_sandbox": requires_sandbox,
            "sandbox_used": sandbox_used,
            "sandbox_attempts": sandbox_attempts,
            "note": (
                "When requires_sandbox is true, C3 passes only if code_sandbox "
                "was used and completed within the first sandbox attempt. When "
                "requires_sandbox is false, C3 passes only if code_sandbox was avoided."
            ),
        },
    }

    return scored_result(
        STATUS_PASSED if section_passed else STATUS_FAILED,
        criterion_ids,
        {
            "C1_correct_tools_called": c1_passed,
            "C2_correct_tool_order": c2_passed,
            "C3_sandbox_first_attempt": c3_passed,
        },
        details=details,
    )


def collect_artifact_paths(record: ExecutionRecord) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()

    for path in record.artifacts:
        norm = os.path.normpath(str(path))
        if norm not in seen:
            seen.add(norm)
            files.append(norm)

    if not files and record.workspace_dir:
        workspace = Path(record.workspace_dir)
        if workspace.exists():
            for file_path in workspace.rglob("*"):
                if file_path.is_file():
                    norm = os.path.normpath(str(file_path))
                    if norm not in seen:
                        seen.add(norm)
                        files.append(norm)

    return files


def evaluate_artifacts(cfg: dict, record: ExecutionRecord) -> dict:
    criterion_ids = RUBRIC_BY_SECTION["artifacts"]
    requires = cfg.get("requires_artifacts", "")
    groups = cfg.get("required_extension_groups", [])

    if requires in ("", None):
        return scored_result(
            STATUS_SKIPPED,
            criterion_ids,
            {},
            reason="requires_artifacts not specified",
            not_applicable=True,
        )
    if requires is False:
        return scored_result(
            STATUS_PASSED,
            criterion_ids,
            {},
            details={"checked": False},
            not_applicable=True,
        )

    files = collect_artifact_paths(record)
    registered_files = [os.path.normpath(str(path)) for path in record.artifacts]
    if not files:
        return scored_result(
            STATUS_FAILED,
            criterion_ids,
            {
                "D1_artifacts_created": False,
                "D2_artifact_type_correct": False,
                "D3_artifact_registered_in_session": False,
            },
            details={
                "reason": "no artifacts found",
                "artifact_count": 0,
                "registered_artifact_count": len(registered_files),
            },
        )

    normalized = [str(path).lower() for path in files]
    registered_normalized = [str(path).lower() for path in registered_files]
    if not groups:
        d1_passed = bool(normalized)
        d2_passed = d1_passed
        d3_passed = bool(registered_normalized)
        return scored_result(
            STATUS_PASSED if d1_passed and d2_passed and d3_passed else STATUS_FAILED,
            criterion_ids,
            {
                "D1_artifacts_created": d1_passed,
                "D2_artifact_type_correct": d2_passed,
                "D3_artifact_registered_in_session": d3_passed,
            },
            details={
                "artifact_count": len(normalized),
                "registered_artifact_count": len(registered_normalized),
            },
        )

    group_results = []
    overall = True
    registered_group_overall = True
    for group in groups:
        exts = [ext.lower() for ext in group.get("extensions", [])]
        min_count = int(group.get("min_count", 0))
        count = sum(
            1
            for file_name in normalized
            if any(file_name.endswith(ext) for ext in exts)
        )
        registered_count = sum(
            1
            for file_name in registered_normalized
            if any(file_name.endswith(ext) for ext in exts)
        )
        passed = count >= min_count
        registered_passed = registered_count >= min_count
        overall = overall and passed
        registered_group_overall = registered_group_overall and registered_passed
        group_results.append(
            {
                "extensions": exts,
                "min_count": min_count,
                "observed_count": count,
                "registered_count": registered_count,
                "passed": passed,
                "registered": registered_passed,
            }
        )

    d1_passed = bool(normalized)
    d2_passed = overall
    d3_passed = bool(registered_normalized) and registered_group_overall
    return scored_result(
        STATUS_PASSED if d1_passed and d2_passed and d3_passed else STATUS_FAILED,
        criterion_ids,
        {
            "D1_artifacts_created": d1_passed,
            "D2_artifact_type_correct": d2_passed,
            "D3_artifact_registered_in_session": d3_passed,
        },
        details={
            "artifact_count": len(normalized),
            "registered_artifact_count": len(registered_normalized),
            "groups": group_results,
        },
    )


def iter_balanced_json_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    pairs = {"{": "}", "[": "]"}
    for start, char in enumerate(text):
        if char not in pairs:
            continue

        stack = [pairs[char]]
        in_string = False
        escape = False
        quote = ""
        for index in range(start + 1, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == quote:
                    in_string = False
                continue

            if current in {"'", '"'}:
                in_string = True
                quote = current
            elif current in pairs:
                stack.append(pairs[current])
            elif stack and current == stack[-1]:
                stack.pop()
                if not stack:
                    fragments.append(text[start : index + 1])
                    break

    return fragments


PARSE_FAILED = object()


def try_parse_json_fragment(fragment: str) -> Any:
    try:
        return json.loads(fragment)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(fragment)
    except Exception:
        return PARSE_FAILED

    if isinstance(parsed, (dict, list, int, float, bool)) or parsed is None:
        return parsed
    return PARSE_FAILED


def try_parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    parsed = try_parse_json_fragment(stripped)
    if parsed is not PARSE_FAILED:
        return parsed

    fragments = iter_balanced_json_fragments(stripped)
    dict_candidates = []
    other_candidates = []
    for fragment in fragments:
        parsed = try_parse_json_fragment(fragment)
        if parsed is PARSE_FAILED:
            continue
        if isinstance(parsed, dict):
            dict_candidates.append(parsed)
        else:
            other_candidates.append(parsed)

    if dict_candidates:
        return dict_candidates[0]
    if other_candidates:
        return other_candidates[0]
    return value


def extract_variables(
    parsed_answer: Any, variables_cfg: dict[str, str]
) -> dict[str, Any]:
    if not isinstance(parsed_answer, dict):
        return {}
    return {key: parsed_answer.get(key) for key in variables_cfg}


def _numeric_pair_from_slice_range(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None

    if isinstance(value, dict):
        start = value.get("slice_start", value.get("start"))
        end = value.get("slice_end", value.get("end"))
        try:
            return int(start), int(end)
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])

    return None


def slice_range_matches(
    slice_range: Any,
    slice_start: Any,
    slice_end: Any,
    expected_start: int,
    expected_end: int,
) -> bool:
    try:
        if int(slice_start) == expected_start and int(slice_end) == expected_end:
            return True
    except (TypeError, ValueError):
        pass

    pair = _numeric_pair_from_slice_range(slice_range)
    return pair == (expected_start, expected_end)


def is_int(value: Any) -> bool:
    """Return whether *value* is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_positive_int(value: Any) -> bool:
    return is_int(value) and value > 0


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_string(value: Any) -> bool:
    return isinstance(value, str)


def is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_positive_finite_number(value: Any) -> bool:
    return is_finite_number(value) and value > 0


def is_finite_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(is_finite_number(item) for item in value)


def is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(is_int(item) for item in value)


def number_or_none_in_range(value: Any, lower: float, upper: float) -> bool:
    """Match the permissive numeric parsing used by the historical verifiers."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "null":
            return True
        try:
            value = float(stripped)
        except ValueError:
            return False
    return is_finite_number(value) and lower <= float(value) <= upper


def numbers_close(
    first: Any,
    second: Any,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 0.0,
) -> bool:
    if not is_finite_number(first) or not is_finite_number(second):
        return False
    return math.isclose(
        float(first),
        float(second),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def is_valid_numpy_dtype(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        import numpy as np

        np.dtype(value.strip())
    except (ImportError, TypeError, ValueError):
        return False
    return True


def safe_eval_rule(
    rule: str,
    variables: dict[str, Any],
    parsed_answer: Any | None = None,
) -> bool:
    answer = parsed_answer if isinstance(parsed_answer, dict) else variables
    safe_context = {
        **variables,
        "answer": answer,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "all": all,
        "any": any,
        "sorted": sorted,
        "range": range,
        "slice_range_matches": slice_range_matches,
        "is_int": is_int,
        "is_positive_int": is_positive_int,
        "is_bool": is_bool,
        "is_string": is_string,
        "is_dict": is_dict,
        "is_finite_number": is_finite_number,
        "is_positive_finite_number": is_positive_finite_number,
        "is_finite_number_list": is_finite_number_list,
        "is_int_list": is_int_list,
        "number_or_none_in_range": number_or_none_in_range,
        "numbers_close": numbers_close,
        "is_valid_numpy_dtype": is_valid_numpy_dtype,
        "True": True,
        "False": False,
        "None": None,
    }
    safe_globals = {"__builtins__": {}, **safe_context}
    return bool(eval(rule, safe_globals, safe_context))


def evaluate_rule(
    rule: str,
    variables: dict[str, Any],
    parsed_answer: Any | None = None,
) -> dict[str, Any]:
    try:
        return {
            "rule": rule,
            "passed": safe_eval_rule(rule, variables, parsed_answer),
        }
    except Exception as exc:
        return {
            "rule": rule,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_verifier(
    file_path: str | Path, function_name: str = "verify"
) -> Callable[..., Any]:
    module_name = f"perf_eval_verifier_{abs(hash((str(file_path), function_name)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def normalize_verifier_result(verifier_result: Any) -> dict:
    if isinstance(verifier_result, bool):
        return result(STATUS_PASSED if verifier_result else STATUS_FAILED)
    if isinstance(verifier_result, dict) and "passed" in verifier_result:
        return result(
            STATUS_PASSED if verifier_result["passed"] else STATUS_FAILED,
            details=verifier_result.get("details", {}),
        )
    raise ValueError("verifier returned unsupported payload")


def call_verifier(
    verifier_fn: Callable[..., Any],
    final_answer: Any,
    task_spec: dict,
    execution_record: ExecutionRecord,
) -> Any:
    try:
        return verifier_fn(
            final_answer,
            task_spec=task_spec,
            execution_record=execution_record,
        )
    except TypeError:
        return verifier_fn(final_answer)


def evaluate_final_output(
    cfg: dict,
    task: dict,
    record: ExecutionRecord,
    spec_dir: str,
) -> dict:
    criterion_ids = RUBRIC_BY_SECTION["final_output"]
    if all(
        is_unconstrained(cfg.get(k))
        for k in (
            "strategy",
            "variables_to_extract",
            "rules",
            "python_verification",
            "biological_note",
        )
    ):
        return scored_result(
            STATUS_SKIPPED,
            criterion_ids,
            {},
            reason="no final_output constraints",
            not_applicable=True,
        )

    py_cfg = cfg.get("python_verification", {})
    if not is_unconstrained(py_cfg):
        verifier_path = Path(spec_dir) / py_cfg["file"]
        verifier_fn = load_verifier(verifier_path, py_cfg.get("function", "verify"))
        verifier_result = call_verifier(verifier_fn, record.final_answer, task, record)
        normalized = normalize_verifier_result(verifier_result)
        normalized["details"] = {
            "verifier_file": str(verifier_path),
            **normalized.get("details", {}),
        }
        return scored_result(
            normalized["status"],
            criterion_ids,
            {"E1_final_output_correct": normalized["status"] == STATUS_PASSED},
            details=normalized.get("details", {}),
            reason=normalized.get("reason"),
        )

    parsed_answer = try_parse_json_like(record.final_answer)
    variables = extract_variables(
        parsed_answer, cfg.get("variables_to_extract", {}) or {}
    )
    rules = cfg.get("rules", []) or []

    rule_results = []
    rule_context = (
        variables
        if variables
        else parsed_answer
        if isinstance(parsed_answer, dict)
        else {}
    )
    for rule in rules:
        rule_results.append(evaluate_rule(rule, rule_context, parsed_answer))

    overall = all(item["passed"] for item in rule_results) if rule_results else True
    return scored_result(
        STATUS_PASSED if overall else STATUS_FAILED,
        criterion_ids,
        {"E1_final_output_correct": overall},
        details={
            "parsed_answer": parsed_answer,
            "variables": variables,
            "rules": rule_results,
        },
    )


def aggregate_task_status(section_results: dict[str, dict]) -> str:
    statuses = [section["status"] for section in section_results.values()]
    if STATUS_ERROR in statuses:
        return STATUS_ERROR
    if STATUS_FAILED in statuses:
        return STATUS_FAILED
    return STATUS_PASSED


def evaluate_task(
    task: dict,
    record: ExecutionRecord,
    spec_dir: str,
) -> dict:
    evaluation = task["evaluation"]
    sections = {
        "routing": evaluate_routing(evaluation["dimension_A_routing"], record),
        "subgraph_task": evaluate_subgraph_task(
            evaluation["dimension_B_task"],
            task,
            record,
        ),
        "tools": evaluate_tools(evaluation["dimension_C_tools"], record),
        "artifacts": evaluate_artifacts(evaluation["dimension_D_artifacts"], record),
        "final_output": evaluate_final_output(
            evaluation["dimension_E_final_output"],
            task,
            record,
            spec_dir,
        ),
    }
    score = aggregate_score(sections)
    return {
        "task_id": task["id"],
        "status": aggregate_task_status(sections),
        "score": score,
        "request_text": build_task_request(task),
        "execution_record": execution_record_payload(record),
        "sections": sections,
        "errors": [],
    }


def filter_tasks_by_id(tasks: list[dict], task_ids: list[str] | None) -> list[dict]:
    if not task_ids:
        return tasks

    allowed = set(task_ids)
    known = {task["id"] for task in tasks}
    known_sources = {
        task["source_task_id"]
        for task in tasks
        if isinstance(task.get("source_task_id"), str)
    }
    unknown = sorted(allowed - known - known_sources)
    if unknown:
        raise ValueError(f"unknown task id(s): {unknown}")

    return [
        task
        for task in tasks
        if task["id"] in allowed or task.get("source_task_id") in allowed
    ]


def execution_record_payload(record: ExecutionRecord | None) -> dict[str, Any]:
    if record is None:
        return {
            "final_answer": "",
            "run_log": [],
            "tool_calls": [],
            "artifacts": [],
            "run_summary": None,
            "workspace_dir": None,
        }

    return {
        "final_answer": record.final_answer,
        "run_log": record.run_log,
        "tool_calls": record.tool_calls,
        "artifacts": record.artifacts,
        "execution_trace": record.execution_trace,
        "run_summary": record.run_summary,
        "workspace_dir": record.workspace_dir,
    }


def build_error_result(
    task: dict,
    error: Exception,
    record: ExecutionRecord | None = None,
) -> dict:
    task_id = task.get("id", "<unknown>")
    sections: dict[str, dict] = {}
    return {
        "task_id": task_id,
        "status": STATUS_ERROR,
        "score": aggregate_score(sections),
        "request_text": build_task_request(task) if isinstance(task, dict) else "",
        "execution_record": execution_record_payload(record),
        "sections": sections,
        "errors": [
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        ],
    }


def build_skipped_result(
    task: dict,
    reason: str,
    record: ExecutionRecord | None = None,
) -> dict:
    task_id = task.get("id", "<unknown>")
    sections: dict[str, dict] = {}
    return {
        "task_id": task_id,
        "status": STATUS_SKIPPED,
        "score": aggregate_score(sections),
        "request_text": build_task_request(task) if isinstance(task, dict) else "",
        "execution_record": execution_record_payload(record),
        "sections": sections,
        "skip_reason": reason,
        "errors": [],
    }


def format_error_for_stderr(task_id: str, error: Exception) -> str:
    traceback_text = traceback.format_exc().rstrip()
    return (
        f"[ERROR] Test ID: {task_id} crashed with "
        f"{type(error).__name__}: {error}\n{traceback_text}"
    )


def safe_path_slug(value: Any, *, fallback: str = "run", max_length: int = 120) -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return (text[:max_length] or fallback).strip("._-") or fallback


def create_evaluator_batch_dir(
    *,
    spec_file: str | Path,
    prefix: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    spec_path = Path(spec_file)
    batch_prefix = safe_path_slug(prefix or spec_path.stem, fallback="evaluation")
    timestamp = local_now().strftime("%d-%m-%Y_%H-%M-%S")
    root = Path(base_dir or DEFAULT_EVALUATOR_RUNS_DIR).expanduser().resolve()
    batch_dir = root / f"{batch_prefix}_{timestamp}"
    counter = 2
    while batch_dir.exists():
        batch_dir = root / f"{batch_prefix}_{timestamp}_{counter}"
        counter += 1
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def extract_run_id(result_item: dict[str, Any]) -> str:
    execution_record = result_item.get("execution_record") or {}
    execution_trace = execution_record.get("execution_trace") or {}
    run_id = execution_trace.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    return f"run-{safe_path_slug(result_item.get('task_id'), fallback='unknown')}"


def write_json_file(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def build_judge_response_payload(result_item: dict[str, Any]) -> dict[str, Any]:
    subgraph_task = (result_item.get("sections") or {}).get("subgraph_task") or {}
    llm_judge = ((subgraph_task.get("details") or {}).get("llm_judge")) or {}
    judge_details = llm_judge.get("details") or {}
    score = (subgraph_task.get("details") or {}).get("score") or {}
    return {
        "task_id": result_item.get("task_id"),
        "run_id": extract_run_id(result_item),
        "judge_model": get_dimension_b_judge_model_name(),
        "judge_backend": "ollama" if dimension_b_judge_uses_ollama() else "openai",
        "run_summary_reduction_threshold_tokens": (
            dimension_b_run_summary_threshold_tokens()
        ),
        "judge_payload_format": judge_details.get("judge_payload_format"),
        "status": llm_judge.get("status"),
        "raw_score_1_to_5": judge_details.get("raw_score_1_to_5"),
        "normalized_score_0_to_1": judge_details.get("normalized_score_0_to_1"),
        "pass_threshold_1_to_5": judge_details.get("pass_threshold_1_to_5"),
        "rationale": judge_details.get("rationale"),
        "expected_task": judge_details.get("expected_task"),
        "rationale_for_llm_judge": judge_details.get("rationale_for_llm_judge"),
        "subgraph_task_score": {
            "earned": score.get("earned"),
            "max": score.get("max"),
            "points": (subgraph_task.get("details") or {}).get("points", {}),
        },
        "error": llm_judge.get("reason"),
    }


def export_task_run_artifacts(
    batch_dir: Path,
    result_item: dict[str, Any],
    *,
    index: int,
) -> Path:
    task_id = safe_path_slug(result_item.get("task_id"), fallback=f"task_{index:03d}")
    run_id = safe_path_slug(extract_run_id(result_item), fallback=f"run_{index:03d}")
    run_dir = batch_dir / f"{index:03d}_{task_id}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result_item["evaluator_run_dir"] = str(run_dir)

    execution_record = result_item.get("execution_record") or {}
    write_json_file(
        run_dir / "judge_response.json",
        build_judge_response_payload(result_item),
    )
    write_json_file(run_dir / "result.json", result_item)

    execution_trace = execution_record.get("execution_trace")
    if execution_trace is not None:
        write_json_file(run_dir / "execution_trace.json", execution_trace)

    run_summary = execution_record.get("run_summary")
    if run_summary is not None:
        write_json_file(run_dir / "run_summary.json", run_summary)

    return run_dir


def evaluate_tasks(
    spec_file: str,
    task_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    workspace_root: str | Path | None = None,
    task_delay_seconds: float = 0.0,
    batch_dir: str | Path | None = None,
    usage_limit_probe_interval_seconds: float | None = None,
    usage_limit_max_wait_seconds: float | None = None,
) -> dict:
    spec_path = Path(spec_file).resolve()
    if tasks is None:
        tasks = load_task_spec(spec_path)
    else:
        tasks = expand_list_valued_task_fields(expand_patient_id_variants(tasks))
    validate_task_spec(tasks)
    tasks = filter_tasks_by_id(tasks, task_ids)

    results = []
    batch_path = Path(batch_dir).expanduser().resolve() if batch_dir else None
    graph_trace_base_dir = batch_path / "raw_graph_traces" if batch_path else None
    if usage_limit_probe_interval_seconds is None:
        usage_limit_probe_interval_seconds = _env_non_negative_float(
            USAGE_LIMIT_PROBE_INTERVAL_ENV,
            DEFAULT_USAGE_LIMIT_PROBE_INTERVAL_SECONDS,
        )
    if usage_limit_max_wait_seconds is None:
        usage_limit_max_wait_seconds = _env_non_negative_float(
            USAGE_LIMIT_MAX_WAIT_ENV,
            DEFAULT_USAGE_LIMIT_MAX_WAIT_SECONDS,
        )

    stop_reason: str | None = None
    index = 0
    while index < len(tasks):
        task = tasks[index]
        display_index = index + 1
        task_id = task["id"]
        task_name = task.get("name") or "<unnamed>"
        if stop_reason is not None:
            print(
                f"[SKIPPED] Test {display_index}/{len(tasks)} | ID: {task_id} | "
                f"Name: {task_name} | Reason: {stop_reason}",
                file=sys.stderr,
                flush=True,
            )
            result_item = build_skipped_result(task, stop_reason)
            results.append(result_item)
            if batch_path:
                export_task_run_artifacts(
                    batch_path,
                    result_item,
                    index=display_index,
                )
            index += 1
            continue

        print(
            f"[STARTING] Test {display_index}/{len(tasks)} | ID: {task_id} | Name: {task_name}",
            file=sys.stderr,
            flush=True,
        )

        record: ExecutionRecord | None = None
        try:
            record = run_graph_for_task(
                task,
                get_compiled_graph(),
                workspace_root=workspace_root,
                execution_trace_base_dir=graph_trace_base_dir,
            )
            result_item = evaluate_task(
                task=task,
                record=record,
                spec_dir=str(spec_path.parent),
            )
            results.append(result_item)
        except Exception as exc:
            if is_ollama_usage_limit_exception(exc):
                print(
                    f"[OLLAMA LIMIT] Test {display_index}/{len(tasks)} | "
                    f"ID: {task_id} hit an Ollama usage limit. Pausing the "
                    "evaluator and probing with a sample request before retrying.",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    wait_for_ollama_usage_limit_recovery(
                        probe_interval_seconds=usage_limit_probe_interval_seconds,
                        max_wait_seconds=usage_limit_max_wait_seconds,
                    )
                except Exception as recovery_error:
                    print(
                        format_error_for_stderr(task_id, recovery_error),
                        file=sys.stderr,
                        flush=True,
                    )
                    result_item = build_error_result(task, recovery_error, record)
                    results.append(result_item)
                    stop_reason = (
                        "Ollama usage limit recovery probe failed; skipping "
                        "remaining tasks to avoid repeated provider failures."
                    )
                else:
                    print(
                        f"[OLLAMA LIMIT] Recovery probe succeeded; retrying "
                        f"test {display_index}/{len(tasks)} | ID: {task_id}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
            else:
                print(
                    format_error_for_stderr(task_id, exc),
                    file=sys.stderr,
                    flush=True,
                )
                result_item = build_error_result(task, exc, record)
                results.append(result_item)

        if batch_path:
            export_task_run_artifacts(batch_path, result_item, index=display_index)

        index += 1
        if task_delay_seconds > 0 and index < len(tasks):
            time.sleep(task_delay_seconds)

    n_passed = sum(1 for item in results if item["status"] == STATUS_PASSED)
    n_failed = sum(1 for item in results if item["status"] == STATUS_FAILED)
    n_errors = sum(1 for item in results if item["status"] == STATUS_ERROR)
    total_score = sum((item.get("score") or {}).get("earned", 0) for item in results)
    total_max_score = sum(
        (item.get("score") or {}).get("max", RUBRIC_MAX_POINTS) for item in results
    )

    summary = {
        "n_tasks": len(tasks),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_errors": n_errors,
        "score": {
            "earned": total_score,
            "max": total_max_score,
            "percentage": round(total_score / total_max_score, 4)
            if total_max_score
            else 0.0,
            "label": f"{total_score}/{total_max_score}",
            "per_task_max": RUBRIC_MAX_POINTS,
        },
        "results": results,
    }
    if batch_path:
        summary["evaluator_batch_dir"] = str(batch_path)
        write_json_file(batch_path / "summary.json", summary)
    return summary


HUMAN_SECTION_ORDER = [
    ("routing", "Routing"),
    ("subgraph_task", "Subgraph Task"),
    ("tools", "Tools"),
    ("artifacts", "Artifacts"),
    ("final_output", "Final Output"),
]


def section_score_label(section: dict) -> str:
    score = (section.get("details") or {}).get("score") or {}
    earned = score.get("earned")
    max_score = score.get("max")
    if earned is None or max_score is None:
        return ""
    return f"{earned}/{max_score}"


def human_section_cell(section_key: str, section: dict, *, use_color: bool) -> str:
    status = section.get("status", "")
    status_label = color_status(status, use_color=use_color)
    score_label = section_score_label(section)
    parts = [status_label]
    if score_label:
        parts.append(score_label)

    if section_key == "subgraph_task":
        judge_details = (
            (section.get("details") or {})
            .get("llm_judge", {})
            .get("details", {})
        )
        raw_score = judge_details.get("raw_score_1_to_5")
        if raw_score is not None:
            parts.append(f"LLM {raw_score}/5")

    return " ".join(parts)


def render_human_summary(summary: dict) -> str:
    use_color = human_summary_color_enabled()
    lines = ["Summary"]
    lines.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["Tasks", str(summary["n_tasks"])],
                [
                    color_status(STATUS_PASSED, "Passed", use_color=use_color),
                    str(summary["n_passed"]),
                ],
                [
                    color_status(STATUS_FAILED, "Failed", use_color=use_color),
                    str(summary["n_failed"]),
                ],
                [
                    color_status(STATUS_ERROR, "Errors", use_color=use_color),
                    str(summary["n_errors"]),
                ],
                [
                    "Score",
                    (summary.get("score") or {}).get("label", "0/0"),
                ],
            ],
        )
    )

    task_rows = []
    for item in summary["results"]:
        sections = item.get("sections", {})
        task_rows.append(
            [
                item["task_id"],
                color_status(item["status"], use_color=use_color),
                (item.get("score") or {}).get("label", "0/0"),
                *[
                    human_section_cell(
                        key,
                        sections.get(key, {}),
                        use_color=use_color,
                    )
                    for key, _label in HUMAN_SECTION_ORDER
                ],
            ]
        )

    if task_rows:
        lines.extend(["", "Tasks"])
        lines.extend(
            render_table(
                [
                    "Task",
                    "Status",
                    "Score",
                    *[label for _key, label in HUMAN_SECTION_ORDER],
                ],
                task_rows,
            )
        )
    error_lines = render_human_error_details(summary)
    if error_lines:
        lines.extend(["", *error_lines])
    return "\n".join(lines)


def render_human_error_details(summary: dict) -> list[str]:
    lines: list[str] = []
    for item in summary.get("results", []):
        if item.get("status") != STATUS_ERROR:
            continue

        task_id = item.get("task_id", "<unknown>")
        errors = item.get("errors") or []
        if not lines:
            lines.append("Errors")

        for error in errors:
            error_type = error.get("type") or "<unknown>"
            message = error.get("message") or "<no message>"
            lines.append(f"[{task_id}] {error_type}: {message}")

            traceback_text = (error.get("traceback") or "").rstrip()
            if traceback_text:
                lines.extend(f"  {line}" for line in traceback_text.splitlines())

        section_errors = render_section_error_details(task_id, item.get("sections", {}))
        if section_errors:
            lines.extend(section_errors)
        elif not errors:
            lines.append(f"[{task_id}] No exception details were captured.")

    return [line for line in lines if line != ""]


def render_section_error_details(task_id: str, sections: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section_name, section_result in sections.items():
        if not isinstance(section_result, dict):
            continue
        if section_result.get("status") != STATUS_ERROR:
            continue

        reason = section_result.get("reason")
        lines.append(f"[{task_id}] section {section_name}: {reason or 'status=error'}")
        details = section_result.get("details")
        if details:
            lines.append(
                f"  details: {json.dumps(details, default=str, ensure_ascii=False)}"
            )

    return lines


def human_summary_color_enabled() -> bool:
    return "NO_COLOR" not in os.environ and os.environ.get("TERM", "").lower() != "dumb"


def color_status(status: str, text: str | None = None, use_color: bool = True) -> str:
    label = text if text is not None else status
    if not use_color or not status:
        return label

    color = {
        STATUS_PASSED: ANSI_GREEN,
        STATUS_FAILED: ANSI_LIGHT_RED,
        STATUS_ERROR: ANSI_DARK_ORANGE,
        STATUS_SKIPPED: ANSI_YELLOW,
        STATUS_NOT_EVALUATED: ANSI_DIM,
    }.get(status)

    if not color:
        return label
    return f"{color}{label}{ANSI_RESET}"


def visible_len(value: Any) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", str(value)))


def pad_visible(value: Any, width: int) -> str:
    text = str(value)
    return text + " " * (width - visible_len(text))


def render_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    widths = [
        max(visible_len(headers[index]), *(visible_len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [
        border,
        "| "
        + " | ".join(
            pad_visible(header, widths[index]) for index, header in enumerate(headers)
        )
        + " |",
        border,
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                pad_visible(cell, widths[index]) for index, cell in enumerate(row)
            )
            + " |"
        )
    lines.append(border)
    return lines


def parse_non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiagent tasks, collect graph state, and evaluate the result."
    )
    parser.add_argument(
        "--spec-file", required=True, help="Path to the task specification JSON file."
    )
    parser.add_argument(
        "--task-id",
        action="append",
        help="Optional task id filter. Can be passed multiple times.",
    )
    parser.add_argument("--output", help="Optional path to write the JSON summary.")
    parser.add_argument(
        "--human", action="store_true", help="Print a compact human-readable summary."
    )
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help=(
            "Read-only base directory for generic relative task paths "
            "(default: BIOMED_WORKSPACE_ROOT or the repository parent)."
        ),
    )
    parser.add_argument(
        "--render-task-templates",
        action="store_true",
        help="Preprocess task spec by rendering task-local placeholders in text fields.",
    )
    parser.add_argument(
        "--task-delay-seconds",
        type=parse_non_negative_float,
        default=os.environ.get(TASK_DELAY_ENV, "0"),
        help=(
            "Optional delay between tasks, useful for rate limiting "
            f"(default: {TASK_DELAY_ENV} or 0)."
        ),
    )
    parser.add_argument(
        "--usage-limit-probe-interval-seconds",
        type=parse_non_negative_float,
        default=os.environ.get(
            USAGE_LIMIT_PROBE_INTERVAL_ENV,
            str(DEFAULT_USAGE_LIMIT_PROBE_INTERVAL_SECONDS),
        ),
        help=(
            "Seconds to wait between sample Ollama probe requests after a "
            "usage-limit response "
            f"(default: {USAGE_LIMIT_PROBE_INTERVAL_ENV} or "
            f"{DEFAULT_USAGE_LIMIT_PROBE_INTERVAL_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--usage-limit-max-wait-seconds",
        type=parse_non_negative_float,
        default=os.environ.get(
            USAGE_LIMIT_MAX_WAIT_ENV,
            str(DEFAULT_USAGE_LIMIT_MAX_WAIT_SECONDS),
        ),
        help=(
            "Maximum seconds to keep probing for Ollama usage-limit recovery "
            "before failing the current task; 0 disables the deadline "
            f"(default: {USAGE_LIMIT_MAX_WAIT_ENV} or "
            f"{DEFAULT_USAGE_LIMIT_MAX_WAIT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        help=(
            "Prefix for the timestamped evaluator run folder "
            "(default: spec file stem)."
        ),
    )
    parser.add_argument(
        "--evaluator-runs-dir",
        default=str(DEFAULT_EVALUATOR_RUNS_DIR),
        help=(
            "Base folder for timestamped evaluator run exports "
            f"(default: {DEFAULT_EVALUATOR_RUNS_DIR})."
        ),
    )
    parser.add_argument(
        "--no-evaluator-run-export",
        action="store_true",
        help="Disable the timestamped per-run evaluator export.",
    )
    parser.add_argument(
        "--preprocess-spec-output",
        help="Optional path to write the preprocessed task specification JSON.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess/print task spec and exit without running evaluation.",
    )
    parser.add_argument(
        "--graph",
        default=os.environ.get("BIOMED_GRAPH_NAME", "full_weak"),
        choices=sorted(_VALID_GRAPH_NAMES),
        help=(
            "Which compiled graph to evaluate. Values follow AgentMode exactly; "
            "for example: full_weak, pi_retries_weak, pi_single, and llm_only "
            "(default: BIOMED_GRAPH_NAME env var or 'full_weak')."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["BIOMED_GRAPH_NAME"] = args.graph

    tasks_for_eval: list[dict] | None = None
    if (
        args.render_task_templates
        or args.preprocess_only
        or args.preprocess_spec_output
    ):
        loaded_tasks = load_task_spec(args.spec_file)
        tasks_for_eval = (
            preprocess_task_spec(loaded_tasks)
            if args.render_task_templates
            else loaded_tasks
        )

        if args.preprocess_spec_output:
            Path(args.preprocess_spec_output).write_text(
                json.dumps(tasks_for_eval, indent=2),
                encoding="utf-8",
            )

        if args.preprocess_only:
            if not args.preprocess_spec_output:
                print(json.dumps(tasks_for_eval, indent=2))
            return 0

    batch_dir = None
    if not args.no_evaluator_run_export:
        batch_dir = create_evaluator_batch_dir(
            spec_file=args.spec_file,
            prefix=args.run_prefix,
            base_dir=args.evaluator_runs_dir,
        )

    summary = evaluate_tasks(
        spec_file=args.spec_file,
        task_ids=args.task_id,
        tasks=tasks_for_eval,
        workspace_root=args.workspace_root,
        task_delay_seconds=args.task_delay_seconds,
        batch_dir=batch_dir,
        usage_limit_probe_interval_seconds=args.usage_limit_probe_interval_seconds,
        usage_limit_max_wait_seconds=args.usage_limit_max_wait_seconds,
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

    if args.human:
        print(render_human_summary(summary))
    else:
        print(json.dumps(summary, indent=2, default=str))

    return 0 if summary["n_failed"] == 0 and summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
