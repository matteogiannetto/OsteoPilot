from __future__ import annotations

import json
import os
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .time_utils import local_now_iso

BIOMED_AGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_TRACE_BASE_DIR = str(BIOMED_AGENT_ROOT / "evaluation_runs")


def _now_ts() -> str:
    return local_now_iso()


def _now_unix() -> float:
    return time.time()


def _safe_test_name(value: Any) -> str:
    text = str(value or "").strip() or "_ungrouped"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120] or "_ungrouped"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "content"):
        try:
            return _json_safe(getattr(value, "content"))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _extract_session_info(state: dict[str, Any]) -> dict[str, str | None]:
    session = state.get("session")
    if isinstance(session, dict):
        return {
            "id": session.get("id") if isinstance(session.get("id"), str) else None,
            "path": session.get("path") if isinstance(session.get("path"), str) else None,
        }
    return {"id": None, "path": None}


def _extract_evaluator_info(state: dict[str, Any]) -> dict[str, str] | None:
    evaluator = state.get("evaluator")
    if isinstance(evaluator, dict):
        out = {
            "source": evaluator.get("source"),
            "task_id": evaluator.get("task_id"),
            "task_name": evaluator.get("task_name"),
        }
    else:
        out = {
            "source": state.get("evaluator_source"),
            "task_id": state.get("evaluator_task_id"),
            "task_name": state.get("evaluator_task_name"),
        }

    cleaned = {
        key: value
        for key, value in out.items()
        if isinstance(value, str) and value.strip()
    }
    return cleaned or None


def ensure_execution_trace(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("execution_trace")
    if isinstance(existing, dict) and existing.get("run_id"):
        rt = deepcopy(existing)
        rt["test_name"] = state.get("test_name", rt.get("test_name"))
        rt["session"] = _extract_session_info(state)
        evaluator = _extract_evaluator_info(state)
        if evaluator:
            rt["evaluator"] = evaluator
        rt.setdefault("graph", {})
        rt["graph"].setdefault("started_at", _now_ts())
        rt["graph"].setdefault("started_at_unix", _now_unix())
        rt["graph"].setdefault("ended_at", None)
        rt["graph"].setdefault("ended_at_unix", None)
        rt.setdefault("route_trace", [])
        rt.setdefault("tool_events", [])
        return rt

    return {
        "run_id": f"run-{uuid.uuid4().hex}",
        "test_name": state.get("test_name"),
        "evaluator": _extract_evaluator_info(state),
        "session": _extract_session_info(state),
        "graph": {
            "started_at": _now_ts(),
            "started_at_unix": _now_unix(),
            "ended_at": None,
            "ended_at_unix": None,
        },
        "route_trace": [],
        "tool_events": [],
    }


def make_tool_event(
    *,
    scope: str,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    success: bool,
    output: Any = None,
    error: str | None = None,
    subgraph_key: str | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    return {
        "event_id": None,
        "timestamp": _now_ts(),
        "timestamp_unix": _now_unix(),
        "scope": scope,
        "subgraph_key": subgraph_key,
        "step_index": step_index,
        "tool_name": str(tool_name),
        "tool_args": _json_safe(tool_args or {}),
        "success": bool(success),
        "output": _json_safe(output),
        "error": error,
    }


def _assign_event_id(
    existing_events: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    safe_event = _json_safe(event)
    if not isinstance(safe_event, dict):
        raise TypeError("A tool event must remain a mapping after JSON normalization.")
    out: dict[str, Any] = deepcopy(safe_event)
    if out.get("event_id"):
        return out
    out["event_id"] = f"evt_{len(existing_events) + 1:04d}"
    return out


def append_tool_event(
    state: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    rt = ensure_execution_trace(state)
    events = list(rt.get("tool_events") or [])
    events.append(_assign_event_id(events, event))
    rt["tool_events"] = events
    return rt


def finalize_execution_trace(state: dict[str, Any]) -> dict[str, Any]:
    rt = ensure_execution_trace(state)
    out = deepcopy(rt)
    out["test_name"] = state.get("test_name", out.get("test_name"))
    evaluator = _extract_evaluator_info(state)
    if evaluator:
        out["evaluator"] = evaluator
    out["session"] = _extract_session_info(state)
    out.setdefault("graph", {})
    out["graph"]["ended_at"] = _now_ts()
    out["graph"]["ended_at_unix"] = _now_unix()

    route_history = state.get("route_history") or []
    if not isinstance(route_history, list):
        route_history = []
    out["route_trace"] = _json_safe(route_history)
    out["tool_events"] = [_json_safe(evt) for evt in list(out.get("tool_events") or [])]
    return out


def persist_execution_trace(
    state: dict[str, Any],
    *,
    default_base_dir: str = DEFAULT_EXECUTION_TRACE_BASE_DIR,
) -> dict[str, Any]:
    rt = finalize_execution_trace(state)
    test_name = _safe_test_name(rt.get("test_name") or state.get("test_name"))
    session = rt.get("session") or {}
    session_id = str(session.get("id") or "no_session")
    base_dir = str(state.get("execution_trace_base_dir") or default_base_dir)
    test_dir = os.path.join(base_dir, test_name)
    session_dir = os.path.join(test_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    run_id = str(rt.get("run_id") or f"run-{uuid.uuid4().hex}")
    run_file = os.path.join(session_dir, f"{run_id}.json")
    index_file = os.path.join(test_dir, "index.jsonl")

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(_json_safe(rt), f, indent=2, ensure_ascii=False)

    tool_events = rt.get("tool_events") or []
    successful_tool_calls = sum(1 for evt in tool_events if isinstance(evt, dict) and evt.get("success") is True)
    failed_tool_calls = sum(1 for evt in tool_events if isinstance(evt, dict) and evt.get("success") is False)

    index_record = {
        "run_id": run_id,
        "test_name": test_name,
        "session_id": session.get("id"),
        "session_path": session.get("path"),
        "started_at": rt.get("graph", {}).get("started_at"),
        "started_at_unix": rt.get("graph", {}).get("started_at_unix"),
        "ended_at": rt.get("graph", {}).get("ended_at"),
        "ended_at_unix": rt.get("graph", {}).get("ended_at_unix"),
        "tool_call_count": len(tool_events),
        "successful_tool_calls": successful_tool_calls,
        "failed_tool_calls": failed_tool_calls,
        "routes": [
            entry.get("route")
            for entry in (rt.get("route_trace") or [])
            if isinstance(entry, dict)
        ],
        "run_file": run_file,
    }
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(index_record), ensure_ascii=False) + "\n")

    return {
        "execution_trace": rt,
        "execution_trace_file": run_file,
        "execution_trace_index_file": index_file,
        "run_file": run_file,
        "index_file": index_file,
    }
