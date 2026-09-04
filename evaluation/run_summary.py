from __future__ import annotations

import json
import math
import os
from typing import Any


SCHEMA_VERSION = "osteopilot.run_summary.v1"
DEFAULT_REDUCTION_THRESHOLD_TOKENS = 40_000
APPROX_CHARS_PER_TOKEN = 4

MAX_REDUCED_STRING_CHARS = 1800
MAX_REDUCED_MESSAGE_CHARS = 2400
MAX_REDUCED_SCRATCHPAD_CHARS = 1200
MAX_REDUCED_LIST_ITEMS = 24
MAX_REDUCED_DICT_ITEMS = 80

CONTROL_ROUTE_KEYS = {
    "__end__",
    "agent",
    "auto_attach",
    "banner",
    "debug_input1",
    "debug_input2",
    "dummy",
    "end",
    "ensure_session",
    "final_answer",
    "final_summary",
    "human_approval",
    "persist_run_trace",
    "planner",
    "post_tool",
    "react_free",
    "router",
    "todo_planner",
    "tools",
}


def estimate_json_tokens(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return int(math.ceil(len(text) / APPROX_CHARS_PER_TOKEN))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    data: dict[str, Any] = {}
    for attr in ("type", "role", "name", "content", "tool_calls"):
        if hasattr(value, attr):
            data[attr] = json_safe(getattr(value, attr))
    if data:
        return data
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return json_safe(value.dict())
        except Exception:
            pass
    return str(value)


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            return calls
        additional = message.get("additional_kwargs")
        if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
            return additional["tool_calls"]
        return []

    calls = getattr(message, "tool_calls", None)
    if isinstance(calls, list):
        return calls
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
        return additional["tool_calls"]
    return []


def normalize_messages(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    visible_raw = state.get("messages") if isinstance(state.get("messages"), list) else []
    internal_raw = (
        state.get("internal_messages")
        if isinstance(state.get("internal_messages"), list)
        else []
    )

    def is_internal_message(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        return str(message.get("visibility") or "").lower() == "internal"

    def normalize_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            typ = str(message.get("type") or message.get("role") or "message")
            content = message.get("content", "")
            name = message.get("name")
            if not name and isinstance(message.get("additional_kwargs"), dict):
                name = message["additional_kwargs"].get("name")
            if isinstance(content, list):
                content = " ".join(str(chunk) for chunk in content)
            out = {"type": typ, "content": str(content)}
            if isinstance(name, str) and name.strip():
                out["name"] = name
            tool_calls = _message_tool_calls(message)
            if tool_calls:
                out["tool_calls"] = json_safe(tool_calls)
            return out

        msg_name = getattr(message, "name", None)
        out = {
            "type": type(message).__name__,
            "content": str(getattr(message, "content", message)),
        }
        if isinstance(msg_name, str) and msg_name.strip():
            out["name"] = msg_name
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            out["tool_calls"] = json_safe(tool_calls)
        return out

    visible = [normalize_message(message) for message in visible_raw if not is_internal_message(message)]
    visible_keys = {(message["type"], message["content"]) for message in visible}
    internal_candidates = [
        *(normalize_message(message) for message in visible_raw if is_internal_message(message)),
        *(normalize_message(message) for message in internal_raw),
    ]
    internal = [
        message
        for message in internal_candidates
        if (message["type"], message["content"]) not in visible_keys
    ]
    return {"user_visible": visible, "internal": internal}


def _route_is_subgraph_candidate(route: Any) -> bool:
    if not isinstance(route, str):
        return False
    cleaned = route.strip()
    return bool(cleaned) and cleaned not in CONTROL_ROUTE_KEYS


def _timestamp_between(value: Any, start: Any, end: Any) -> bool:
    if not isinstance(value, str) or not isinstance(start, str):
        return False
    if value < start:
        return False
    return not isinstance(end, str) or value < end


def _truncate_text(value: str, limit: int = MAX_REDUCED_STRING_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}... [truncated {omitted} chars]"


def _summarize_tree_struct(value: Any) -> dict[str, Any]:
    def walk(node: Any) -> tuple[int, int]:
        if not isinstance(node, dict):
            return 0, 0
        typ = node.get("type")
        file_count = 1 if typ == "file" else 0
        dir_count = 1 if typ in {"directory", "symlink"} else 0
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                c_files, c_dirs = walk(child)
                file_count += c_files
                dir_count += c_dirs
        return file_count, dir_count

    if not isinstance(value, dict):
        return {"summary": "non-object directory tree omitted"}
    file_count, dir_count = walk(value)
    return {
        "summary": "directory tree omitted from reduced payload",
        "root_name": value.get("name"),
        "root_path": value.get("path"),
        "file_count": file_count,
        "directory_count": dir_count,
    }


def _reduce_json(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        items = [_reduce_json(item) for item in value[:MAX_REDUCED_LIST_ITEMS]]
        if len(value) > MAX_REDUCED_LIST_ITEMS:
            items.append({"omitted_items": len(value) - MAX_REDUCED_LIST_ITEMS})
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, child) in enumerate(value.items()):
            key_s = str(key)
            if key_s == "tree_struct":
                out[key_s] = _summarize_tree_struct(child)
                continue
            if idx >= MAX_REDUCED_DICT_ITEMS:
                out["omitted_keys"] = len(value) - MAX_REDUCED_DICT_ITEMS
                break
            out[key_s] = _reduce_json(child)
        return out
    return _reduce_json(json_safe(value))


def _reduce_message(
    message: dict[str, Any],
    *,
    max_chars: int = MAX_REDUCED_MESSAGE_CHARS,
) -> dict[str, Any]:
    out = dict(message)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = _truncate_text(content, max_chars)
    return out


def _reduce_messages(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = MAX_REDUCED_MESSAGE_CHARS,
) -> list[dict[str, Any]]:
    return [_reduce_message(message, max_chars=max_chars) for message in messages]


def _shape_tool_event(event: dict[str, Any], *, reduce_payload: bool) -> dict[str, Any]:
    shape_value = _reduce_json if reduce_payload else json_safe
    return {
        "id": event.get("event_id"),
        "timestamp": event.get("timestamp"),
        "scope": event.get("scope"),
        "subgraph_key": event.get("subgraph_key"),
        "step_index": event.get("step_index"),
        "tool_name": event.get("tool_name"),
        "tool_args": shape_value(event.get("tool_args") or {}),
        "success": event.get("success"),
        "error": event.get("error"),
        "output": shape_value(event.get("output")),
    }


def _sandbox_execution_from_state(
    state: dict[str, Any],
    *,
    reduce_payload: bool,
) -> dict[str, Any] | None:
    """Return structured evidence for the code_sandbox execution boundary.

    The sandbox executes generated Python directly instead of invoking graph tools,
    so normal tool_event rows are not expected for it. This block gives evaluators
    an explicit place to inspect subprocess execution and sandbox_result.json.
    """
    sandbox_result = state.get("sandbox_result")
    if not isinstance(sandbox_result, dict):
        return None

    shape_value = _reduce_json if reduce_payload else json_safe
    stdout = sandbox_result.get("stdout") or ""
    stderr = sandbox_result.get("stderr") or ""
    execution = {
        "evidence_kind": "sandbox_execution",
        "note": (
            "code_sandbox executes generated Python directly; empty tool_event_ids "
            "mean no nested graph tools were called, not that code execution was absent."
        ),
        "attempts": state.get("sandbox_attempts"),
        "exit_code": sandbox_result.get("exit_code"),
        "run_dir": sandbox_result.get("run_dir"),
        "result": shape_value(sandbox_result.get("result")),
        "stdout": _truncate_text(str(stdout), MAX_REDUCED_MESSAGE_CHARS)
        if reduce_payload
        else str(stdout),
        "stderr": _truncate_text(str(stderr), MAX_REDUCED_MESSAGE_CHARS)
        if reduce_payload
        else str(stderr),
        "sandbox_error": state.get("sandbox_error"),
        "withdrawn": state.get("sandbox_withdrawn"),
        "withdrawal_summary": state.get("sandbox_withdrawal_summary"),
    }

    code = state.get("sandbox_code")
    if isinstance(code, str) and code.strip():
        execution["code_present"] = True
        execution["code_preview"] = _truncate_text(
            code,
            MAX_REDUCED_SCRATCHPAD_CHARS if reduce_payload else 4000,
        )
    else:
        execution["code_present"] = False

    spec = state.get("sandbox_spec")
    if isinstance(spec, dict):
        execution["spec"] = shape_value(spec)

    return json_safe(execution)


def _synthetic_sandbox_tool_event(
    sandbox_execution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(sandbox_execution, dict):
        return None

    exit_code = sandbox_execution.get("exit_code")
    success = (
        exit_code == 0
        and not sandbox_execution.get("sandbox_error")
        and sandbox_execution.get("withdrawn") is not True
    )
    if exit_code is None and sandbox_execution.get("result") is None:
        success = False

    return {
        "id": "sandbox_evt_0001",
        "timestamp": None,
        "scope": "sandbox_execution",
        "subgraph_key": "code_sandbox",
        "step_index": None,
        "tool_name": "code_sandbox_execute_python",
        "tool_args": {
            "note": (
                "Synthetic evaluator evidence event. code_sandbox does not emit "
                "normal nested tool calls."
            )
        },
        "success": success,
        "error": sandbox_execution.get("sandbox_error")
        or sandbox_execution.get("withdrawal_summary"),
        "output": sandbox_execution,
    }


def _shape_messages(
    messages: list[dict[str, Any]],
    *,
    reduce_payload: bool,
    max_chars: int = MAX_REDUCED_MESSAGE_CHARS,
) -> list[dict[str, Any]]:
    if reduce_payload:
        return _reduce_messages(messages, max_chars=max_chars)
    return json_safe(messages)


def _last_assistant_content(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        typ = str(message.get("type") or "").lower()
        if "ai" in typ or "assistant" in typ:
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _collect_artifacts(
    state: dict[str, Any],
    tool_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}

    def add(value: Any, source: str) -> None:
        if isinstance(value, dict):
            path = (
                value.get("path")
                or value.get("file")
                or value.get("filename")
                or value.get("name")
            )
            if not path:
                return
            path_s = str(path)
            artifacts[path_s] = {
                "name": os.path.basename(path_s),
                "path": path_s,
                "source": source,
                **{
                    k: value.get(k)
                    for k in ("size", "size_mb", "mime", "mimetype")
                    if k in value
                },
            }
        elif isinstance(value, str) and value.strip():
            path_s = value.strip()
            artifacts[path_s] = {
                "name": os.path.basename(path_s),
                "path": path_s,
                "source": source,
            }

    for key in (
        "attachments",
        "current_run_produced_attachments",
        "sandbox_generated",
        "artifacts",
    ):
        values = state.get(key)
        if isinstance(values, list):
            for item in values:
                add(item, key)

    for event in tool_events:
        output = event.get("output")
        if not isinstance(output, dict):
            continue
        attachments = output.get("attachments")
        if isinstance(attachments, list):
            for item in attachments:
                add(item, f"tool:{event.get('tool_name') or 'unknown'}")

    return sorted(
        artifacts.values(),
        key=lambda item: item.get("path") or item.get("name") or "",
    )


def _extract_evaluator_info(state: dict[str, Any], run_trace: dict[str, Any]) -> dict[str, Any] | None:
    sources: list[Any] = [
        run_trace.get("evaluator") if isinstance(run_trace, dict) else None,
        state.get("evaluator") if isinstance(state, dict) else None,
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue
        out = {
            "source": source.get("source"),
            "task_id": source.get("task_id"),
            "task_name": source.get("task_name"),
        }
        cleaned = {
            key: value
            for key, value in out.items()
            if isinstance(value, str) and value.strip()
        }
        if cleaned.get("task_id"):
            return cleaned

    task_id = state.get("evaluator_task_id") if isinstance(state, dict) else None
    task_name = state.get("evaluator_task_name") if isinstance(state, dict) else None
    source = state.get("evaluator_source") if isinstance(state, dict) else None
    if isinstance(task_id, str) and task_id.strip():
        out = {"task_id": task_id.strip()}
        if isinstance(task_name, str) and task_name.strip():
            out["task_name"] = task_name.strip()
        if isinstance(source, str) and source.strip():
            out["source"] = source.strip()
        return out

    return None


def _extract_session_info(state: dict[str, Any], run_trace: dict[str, Any]) -> dict[str, Any] | None:
    session = run_trace.get("session") if isinstance(run_trace, dict) else None
    if not isinstance(session, dict):
        session = state.get("session") if isinstance(state, dict) else None
    if not isinstance(session, dict):
        return None

    out = {
        "id": session.get("id"),
        "path": session.get("path"),
    }
    cleaned = {
        key: value
        for key, value in out.items()
        if isinstance(value, str) and value.strip()
    }
    return cleaned or None


def _build_summary_once(
    *,
    run_id: str | None,
    task: str | None,
    final_state: dict[str, Any],
    subgraph_snapshots: list[dict[str, Any]] | None,
    raw_endpoint: str | None,
    step_ids: list[Any] | None,
    reduce_payload: bool,
) -> dict[str, Any]:
    state = json_safe(final_state)
    run_trace = state.get("run_trace") if isinstance(state.get("run_trace"), dict) else {}
    graph_meta = run_trace.get("graph") if isinstance(run_trace.get("graph"), dict) else {}
    raw_routes = (
        run_trace.get("route_trace")
        if isinstance(run_trace.get("route_trace"), list)
        else state.get("route_history")
        if isinstance(state.get("route_history"), list)
        else []
    )
    raw_tools = (
        run_trace.get("tool_events")
        if isinstance(run_trace.get("tool_events"), list)
        else []
    )

    messages = normalize_messages(state)
    user_visible = _shape_messages(
        messages.get("user_visible", []),
        reduce_payload=reduce_payload,
        max_chars=5000,
    )
    internal = _shape_messages(
        messages.get("internal", []),
        reduce_payload=reduce_payload,
        max_chars=MAX_REDUCED_MESSAGE_CHARS,
    )
    tool_events = [
        _shape_tool_event(event, reduce_payload=reduce_payload)
        for event in raw_tools
        if isinstance(event, dict)
    ]
    sandbox_execution = _sandbox_execution_from_state(
        state,
        reduce_payload=reduce_payload,
    )
    sandbox_event = _synthetic_sandbox_tool_event(sandbox_execution)
    if sandbox_event is not None:
        tool_events.append(sandbox_event)
    tool_by_id = {
        str(tool.get("id")): tool
        for tool in tool_events
        if tool.get("id") is not None
    }

    routes: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_routes, start=1):
        if not isinstance(entry, dict):
            continue
        routes.append(
            {
                "id": f"route_{index:03d}",
                "index": index,
                "route": entry.get("route"),
                "task": entry.get("subgraph_task"),
                "timestamp": entry.get("timestamp"),
                "subgraph_calls": {
                    "used": entry.get("subgraph_calls_used"),
                    "remaining": entry.get("subgraph_calls_remaining"),
                },
            }
        )

    snapshots_by_key: dict[str, list[dict[str, Any]]] = {}
    for snapshot in subgraph_snapshots or []:
        key = snapshot.get("subgraph_key") or snapshot.get("key")
        if isinstance(key, str) and key.strip():
            snapshots_by_key.setdefault(key.strip(), []).append(snapshot)

    subgraphs: list[dict[str, Any]] = []
    occurrence_by_key: dict[str, int] = {}
    for route in routes:
        key = route.get("route")
        if not _route_is_subgraph_candidate(key):
            continue
        key_s = str(key)
        occurrence = occurrence_by_key.get(key_s, 0) + 1
        occurrence_by_key[key_s] = occurrence
        next_route = next(
            (
                later
                for later in routes
                if later["index"] > route["index"] and later.get("timestamp")
            ),
            None,
        )
        tool_ids = [
            str(tool.get("id"))
            for tool in tool_events
            if tool.get("id")
            and tool.get("subgraph_key") == key_s
            and (
                not route.get("timestamp")
                or _timestamp_between(
                    tool.get("timestamp"),
                    route.get("timestamp"),
                    next_route.get("timestamp") if next_route else None,
                )
            )
        ]
        if not tool_ids:
            tool_ids = [
                str(tool.get("id"))
                for tool in tool_events
                if tool.get("id") and tool.get("subgraph_key") == key_s
            ]

        matching_snapshots = snapshots_by_key.get(key_s, [])
        scratchpad = []
        if occurrence <= len(matching_snapshots):
            scratchpad = matching_snapshots[occurrence - 1].get("scratchpad") or []
        elif matching_snapshots:
            scratchpad = matching_snapshots[-1].get("scratchpad") or []
        scratchpad = _shape_messages(
            scratchpad,
            reduce_payload=reduce_payload,
            max_chars=MAX_REDUCED_SCRATCHPAD_CHARS,
        )

        subgraph_tools = [tool_by_id[tool_id] for tool_id in tool_ids if tool_id in tool_by_id]
        subgraph_record = {
            "id": f"subgraph_{len(subgraphs) + 1:03d}",
            "key": key_s,
            "call_index": occurrence,
            "route_id": route["id"],
            "task": route.get("task"),
            "started_at": route.get("timestamp"),
            "tool_event_ids": tool_ids,
            "tool_names": sorted(
                {
                    tool.get("tool_name")
                    for tool in subgraph_tools
                    if tool.get("tool_name")
                }
            ),
            "status": "failed"
            if any(tool.get("success") is False for tool in subgraph_tools)
            else "completed",
            "scratchpad": scratchpad,
        }
        if key_s == "code_sandbox" and sandbox_execution is not None:
            if sandbox_event is not None and sandbox_event["id"] not in tool_ids:
                subgraph_record["tool_event_ids"] = [*tool_ids, sandbox_event["id"]]
                subgraph_record["tool_names"] = sorted(
                    {
                        *subgraph_record["tool_names"],
                        sandbox_event["tool_name"],
                    }
                )
            subgraph_record["sandbox_execution"] = sandbox_execution
            if sandbox_event is not None and sandbox_event.get("success") is False:
                subgraph_record["status"] = "failed"
        subgraphs.append(subgraph_record)

    final_error = state.get("error") if isinstance(state, dict) else None
    has_tool_errors = any(tool.get("success") is False for tool in tool_events)
    status = "error" if final_error else ("completed_with_tool_errors" if has_tool_errors else "completed")
    effective_run_id = str(run_id or run_trace.get("run_id") or "run_unknown")
    started_at = graph_meta.get("started_at")
    ended_at = graph_meta.get("ended_at")
    evaluator = _extract_evaluator_info(state, run_trace)

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "id": effective_run_id,
            "task": task,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "session": _extract_session_info(state, run_trace),
            "evaluator": evaluator,
        },
        "conversation": {
            "user_visible": user_visible,
            "internal": internal,
        },
        "routes": routes,
        "subgraphs": subgraphs,
        "tools": tool_events,
        "artifacts": _collect_artifacts(state, raw_tools),
        "final": {
            "answer": _last_assistant_content(user_visible),
            "error": final_error,
        },
        "debug": {
            "raw_steps_available": bool(raw_endpoint),
            "raw_endpoint": raw_endpoint,
            "step_ids": step_ids or [],
            "final_state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
        },
    }


def build_run_summary_from_state(
    *,
    final_state: dict[str, Any],
    run_id: str | None = None,
    task: str | None = None,
    subgraph_snapshots: list[dict[str, Any]] | None = None,
    raw_endpoint: str | None = None,
    step_ids: list[Any] | None = None,
    reduction_threshold_tokens: int = DEFAULT_REDUCTION_THRESHOLD_TOKENS,
) -> dict[str, Any]:
    full = _build_summary_once(
        run_id=run_id,
        task=task,
        final_state=final_state,
        subgraph_snapshots=subgraph_snapshots,
        raw_endpoint=raw_endpoint,
        step_ids=step_ids,
        reduce_payload=False,
    )
    tokens_before = estimate_json_tokens(full)
    if tokens_before <= reduction_threshold_tokens:
        full["debug"]["payload_reduction"] = {
            "enabled": False,
            "threshold_tokens": reduction_threshold_tokens,
            "estimated_tokens": tokens_before,
        }
        return full

    reduced = _build_summary_once(
        run_id=run_id,
        task=task,
        final_state=final_state,
        subgraph_snapshots=subgraph_snapshots,
        raw_endpoint=raw_endpoint,
        step_ids=step_ids,
        reduce_payload=True,
    )
    reduced["debug"]["payload_reduction"] = {
        "enabled": True,
        "threshold_tokens": reduction_threshold_tokens,
        "estimated_tokens_before": tokens_before,
        "estimated_tokens_after": estimate_json_tokens(reduced),
    }
    return reduced
