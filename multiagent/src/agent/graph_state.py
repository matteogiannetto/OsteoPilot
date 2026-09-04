from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from .tools.session_management import AttachmentInfo, SessionInfo

DEFAULT_DEBUG = True
DEFAULT_SUBGRAPH_CALL_LIMIT = 7
DEFAULT_MAIN_TOOL_CALL_LIMIT = 8

TodoStatus = Literal["pending", "in_progress", "done", "blocked"]

class TodoItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    content: str = Field(validation_alias=AliasChoices("content", "text"))
    status: TodoStatus = "pending"
    route: str | None = None
    subgraph_task: str | None = None
    depends_on: list[str] = Field(default_factory=list)

class TodoPlan(BaseModel):
    todos: list[TodoItem]


DEFAULT_DEBUG = False  # already in your file

def attachment_reducer(
    left: list[AttachmentInfo] | None,
    right: list[AttachmentInfo] | None,
) -> list[AttachmentInfo] | None:
    """
    Merge attachments without duplicating entries.

    Identity priority:
    1) 'id' (preferred, stable across moves)
    2) 'origin_path' (stable for drop_off sources)
    3) 'path' (fallback)

    Logging is compact and controlled by DEFAULT_DEBUG.
    """
    if left is None:
        if DEFAULT_DEBUG:
            print("[attachments] left=None -> using right "
                  f"(right_len={len(right) if right is not None else 0})",
                  flush=True)
        return right

    if right is None:
        if DEFAULT_DEBUG:
            print("[attachments] right=None -> using left "
                  f"(left_len={len(left) if left is not None else 0})",
                  flush=True)
        return left

    def _key(item: object) -> str | None:
        if not isinstance(item, dict):
            return None
        v = item.get("id")
        if isinstance(v, str) and v:
            return f"id:{v}"
        v = item.get("origin_path")
        if isinstance(v, str) and v:
            return f"origin:{v}"
        v = item.get("path")
        if isinstance(v, str) and v:
            return f"path:{v}"
        return None

    merged: dict[str, AttachmentInfo] = {}
    kept_unkeyed: list[AttachmentInfo] = []

    # seed from left (preserve order)
    for item in left:
        k = _key(item)
        if k is None:
            if isinstance(item, dict):
                kept_unkeyed.append(item)
            continue
        merged[k] = item

    added = 0
    for item in right:
        k = _key(item)
        if k is None:
            if isinstance(item, dict):
                kept_unkeyed.append(item)
            continue
        if k not in merged:
            added += 1
        merged[k] = item

    result = list(merged.values()) + kept_unkeyed

    if DEFAULT_DEBUG:
        print(
            "[attachments] merge "
            f"left={len(left)}, right={len(right)}, "
            f"result={len(result)}, added={added}",
            flush=True,
        )

    return result

def session_reducer(
    left: SessionInfo | None,
    right: object,
) -> SessionInfo | None:
    """Reducer function to update the session with DEBUG printing."""
    
    
    if isinstance(right, dict) and "id" in right:
        #print("DECISION: 'right' is a valid SessionInfo dict. Using 'right'.", flush=True)
        #print("="*80 + "\n", flush=True)
        return cast(SessionInfo, right)
        
    #print("DECISION: 'right' is NOT valid. Keeping 'left'.", flush=True)
    #print("="*80 + "\n", flush=True)
    return left

class BiomedAgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    internal_messages: Annotated[list[BaseMessage], add_messages]

    todos: list[TodoItem]
    current_plan_hash: str
    plan_approved: bool

    session: Annotated[SessionInfo, session_reducer]
    attachments: Annotated[list[AttachmentInfo], attachment_reducer]

    route_decision: str
    session_banner: str
    route_history: list[dict[str, Any]]   # {"route": ..., "subgraph_task": ..., "timestamp": ...}
    subgraph_call_limit: int
    subgraph_calls_made: int

    main_tool_call_limit: int
    main_tool_calls_made: int

    router_subgraph_task: str
    router_reason: str

    test_name: str
    evaluator: dict[str, Any]
    evaluator_task_id: str
    evaluator_task_name: str
    workspace_root: str

    execution_trace: dict[str, Any]
    execution_trace_file: str
    execution_trace_index_file: str

    
class subAgentState(BiomedAgentState, total=False):
    private_scratchpad: Annotated[list[BaseMessage], add_messages]

    plan: SubgraphTaskPlan | dict[str, Any]
    imaging_phase: str
    imaging_agent_step_count: int
    imaging_agent_continue: bool
    imaging_agent_consecutive_failed_tool_batches: int

    agent_notes: str | None

    current_run_produced_attachments: Annotated[
        list[AttachmentInfo],
        attachment_reducer,
    ]

    
class SandboxStateExtension(TypedDict, total=False):
    private_scratchpad: Annotated[list[BaseMessage], add_messages]
    # High-level spec the planner creates for the sandbox
    sandbox_spec: dict[str, Any]                # e.g. {"task_type": "csv_preview", ...}
    sandbox_code: str
    sandbox_result: dict[str, Any]
    sandbox_error: str | None
    sandbox_attempts: int
    sandbox_run_dir: str
    sandbox_repair_action: str
    sandbox_repair_reason: str
    sandbox_repair_message: str
    sandbox_repair_context: str
    sandbox_failure_history: list[dict[str, Any]]
    sandbox_withdrawn: bool
    sandbox_withdrawal_summary: str
    # Files generated by the sandbox (PNGs, TIFFs, CSVs, etc.).
    # This uses the same reducer as attachments to merge without duplicates.
    sandbox_generated: Annotated[list[AttachmentInfo], attachment_reducer]


class subAgentExtendedState(subAgentState):
    pass
    
class SandboxExtendedState(BiomedAgentState, SandboxStateExtension):
    pass
    
    
############################### this part is used fro planning in the palnner node of the subgraphs #######

TaskStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported_for_this_subgraph",
]

class SubgraphTaskPlan(BaseModel):
    """
    Shared structured planner output for all subgraphs.

    This object is produced by a planner node and consumed by the executor
    and final summarization logic.
    """

    task_status: TaskStatus = Field(
        description=(
            "Whether the routed subtask is fully supported, partially supported, "
            "or unsupported for the current subgraph."
        )
    )
    task_summary: str = Field(
        min_length=1,
        description="Short summary of the routed subtask and relevant files."
    )
    steps: list[str] = Field(
        default_factory=list,
        description="Optional high-level execution steps for the executor."
    )
    primary_inputs: list[str] = Field(
        default_factory=list,
        description="Relevant input file paths for this subtask."
    )
    planner_notes: str = Field(
        default="",
        description="Optional short technical notes or strategy."
    )

######################################

################# part for agent step in subgraphs ########################

class NormalizedToolCall(BaseModel):
    tool_call_id: str | None = Field(
        default=None,
        description="Tool call id from the model output, if present."
    )
    tool_name: str = Field(
        min_length=1,
        description="Resolved tool name."
    )
    args_dict: dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized tool arguments as a dictionary."
    )
   

class StateViews:
    """
    Formatting/view utilities for LangGraph state.

    Held by the agent class:
        self.views = StateViews()
    """

    NON_DATA_ATTACHMENT_VIEW_LIMIT = 15

    @staticmethod
    def _is_under_path(candidate: str, root: str) -> bool:
        try:
            candidate_abs = os.path.abspath(os.fspath(Path(candidate).expanduser()))
            root_abs = os.path.abspath(os.fspath(Path(root).expanduser()))
            if os.path.commonpath([root_abs, candidate_abs]) == root_abs:
                return True
        except Exception:
            pass

        try:
            candidate_real = os.path.realpath(os.fspath(Path(candidate).expanduser()))
            root_real = os.path.realpath(os.fspath(Path(root).expanduser()))
            return os.path.commonpath([root_real, candidate_real]) == root_real
        except Exception:
            return False

    @classmethod
    def _is_data_attachment(
        cls,
        att: Any,
        *,
        session: Any = None,
        workspace_root: Any = None,
    ) -> bool:
        if not isinstance(att, dict):
            return False

        raw_paths = [
            value
            for value in (att.get("path"), att.get("origin_path"))
            if isinstance(value, str) and value.strip()
        ]
        for raw_path in raw_paths:
            path = raw_path.strip()
            if path == "data" or path.startswith("data/"):
                return True

            if isinstance(session, dict):
                session_path = session.get("path")
                if isinstance(session_path, str) and session_path.strip():
                    data_link = os.path.join(session_path, "data")
                    if cls._is_under_path(path, data_link):
                        return True

            if isinstance(workspace_root, str) and workspace_root.strip():
                workspace_data = os.path.join(workspace_root, "data")
                if cls._is_under_path(path, workspace_data):
                    return True

        return False

    def visible_attachments(
        self,
        attachments: Any,
        *,
        session: Any = None,
        workspace_root: Any = None,
    ) -> tuple[list[Any], int]:
        if not isinstance(attachments, list) or not attachments:
            return [], 0

        data_indexes: set[int] = set()
        non_data_indexes: list[int] = []
        for index, att in enumerate(attachments):
            if self._is_data_attachment(att, session=session, workspace_root=workspace_root):
                data_indexes.add(index)
            else:
                non_data_indexes.append(index)

        visible_non_data = set(non_data_indexes[-self.NON_DATA_ATTACHMENT_VIEW_LIMIT :])
        visible_indexes = data_indexes | visible_non_data
        hidden_non_data = max(len(non_data_indexes) - len(visible_non_data), 0)
        return [att for index, att in enumerate(attachments) if index in visible_indexes], hidden_non_data

    def _format_attachment(self, att: Any) -> str:
        if not isinstance(att, dict):
            return f"- <invalid attachment: {type(att).__name__}>"

        filename = att.get("filename") or "unknown_file"
        path = att.get("path") or "unknown_path"
        origin_path = att.get("origin_path")

        # Optional identity/provenance fields
        att_id = att.get("id")
        kind = att.get("kind")
        origin = att.get("origin")
        mimetype = att.get("mimetype")

        # Size
        size_raw = att.get("size_mb")
        if size_raw is None:
            size_str = "N/A"
        else:
            try:
                size_str = f"{float(size_raw):.3f} MB"
            except Exception:
                size_str = "N/A"

        parts = [f"- {filename}"]

        if isinstance(att_id, str) and att_id:
            parts.append(f"id={att_id[:8]}…")
        if isinstance(kind, str) and kind:
            parts.append(f"kind={kind}")
        if isinstance(origin, str) and origin:
            parts.append(f"origin={origin}")
        if isinstance(mimetype, str) and mimetype:
            parts.append(f"mime={mimetype}")
        parts.append(f"path={path}")

        if isinstance(origin_path, str) and origin_path and origin_path != path:
            parts.append(f"origin_path={origin_path}")

        parts.append(f"size={size_str}")

        description = att.get("description")
        if isinstance(description, str) and description.strip():
            compact_description = " ".join(description.split())
            parts.append(f"description={compact_description}")

        return " | ".join(parts)

    def router_view(self, state: Any) -> str:
        st = state if isinstance(state, dict) else {}

        session = st.get("session")
        workspace_root = st.get("workspace_root")
        attachments = st.get("attachments")
        todos = st.get("todos")
        plan_hash = st.get("current_plan_hash")
        subgraph_call_limit = st.get("subgraph_call_limit", DEFAULT_SUBGRAPH_CALL_LIMIT)
        subgraph_calls_made = st.get("subgraph_calls_made", 0)

        try:
            subgraph_call_limit = max(int(subgraph_call_limit), 0)
        except Exception:
            subgraph_call_limit = DEFAULT_SUBGRAPH_CALL_LIMIT

        try:
            subgraph_calls_made = max(int(subgraph_calls_made), 0)
        except Exception:
            subgraph_calls_made = 0

        subgraph_calls_remaining = max(subgraph_call_limit - subgraph_calls_made, 0)

        lines: list[str] = []
        lines.append("CURRENT CONTEXT SNAPSHOT (authoritative):")

        # Session
        if isinstance(session, dict) and session.get("id"):
            sid = session.get("id")
            spath = session.get("path")
            lines.append(f"- SESSION: id={sid}" + (f" | path={spath}" if spath else ""))
        else:
            lines.append("- SESSION: <missing>")

        if isinstance(workspace_root, str) and workspace_root:
            lines.append(f"- WORKSPACE ROOT: {workspace_root}")

        lines.append(
            "- SUBGRAPH BUDGET: "
            f"used={subgraph_calls_made}/{subgraph_call_limit} | remaining={subgraph_calls_remaining}"
        )

        # Attachments
        lines.append("- ATTACHMENTS:")
        if isinstance(attachments, list) and attachments:
            visible_attachments, hidden_non_data = self.visible_attachments(
                attachments,
                session=session,
                workspace_root=workspace_root,
            )
            for att in visible_attachments:
                lines.append(self._format_attachment(att))
            if hidden_non_data:
                lines.append(f"- Hidden non-data attachments: {hidden_non_data}")
        else:
            lines.append("- (none)")

        # Todos
        lines.append(f"- TODO PLAN (current_plan_hash={plan_hash if plan_hash is not None else 'null'}):")
        if isinstance(todos, list) and todos:
            for t in todos:
                if isinstance(t, dict):
                    tid = t.get("id", "?")
                    status = t.get("status", "?")
                    route = t.get("route")
                    content = t.get("content", "")
                else:
                    tid = getattr(t, "id", "?")
                    status = getattr(t, "status", "?")
                    route = getattr(t, "route", None)
                    content = getattr(t, "content", "")

                content = (str(content) if content is not None else "").strip()
                if len(content) > 140:
                    content = content[:140] + "…"

                if route:
                    lines.append(f"- {tid} [{status}] ({route}): {content}")
                else:
                    lines.append(f"- {tid} [{status}]: {content}")
        else:
            lines.append("- (none)")

        withdrawal = st.get("sandbox_withdrawal_summary")
        if isinstance(withdrawal, str) and withdrawal.strip():
            lines.append("- LAST SANDBOX WITHDRAWAL:")
            lines.append(f"- {withdrawal.strip()}")

        return "\n".join(lines)

    def route_history_view(self, state: Any, limit: int = 15) -> str:
        st = state if isinstance(state, dict) else {}
        rh = st.get("route_history")

        if not isinstance(rh, list) or not rh:
            return "(no route history yet)"

        tail = rh[-max(int(limit), 1):]
        lines: list[str] = []

        for i, entry in enumerate(tail, 1):
            if not isinstance(entry, dict):
                lines.append(f"{i}) <invalid entry: {type(entry).__name__}>")
                continue

            r = entry.get("route", "unknown")
            task = entry.get("subgraph_task")
            outcome = entry.get("outcome")
            ts = entry.get("timestamp")

            s = f"{i}) {r}"
            if isinstance(task, str) and task.strip():
                s += f" — task: {task.strip()}"
            if isinstance(outcome, str) and outcome.strip():
                s += f" — outcome: {outcome.strip()}"
            if ts is not None:
                s += f" — t={ts}"
            lines.append(s)

        return "\n".join(lines)
    
    def llm_state_view(self, state: Any) -> str:
        """
        Build a full, detailed, LLM-friendly view of the current state.

        Goals:
        - Be comprehensive about *structured state* (session, attachments, todos, results, specs, routing).
        - Avoid dumping entire message histories (too big/noisy); provide counts + last messages preview.
        - Never crash; always return a string.
        """
        try:
            st = state if isinstance(state, dict) else {}

            lines: list[str] = []
            lines.append("STATE VIEW (structured, authoritative)")
            lines.append("")

            # -------------------------
            # Session
            # -------------------------
            session = st.get("session")
            lines.append("## Session")
            if isinstance(session, dict):
                sid = session.get("id")
                spath = session.get("path")
                lines.append(f"- id: {sid if sid is not None else '(missing)'}")
                lines.append(f"- path: {spath if spath is not None else '(missing)'}")
                # include any other session keys
                extra = {k: v for k, v in session.items() if k not in ("id", "path")}
                if extra:
                    lines.append("- extra:")
                    lines.append("```json")
                    lines.append(json.dumps(extra, indent=2, default=str))
                    lines.append("```")
            else:
                lines.append("- (missing or invalid)")
            lines.append("")

            workspace_root = st.get("workspace_root")
            if isinstance(workspace_root, str) and workspace_root:
                lines.append("## Workspace Root")
                lines.append(f"- path: {workspace_root}")
                lines.append("")

            # -------------------------
            # Messages (do not dump full)
            # -------------------------
            lines.append("## Messages (high-level)")
            msgs = st.get("messages")
            internal = st.get("internal_messages")
            msg_count = len(msgs) if isinstance(msgs, list) else 0
            internal_count = len(internal) if isinstance(internal, list) else 0
            lines.append(f"- messages: {msg_count}")
            lines.append(f"- internal_messages: {internal_count}")

            # last user-visible message preview
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                last_type = type(last).__name__
                last_content = getattr(last, "content", None)
                if not isinstance(last_content, str):
                    last_content = str(last_content) if last_content is not None else ""
                last_content = last_content.strip()
                if len(last_content) > 800:
                    last_content = last_content[:800] + "…"
                lines.append(f"- last message type: {last_type}")
                lines.append(f"- last message preview: {last_content!r}")
            else:
                lines.append("- last message: (none)")

            # last internal message preview (often tool output)
            if isinstance(internal, list) and internal:
                last_i = internal[-1]
                last_i_type = type(last_i).__name__
                last_i_content = getattr(last_i, "content", None)
                if not isinstance(last_i_content, str):
                    last_i_content = str(last_i_content) if last_i_content is not None else ""
                last_i_content = last_i_content.strip()
                if len(last_i_content) > 800:
                    last_i_content = last_i_content[:800] + "…"
                lines.append(f"- last internal message type: {last_i_type}")
                lines.append(f"- last internal preview: {last_i_content!r}")
            else:
                lines.append("- last internal message: (none)")
            lines.append("")

            # -------------------------
            # Attachments
            # -------------------------
            lines.append("## Attachments")
            attachments = st.get("attachments")
            if isinstance(attachments, list) and attachments:
                lines.append(f"- count: {len(attachments)}")
                visible_attachments, hidden_non_data = self.visible_attachments(
                    attachments,
                    session=session,
                    workspace_root=workspace_root,
                )
                lines.append(f"- visible_count: {len(visible_attachments)}")
                if hidden_non_data:
                    lines.append(f"- hidden_non_data_count: {hidden_non_data}")
                for att in visible_attachments:
                    lines.append(self._format_attachment(att))
            else:
                lines.append("- (none)")
            lines.append("")

            # -------------------------
            # Todos / Plan
            # -------------------------
            lines.append("## Todo plan")
            plan_hash = st.get("current_plan_hash", None)
            lines.append(f"- current_plan_hash: {plan_hash if plan_hash is not None else 'null'}")

            todos = st.get("todos")
            if isinstance(todos, list) and todos:
                lines.append(f"- count: {len(todos)}")
                # show compact list (then raw)
                for t in todos:
                    if isinstance(t, dict):
                        tid = t.get("id", "?")
                        status = t.get("status", "?")
                        route = t.get("route", None)
                        content = t.get("content", "")
                        depends = t.get("depends_on", []) or []
                        subtask = t.get("subgraph_task", None)
                    else:
                        tid = getattr(t, "id", "?")
                        status = getattr(t, "status", "?")
                        route = getattr(t, "route", None)
                        content = getattr(t, "content", "")
                        depends = getattr(t, "depends_on", []) or []
                        subtask = getattr(t, "subgraph_task", None)

                    content = (str(content) if content is not None else "").strip()
                    if len(content) > 180:
                        content = content[:180] + "…"

                    s = f"- {tid} [{status}]"
                    if route:
                        s += f" ({route})"
                    s += f": {content}"

                    if depends:
                        s += f" | depends_on={depends}"
                    if isinstance(subtask, str) and subtask.strip():
                        subtask_s = subtask.strip()
                        if len(subtask_s) > 180:
                            subtask_s = subtask_s[:180] + "…"
                        s += f" | subgraph_task={subtask_s}"

                    lines.append(s)
            else:
                lines.append("- (none)")
            lines.append("")

            # -------------------------
            # Routing state
            # -------------------------
            lines.append("## Routing")
            route_decision = st.get("route_decision", None)
            router_reason = st.get("router_reason", None)
            router_subgraph_task = st.get("router_subgraph_task", None)
            subgraph_call_limit = st.get("subgraph_call_limit", DEFAULT_SUBGRAPH_CALL_LIMIT)
            subgraph_calls_made = st.get("subgraph_calls_made", 0)
            try:
                subgraph_call_limit = max(int(subgraph_call_limit), 0)
            except Exception:
                subgraph_call_limit = DEFAULT_SUBGRAPH_CALL_LIMIT
            try:
                subgraph_calls_made = max(int(subgraph_calls_made), 0)
            except Exception:
                subgraph_calls_made = 0
            subgraph_calls_remaining = max(subgraph_call_limit - subgraph_calls_made, 0)
            lines.append(f"- route_decision: {route_decision if route_decision is not None else 'null'}")
            lines.append(f"- router_reason: {router_reason if router_reason is not None else 'null'}")
            lines.append(f"- router_subgraph_task: {router_subgraph_task if router_subgraph_task is not None else 'null'}")
            lines.append(f"- subgraph_call_limit: {subgraph_call_limit}")
            lines.append(f"- subgraph_calls_made: {subgraph_calls_made}")
            lines.append(f"- subgraph_calls_remaining: {subgraph_calls_remaining}")

            rh = st.get("route_history")
            if isinstance(rh, list) and rh:
                lines.append(f"- route_history count: {len(rh)}")
                # include the last 15 in a readable trace
                tail = rh[-15:]
                lines.append("### Route trace (last 15)")
                for i, entry in enumerate(tail, 1):
                    if not isinstance(entry, dict):
                        lines.append(f"{i}) <invalid entry: {type(entry).__name__}>")
                        continue
                    r = entry.get("route", "unknown")
                    task = entry.get("subgraph_task")
                    outcome = entry.get("outcome")
                    ts = entry.get("timestamp")
                    s = f"{i}) {r}"
                    if isinstance(task, str) and task.strip():
                        s += f" — task: {task.strip()}"
                    if isinstance(outcome, str) and outcome.strip():
                        s += f" — outcome: {outcome.strip()}"
                    if ts is not None:
                        s += f" — t={ts}"
                    lines.append(s)
            else:
                lines.append("- route_history: (none)")
            lines.append("")

            # -------------------------
            # Known “important” keys (if present)
            # -------------------------
            lines.append("## Key artifacts / specs (if present)")
            for k in (
                "op_spec",
                "session_banner",
                "sandbox_spec",
                "sandbox_code",
                "sandbox_run_dir",
                "sandbox_error",
                "sandbox_attempts",
                "sandbox_repair_action",
                "sandbox_repair_reason",
                "sandbox_repair_message",
                "sandbox_repair_context",
                "sandbox_failure_history",
                "sandbox_withdrawn",
                "sandbox_withdrawal_summary",
            ):
                if k in st:
                    lines.append(f"### {k}")
                    lines.append("```json")
                    lines.append(json.dumps(st.get(k), indent=2, default=str))
                    lines.append("```")
            lines.append("")

            # Sandbox outputs
            if "sandbox_result" in st or "sandbox_generated" in st:
                lines.append("## Sandbox outputs")
                if "sandbox_result" in st:
                    lines.append("### sandbox_result")
                    lines.append("```json")
                    lines.append(json.dumps(st.get("sandbox_result"), indent=2, default=str))
                    lines.append("```")
                if "sandbox_generated" in st:
                    lines.append("### sandbox_generated")
                    lines.append("```json")
                    lines.append(json.dumps(st.get("sandbox_generated"), indent=2, default=str))
                    lines.append("```")
                lines.append("")

            # Imaging outputs / status
            imaging_keys = [
                "imaging_phase", "imaging_next_step", "imaging_plan",
                "imaging_task_summary", "imaging_agent_step_count",
                "imaging_agent_continue", "imaging_agent_decisions", "imaging_agent_notes",
            ]
            any_imaging = any(k in st for k in imaging_keys)
            if any_imaging:
                lines.append("## Imaging state")
                for k in imaging_keys:
                    if k in st:
                        lines.append(f"### {k}")
                        lines.append("```json")
                        lines.append(json.dumps(st.get(k), indent=2, default=str))
                        lines.append("```")
                lines.append("")

            # -------------------------
            # Generic results: *_summary and *_result (structured, full)
            # -------------------------
            # You earlier disliked the naive scanning in final_summary_node, but here the purpose is:
            # "full and detailed state view". So including these is appropriate.
            # If later you want finer control, you can replace this section.
            summary_like: dict[str, Any] = {}
            for key, val in st.items():
                if isinstance(key, str) and (key.endswith("_summary") or key.endswith("_result")):
                    summary_like[key] = val

            if summary_like:
                lines.append("## Computed outputs (*_summary / *_result)")
                lines.append("```json")
                lines.append(json.dumps(summary_like, indent=2, default=str))
                lines.append("```")
                lines.append("")

            return "\n".join(lines)

        except Exception:
            # absolute last resort
            return "{}"
