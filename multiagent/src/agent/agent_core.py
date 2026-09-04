from __future__ import annotations

import json
import logging
import os
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Hashable, Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import AliasChoices, BaseModel, Field, create_model, model_validator

from .logger_config import get_logger
from .ollama_retry import is_ollama_usage_limit_error
from .graph_state import (
    BiomedAgentState,
    DEFAULT_DEBUG,
    DEFAULT_MAIN_TOOL_CALL_LIMIT,
    DEFAULT_SUBGRAPH_CALL_LIMIT,
    StateViews,
)
from .subgraphs.subgraph_contract import SubgraphModule
from .subgraphs.code_sandbox import CodeSandboxSubgraph, MAX_SANDBOX_ATTEMPTS
from .subgraphs.domain_subgraph_factories import build_default_subgraphs, ALL_DOMAIN_TOOL_SPECS
from .subgraphs.generic_subgraph_agent import _import_module_from_path
from .system_prompts import (
    _build_direct_tool_agent_standard_system_prompt,
    _build_direct_tool_agent_strict_system_prompt,
    _build_multi_agent_strict_system_prompt,
    _build_python_agent_retry_strict_system_prompt,
)
from .tools.session_management import create_session
from .tools.todo_planning_tools import read_todos, write_todos
from .agent_execution_recorder import (
    append_tool_event,
    ensure_execution_trace,
    make_tool_event,
    persist_execution_trace,
)
from .time_utils import local_now_iso


load_dotenv()
os.environ["PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS"] = "true"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE_ROOT = os.environ.get("BIOMED_WORKSPACE_ROOT", str(REPO_ROOT))
ROUTER_TOKEN_BUDGET = 128000
ROUTER_MAX_CHARS = 8192
FINAL_SUMMARY_STATE_VIEW_MAX_CHARS = 48000
MAIN_GRAPH_TOOL_RESULT_MAX_CHARS = 8000
LOG_SEPARATOR_WIDTH = 70
MAX_SUBGRAPH_CALLS_PER_RUN = DEFAULT_SUBGRAPH_CALL_LIMIT
MAX_MAIN_TOOL_CALLS_PER_RUN = DEFAULT_MAIN_TOOL_CALL_LIMIT

logger = get_logger(__name__, level="DEBUG")

MainCompiledGraph = CompiledStateGraph[
    BiomedAgentState,
    None,
    BiomedAgentState,
    BiomedAgentState,
]

PATH_PATTERN = re.compile(
    r"(?:(?<![A-Za-z0-9])/[\w .~@%+=:,;(){}\[\]-]+(?:/[\w .~@%+=:,;(){}\[\]-]+)+|"
    r"(?:data|artifacts|sessions)/[^\s,;)\]}]+)"
)


def _extract_paths_from_messages(messages: list[Any], limit: int = 20) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        for match in PATH_PATTERN.finditer(content):
            path = match.group(0).rstrip(".,;:)]}")
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
                if len(paths) >= limit:
                    return paths
    return paths


def _task_mentions_any_path(task: str | None, paths: list[str]) -> bool:
    if not isinstance(task, str) or not task.strip():
        return False
    return any(path in task for path in paths)


def _as_message_list(value: Any) -> list[BaseMessage]:
    if value is None:
        return []
    if isinstance(value, list):
        return [msg for msg in value if isinstance(msg, BaseMessage)]
    if isinstance(value, BaseMessage):
        return [value]
    return []


def _sanitize_command_update(update: Any) -> dict[str, Any]:
    """
    Keep tool/progress messages out of the user-visible chat stream.

    LangGraph `messages` is the UI-visible conversation channel in this app.
    Command-style tools may accidentally return ToolMessage objects there; those
    belong in `internal_messages` so the agent can reason over them without
    exposing implementation chatter to the user.
    """
    if not isinstance(update, dict):
        return {}

    sanitized = dict(update)
    visible_messages = _as_message_list(sanitized.get("messages"))
    if not visible_messages:
        return sanitized

    leaked_tool_messages = [msg for msg in visible_messages if isinstance(msg, ToolMessage)]
    if not leaked_tool_messages:
        return sanitized

    remaining_visible = [msg for msg in visible_messages if not isinstance(msg, ToolMessage)]
    internal_messages = _as_message_list(sanitized.get("internal_messages"))
    sanitized["internal_messages"] = internal_messages + leaked_tool_messages

    if remaining_visible:
        sanitized["messages"] = remaining_visible
    else:
        sanitized.pop("messages", None)

    return sanitized


def _normalize_write_todos_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}

    normalized = dict(args)
    todos = normalized.get("todos")
    if not isinstance(todos, list):
        return normalized

    normalized_todos = []
    for todo in todos:
        if not isinstance(todo, dict):
            normalized_todos.append(todo)
            continue

        normalized_todo = dict(todo)
        if "content" not in normalized_todo and "text" in normalized_todo:
            normalized_todo["content"] = normalized_todo["text"]
        normalized_todos.append(normalized_todo)

    normalized["todos"] = normalized_todos
    return normalized


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction for model outputs with wrappers/prose."""
    if not isinstance(text, str):
        return None

    raw = text.strip()
    if not raw:
        return None

    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
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
                    parsed = json.loads(raw[start : index + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None

    return None


def _salvage_router_decision_from_prose(text: str, allowed_routes: list[str]) -> dict[str, Any] | None:
    """Recover the common router-explanation prose shape when JSON parsing fails."""
    if not isinstance(text, str) or not text.strip():
        return None

    route_match = re.search(
        r"Routing decision:\s*(?:\*\*)?([A-Za-z0-9_ -]+)(?:\*\*)?",
        text,
        flags=re.IGNORECASE,
    )
    if not route_match:
        return None

    route = route_match.group(1).strip().replace(" ", "_")
    if route not in allowed_routes:
        return None

    reason = ""
    reason_match = re.search(
        r"Reason:\s*(.*?)(?=\nSubgraph task:|\nTask:|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if reason_match:
        reason = reason_match.group(1).strip()

    subgraph_task: str | None = None
    task_match = re.search(
        r"Subgraph task:\s*(.*?)(?=\n\nUser-provided paths|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if task_match:
        subgraph_task = task_match.group(1).strip()

    return {
        "route": route,
        "reason": reason or "Recovered route from non-JSON router output.",
        "subgraph_task": subgraph_task,
    }

FILE_HINT_KEYS = (
    "filename",
    "name",
    "path",
    "paths",
    "file",
    "files",
    "url",
    "size",
    "size_mb",
    "mime",
    "mimetype",
)
PATH_HINT_RE = re.compile(r"(\/|\\).+\.[A-Za-z0-9]{1,6}$")
URL_HINT_RE = re.compile(r"https?://")


def log_to_both(message: str, level: str = "warning") -> None:
    langgraph_logger = logging.getLogger("langgraph")
    log_method = getattr(logger, level, logger.warning)
    langgraph_method = getattr(langgraph_logger, level, langgraph_logger.warning)
    log_method(message)
    langgraph_method(message)


def log_separator(title: str, width: int = LOG_SEPARATOR_WIDTH) -> None:
    sep = "=" * width
    logger.debug(f"{sep} {title} {sep}")


def truncate_middle(text: Any, max_chars: int, label: str = "content") -> str:
    raw = text if isinstance(text, str) else str(text)
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw

    marker = (
        f"\n\n[... {label} truncated: omitted "
        f"{len(raw) - max_chars:,} characters to keep the final prompt within budget ...]\n\n"
    )
    if len(marker) >= max_chars:
        return raw[:max_chars]

    remaining = max_chars - len(marker)
    head_chars = max(remaining // 2, 0)
    tail_chars = max(remaining - head_chars, 0)
    return raw[:head_chars] + marker + raw[-tail_chars:]


def collect_file_hints(obj: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stack = [obj]

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if any(key in current for key in ("path", "file", "filename", "url", "size", "size_mb")):
                slim = {key: current.get(key) for key in FILE_HINT_KEYS if key in current}
                if slim:
                    candidates.append(slim)
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            stack.extend(reversed(current))
        elif isinstance(current, str) and (PATH_HINT_RE.search(current) or URL_HINT_RE.search(current)):
            candidates.append({"string_hint": current})

    return candidates


def parse_tool_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) and "tool" in obj and "args" in obj else None


def make_todo_models(allowed_routes: list[str]) -> tuple[type[BaseModel], type[BaseModel]]:
    Route = Literal[tuple(allowed_routes)]  # type: ignore

    TodoItemModel = create_model(
        "TodoItem",
        id=(str, ...),
        content=(str, Field(..., validation_alias=AliasChoices("content", "text"))),
        status=(Literal["pending", "in_progress", "done", "blocked"], "pending"),
        route=(Route, ...),
        subgraph_task=(str | None, None),
        depends_on=(list[str], Field(default_factory=list)),
    )

    class _TodoPlanBase(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def _wrap_list(cls, v: Any) -> Any:
            if isinstance(v, list):
                return {"todos": v}
            return v

    TodoPlanModel = create_model(
        "TodoPlan",
        __base__=_TodoPlanBase,
        todos=(list[TodoItemModel], ...),  # type: ignore[valid-type]
    )

    return TodoItemModel, TodoPlanModel


def branch_from_router(state: BiomedAgentState) -> str:
    return state["route_decision"]



class AgentMode(str, Enum):
    """Experiment architectures, named by topology and prompt strength."""

    MULTI_AGENT_STANDARD = "full_weak"  # Domain subgraphs plus custom tools; reduced constraint prompt.
    MULTI_AGENT_STRICT = "full_strong"  # Domain subgraphs plus custom tools; strong constraint prompt.

    PYTHON_AGENT_RETRY_STANDARD = "pi_retries_weak"  # Python interpreter only; retry baseline; reduced constraint prompt.
    PYTHON_AGENT_RETRY_STRICT = "pi_retries_strong"  # Python interpreter only; retry baseline; strong constraint prompt.
    PYTHON_AGENT_ONESHOT = "pi_single"  # Python interpreter only, one execution attempt.

    LLM_ONLY = "llm_only"  # Direct LLM response: no tools and no code execution.

    DIRECT_TOOL_AGENT_STANDARD = "single_agent_weak"  # All domain tools flat; reduced constraint prompt.
    DIRECT_TOOL_AGENT_STRICT = "single_agent_strong"  # All domain tools flat; strong constraint prompt.


def _load_domain_tools(
    tool_specs: list[tuple[str, str]],
) -> dict[str, "BaseTool"]:
    """
    Load BaseTool instances from a list of (friendly_name, absolute_path) pairs.

    Mirrors the loading logic in SubgraphAgent._load_scoped_tools but operates
    without a subgraph context so the tools can be registered directly on the
    main-graph agent.
    """
    import pathlib
    registry: dict[str, BaseTool] = {}
    seen_paths: set[str] = set()

    for friendly_name, path in tool_specs:
        canonical = str(pathlib.Path(path).expanduser().resolve())
        if canonical in seen_paths:
            continue
        seen_paths.add(canonical)

        try:
            module = _import_module_from_path(path)
        except Exception as e:
            logger.warning("[single_agent][tools] Failed to import %s: %r", path, e)
            continue

        exported = getattr(module, "EXPORTED_TOOLS", None)
        if not isinstance(exported, dict):
            logger.warning("[single_agent][tools] No EXPORTED_TOOLS in %s", path)
            continue

        for key, tool in exported.items():
            if not isinstance(tool, BaseTool):
                continue
            if key in registry:
                logger.debug(
                    "[single_agent][tools] Skipping duplicate tool key %r from %s",
                    key, path,
                )
                continue
            registry[key] = tool
            logger.info("[single_agent][tools] Registered tool: %s", key)

    return registry


class BiomedAgent:
    _truncate_middle = staticmethod(truncate_middle)
    _normalize_write_todos_args = staticmethod(_normalize_write_todos_args)
    make_todo_models = staticmethod(make_todo_models)

    @staticmethod
    def _build_subgraph_tool_catalog(
        subgraphs: list[SubgraphModule],
    ) -> dict[str, dict[str, BaseTool]]:
        """Collect tools by owning subgraph without exposing them to the main graph."""
        catalog: dict[str, dict[str, BaseTool]] = {}
        for subgraph in subgraphs:
            tools_by_name = getattr(subgraph, "tools_by_name", None)
            if isinstance(tools_by_name, dict) and tools_by_name:
                catalog[subgraph.key] = dict(tools_by_name)
        return catalog

    def __init__(
        self,
        llm: BaseChatModel,
        checkpoint_path: str = "checkpoints.db",
        debug: bool = DEFAULT_DEBUG,
        mode: AgentMode = AgentMode.MULTI_AGENT_STANDARD,
    ) -> None:
        self.views = StateViews()
        self.debug = debug
        self.checkpoint_path = checkpoint_path
        self.llm = llm
        self.mode = mode

        strict_tool_loading = os.environ.get(
            "STRICT_SUBGRAPH_TOOL_LOADING",
            "",
        ).lower() in {"1", "true", "yes", "on"}

        # --- subgraph selection -------------------------------------------
        if mode in (AgentMode.MULTI_AGENT_STANDARD, AgentMode.MULTI_AGENT_STRICT):
            self.subgraphs: list[SubgraphModule] = [
                CodeSandboxSubgraph(llm=self.llm),
                *build_default_subgraphs(
                    self.llm,
                    strict_tool_loading=strict_tool_loading,
                ),
            ]
        elif mode in (AgentMode.PYTHON_AGENT_RETRY_STANDARD, AgentMode.PYTHON_AGENT_RETRY_STRICT):
            self.subgraphs = [CodeSandboxSubgraph(llm=self.llm, max_repairs=MAX_SANDBOX_ATTEMPTS, unrestricted=True)]
        elif mode == AgentMode.PYTHON_AGENT_ONESHOT:
            self.subgraphs = [CodeSandboxSubgraph(llm=self.llm, max_repairs=0, unrestricted=True)]
        else:  # LLM_ONLY, DIRECT_TOOL_AGENT_STRICT, DIRECT_TOOL_AGENT_STANDARD
            self.subgraphs = []

        for sg in self.subgraphs:
            has_tool_loading_diagnostics = hasattr(sg, "tool_loading_errors")
            errors = getattr(sg, "tool_loading_errors", [])
            if errors:
                logger.warning(
                    "[graph-init] Subgraph %s instantiated with %d tool-loading error(s).",
                    sg.key,
                    len(errors),
                )
                for err in errors:
                    logger.warning(
                        "[graph-init] subgraph=%s path=%s stage=%s error=%s",
                        sg.key,
                        err.get("path"),
                        err.get("stage"),
                        err.get("error"),
                    )
            if has_tool_loading_diagnostics and not getattr(sg, "tools_by_name", {}):
                logger.error(
                    "[graph-init] Subgraph %s has zero loaded tools.",
                    sg.key,
                )
        self.subgraph_by_key = {m.key: m for m in self.subgraphs}
        self.subgraph_tool_catalog = self._build_subgraph_tool_catalog(self.subgraphs)

        # both PI baselines allow exactly one outer subgraph dispatch;
        # Python retry modes differ from the one-shot mode only in their repair loop.
        self._subgraph_call_limit = (
            1 if mode in (AgentMode.PYTHON_AGENT_ONESHOT, AgentMode.PYTHON_AGENT_RETRY_STANDARD, AgentMode.PYTHON_AGENT_RETRY_STRICT)
            else MAX_SUBGRAPH_CALLS_PER_RUN
        )

        # flat tool-loop modes have no subgraph budget to bound their runs, so give
        # them an equivalent direct-tool-call budget (mirrors _subgraph_call_limit above).
        self._main_tool_call_limit = (
            MAX_MAIN_TOOL_CALLS_PER_RUN
            if mode in (AgentMode.DIRECT_TOOL_AGENT_STRICT, AgentMode.DIRECT_TOOL_AGENT_STANDARD)
            else None
        )

        # --- main-graph tool selection ------------------------------------
        # Domain tools remain scoped to their owning subgraphs.  This map is
        # deliberately separate so the react_free path cannot invoke them.
        todo_planning_tools = [write_todos, read_todos] if mode in (AgentMode.MULTI_AGENT_STANDARD, AgentMode.MULTI_AGENT_STRICT) else []
        session_management_tools = [create_session]

        self.main_graph_tools: list[BaseTool] = [
            *todo_planning_tools,
            *session_management_tools,
        ]
        # main_graph_tools_by_name includes create_session so ensure_session_node can always find it,
        # even for PI modes where the LLM never calls tools directly.
        self.main_graph_tools_by_name: dict[str, BaseTool] = {
            tool.name: tool for tool in self.main_graph_tools
        }

        if mode in (AgentMode.DIRECT_TOOL_AGENT_STRICT, AgentMode.DIRECT_TOOL_AGENT_STANDARD):
            domain_tools = _load_domain_tools(ALL_DOMAIN_TOOL_SPECS)
            self.main_graph_tools_by_name.update(domain_tools)
            self.main_graph_tools = list(self.main_graph_tools_by_name.values())

        tools_description = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.main_graph_tools
        )

        # --- system prompt ------------------------------------------------
        if mode == AgentMode.LLM_ONLY:
            self.system_prompt = (
                "You are a helpful biomedical assistant. "
                "Answer questions in natural language and provide Python code examples where relevant. "
                "You do not have access to a code execution environment or any tools."
            )
        elif mode in (AgentMode.PYTHON_AGENT_RETRY_STANDARD, AgentMode.PYTHON_AGENT_ONESHOT):
            self.system_prompt = (
                "You are a biomedical assistant with access to a Python 3.11 execution environment. "
                "Solve tasks by writing and running Python code.\n\n"
                f"**TOOLS:**\n{tools_description}"
            )
        elif mode == AgentMode.PYTHON_AGENT_RETRY_STRICT:
            self.system_prompt = _build_python_agent_retry_strict_system_prompt()
        elif mode == AgentMode.DIRECT_TOOL_AGENT_STRICT:
            self.system_prompt = _build_direct_tool_agent_strict_system_prompt(self.main_graph_tools_by_name)
        elif mode == AgentMode.DIRECT_TOOL_AGENT_STANDARD:
            self.system_prompt = _build_direct_tool_agent_standard_system_prompt(self.main_graph_tools_by_name)
        elif mode == AgentMode.MULTI_AGENT_STRICT:
            self.system_prompt = _build_multi_agent_strict_system_prompt(tools_description)
        else:  # MULTI_AGENT_STANDARD
            self.system_prompt = f""" You are an helpfull agent assistant for biomedical porpuses.

        **SESSION & FILES:**
        You have a session ID, a read-only session data folder, and a bounded list of visible attachments.
        - You can operate on files using their paths (e.g., in `preprocess_microscopy_tiff`).
        - DO NOT attempt to read file bytes directly.
        - Binary upload into this graph is not supported; input datasets must already live under workspace `data/`.
        - If the session contains `data`, treat it as task input only. Read from it when
          needed, but place all derived files in writable session output folders outside `data`.

        **WORKFLOW:**
        1. Plan tasks at the beginning using the todo route. especially if the task requires a lot of steps.
        2. Execute the tasks in the order of the TODO list or follow your mental plan.

        **TOOLS:**
        {tools_description}

        **JSON RESPONSE FORMAT:**
        {{"tool": "nome_tool", "args": {{"param1": "value1"}}}}
        """

    def _log_truncation_to_langgraph(self, response: Any, where: str) -> None:
        """
        If an LLM response was truncated by length and has empty content,
        log it both to the agent logger and to the LangGraph default logger.
        """
        try:
            resp_meta = getattr(response, "response_metadata", None)
            finish_reason = None
            if isinstance(resp_meta, dict):
                finish_reason = resp_meta.get("finish_reason")

            content = getattr(response, "content", "") or ""

            if finish_reason == "length" and not content.strip():
                msg = (
                    f"[LLM TRUNCATION] finish_reason='length' with empty content in {where}. "
                    "The model likely hit its output token limit."
                )
                log_to_both(msg, level="error")
        except Exception as e:
            logger.exception(f"[LLM TRUNCATION CHECK FAILED in {where}] {e}")

    def start_run_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Reset per-run bookkeeping when a saved thread is entered again.

        Session, attachments, todos, and messages are conversational state and
        should survive across user turns. The subgraph budget, route trace, and
        execution trace describe one graph execution, so they need a fresh value each
        time execution starts from START.
        """
        state_for_new_trace = dict(state)
        state_for_new_trace.pop("execution_trace", None)

        return {
            "subgraph_call_limit": self._subgraph_call_limit,
            "subgraph_calls_made": 0,
            "main_tool_call_limit": self._main_tool_call_limit or 0,
            "main_tool_calls_made": 0,
            "route_history": [],
            "execution_trace": ensure_execution_trace(state_for_new_trace),
            "execution_trace_file": "",
            "execution_trace_index_file": "",
        }

    def _build_router_conversation_text(self, state: BiomedAgentState, max_chars: int = ROUTER_MAX_CHARS) -> str:
        """
        Build a compact, user-visible conversation snapshot.

        - Uses LangGraph's trim_messages to keep the most recent messages within a token budget.
        - Formats as 'ROLE: content' lines.
        - Best-effort: never returns empty if there are messages.
        """
        history = state.get("messages") or []
        if not isinstance(history, list) or not history:
            return "No prior messages."

        approx_max_tokens = max(max_chars // 4, 1)

        trimmed = trim_messages(
            history,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=approx_max_tokens,
            start_on="human",
            end_on=None,
        )
        if not trimmed:
            trimmed = trim_messages(
                history,
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=approx_max_tokens,
                start_on=None,
                end_on=None,
            )

        lines = []
        for msg in trimmed:
            if isinstance(msg, HumanMessage):
                role = "USER"
            elif isinstance(msg, AIMessage):
                role = "ASSISTANT"
            elif isinstance(msg, ToolMessage):
                role = "TOOL"
            else:
                role = type(msg).__name__.upper()

            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content)
            text = text.replace("\n", "\\n")
            if len(text) > 800:
                text = text[:800] + "…"
            lines.append(f"{role}: {text}\n")
        out = "".join(lines).strip()

        dropped = len(history) - len(trimmed)
        if dropped > 0:
            trimming_notice = (
                f"[ROUTER CONTEXT] Conversation trimmed: kept {len(trimmed)} messages, "
                f"dropped {dropped} older messages to respect ~{max_chars} chars (~{approx_max_tokens} tokens)."
            )
            log_to_both(trimming_notice, level="warning")

        content = getattr(history[-1], "content", "")
        text = content if isinstance(content, str) else str(content)
        text = text.replace("\n", "\\n")
        return out or (text[:800] + "…" if len(text) > 800 else text)

    def ensure_session_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Node that checks if a session exists. If not, it calls the
        `create_session` tool directly, updates the state, and records
        the tool event in execution_trace.
        """
        log_separator("DEBUG: ensure_session_node", width=25)
        session = state.get("session")
        logger.debug(f"Session on entry: type={type(session)}, value={session}")

        if isinstance(session, dict) and "id" in session:
            log_separator("DEBUG: ensure_session_node end", width=70)
            return {
                "execution_trace": ensure_execution_trace(dict(state))
            }

        logger.info("DECISION: Session is missing or invalid. Creating a new one.")
        tool = self.main_graph_tools_by_name["create_session"]
        tool_call_id = f"manual:create_session:{uuid.uuid4().hex[:8]}"

        try:
            tool_message_result = tool.invoke(
                {
                    "type": "tool_call",
                    "name": "create_session",
                    "args": {},
                    "id": tool_call_id,
                }
            )

            logger.debug(f"ToolMessage result from tool.invoke: {tool_message_result}")

            session_data = json.loads(tool_message_result.content)
            logger.info(f"Successfully parsed session data: {session_data}")
            log_separator("DEBUG: ensure_session_node end", width=70)
            msgs = state.get("messages", []) or []
            latest_ui = msgs[-1] if isinstance(msgs, list) and msgs else None

            temp_state = dict(state)
            temp_state["session"] = session_data

            event = make_tool_event(
                scope="main_graph",
                subgraph_key=None,
                step_index=None,
                tool_name="create_session",
                tool_args={},
                success=True,
                output=session_data,
                error=None,
            )

            return {
                "session": session_data,
                "execution_trace": append_tool_event(temp_state, event),
                "internal_messages": [latest_ui] if latest_ui is not None else [],
            }

        except Exception as e:
            err_msg = f"Error while executing create_session: {e}"
            logger.critical(err_msg)
            log_separator("DEBUG: ensure_session_node failed", width=70)
            raise

    def banner_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Node that prepares the dynamic "banner" (a context string)
        for the LLM, listing the session ID and attached files.
        This string is stored in state.session_banner.
        """
        session = state.get("session", {})
        if not isinstance(session, dict):
            logger.warning("Session state corrupted, resetting to empty dict.")
            session = {}

        session_id = session.get("id", "N/A (Session not created)")
        session_path = session.get("path")
        data_path = session.get("data_path") or (
            os.path.join(session_path, "data") if isinstance(session_path, str) else None
        )
        attachments = state.get("attachments") or []
        workspace_root = state.get("workspace_root") or DEFAULT_WORKSPACE_ROOT

        banner_lines = [
            "--- SESSION CONTEXT ---",
            f"Session ID: {session_id}",
        ]
        if isinstance(session_path, str) and session_path:
            banner_lines.append(
                f"Session path (writable output dir — pass as session_path to every tool): {session_path}"
            )
        if isinstance(data_path, str) and data_path:
            banner_lines.append(
                f"Session data folder (read-only input — do NOT write here): {data_path}"
            )
        if isinstance(workspace_root, str) and workspace_root:
            banner_lines.append(
                f"Workspace root (pass as workspace_root when tools accept it): {workspace_root}"
            )

        if not isinstance(attachments, list):
            attachments = []
        visible_attachments, hidden_non_data = self.views.visible_attachments(
            attachments,
            session=session,
            workspace_root=workspace_root,
        )

        if visible_attachments:
            banner_lines.append("Visible Attachments:")

            for att in visible_attachments:
                if not isinstance(att, dict):
                    banner_lines.append(f"- <invalid attachment: {type(att).__name__}>")
                    continue

                f_name = att.get("filename") or att.get("name") or "unknown_file"
                f_path = att.get("path") or "unknown_path"

                size_raw = att.get("size_mb", None)
                try:
                    f_size = float(size_raw) if isinstance(size_raw, (int, float, str)) else 0.0
                    size_str = f"{f_size:.3f} MB"
                except Exception:
                    size_str = "N/A"

                origin_path = att.get("origin_path")
                if isinstance(origin_path, str) and origin_path and origin_path != f_path:
                    banner_lines.append(f"- {f_name} (Path: {f_path}, Size: {size_str}, Origin: {origin_path})")
                else:
                    banner_lines.append(f"- {f_name} (Path: {f_path}, Size: {size_str})")
            if hidden_non_data:
                banner_lines.append(f"Hidden non-data attachments: {hidden_non_data}")
        else:
            banner_lines.append("Visible Attachments: None")

        banner_lines.append("--- END CONTEXT ---")
        return {"session_banner": "\n".join(banner_lines)}

    def tool_node(self, state: BiomedAgentState) -> dict[str, Any]:
        internal = state.get("internal_messages", []) or []
        if not isinstance(internal, list) or not internal:
            return {"messages": []}

        last = internal[-1]
        content = getattr(last, "content", str(last))

        try:
            parsed = json.loads(content) if isinstance(content, str) else None
        except Exception:
            parsed = None

        if not isinstance(parsed, dict):
            return {"messages": []}

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {}) or {}

        if not tool_name:
            return {"messages": []}

        if tool_name == "write_todos":
            tool_args = _normalize_write_todos_args(tool_args)

        tool = self.main_graph_tools_by_name.get(tool_name)
        tool_call_id = f"manual:{tool_name}:{uuid.uuid4().hex[:8]}"

        if tool is None:
            err_msg = f"Error: unknown tool '{tool_name}'"
            tm = ToolMessage(content=err_msg, tool_call_id=tool_call_id, name=f"main_graph_tool_{tool_name}")

            event = make_tool_event(
                scope="main_graph",
                subgraph_key=None,
                step_index=None,
                tool_name=tool_name,
                tool_args=tool_args,
                success=False,
                output=None,
                error=err_msg,
            )

            return {
                "internal_messages": [tm],
                "messages": [],
                "execution_trace": append_tool_event(dict(state), event),
            }

        # flat single-agent modes have no subgraph budget to bound the run, so track
        # direct tool-call usage against self._main_tool_call_limit (see should_continue).
        main_tool_calls_made = None
        if self.mode in (AgentMode.DIRECT_TOOL_AGENT_STRICT, AgentMode.DIRECT_TOOL_AGENT_STANDARD):
            try:
                main_tool_calls_made = max(int(state.get("main_tool_calls_made", 0) or 0), 0) + 1
            except Exception:
                main_tool_calls_made = 1

        try:
            tool_call = {
                "type": "tool_call",
                "name": tool_name,
                "args": tool_args,
                "id": tool_call_id,
            }
            result = tool.invoke(tool_call)

            if isinstance(result, Command):
                trace = ToolMessage(
                    content=f"[tool:{tool_name}] Command update applied.",
                    tool_call_id=tool_call_id,
                    name=f"main_graph_tool_{tool_name}",
                )

                upd = _sanitize_command_update(result.update)
                internal_messages = _as_message_list(upd.get("internal_messages"))
                upd["internal_messages"] = internal_messages + [trace]
                if main_tool_calls_made is not None:
                    upd["main_tool_calls_made"] = main_tool_calls_made

                output_snapshot = dict(upd)

                merged_state = dict(state)
                merged_state.update(upd)

                event = make_tool_event(
                    scope="main_graph",
                    subgraph_key=None,
                    step_index=None,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    success=True,
                    output=output_snapshot,
                    error=None,
                )

                upd["execution_trace"] = append_tool_event(merged_state, event)
                return upd

            try:
                if hasattr(result, "__dict__"):
                    result_str = json.dumps(result.__dict__)
                else:
                    result_str = json.dumps(result)
            except Exception:
                result_str = str(result)

            if self.mode in (AgentMode.DIRECT_TOOL_AGENT_STRICT, AgentMode.DIRECT_TOOL_AGENT_STANDARD):
                original_result_len = len(result_str)
                result_str = truncate_middle(
                    result_str,
                    MAIN_GRAPH_TOOL_RESULT_MAX_CHARS,
                    label=f"{tool_name} result",
                )
                if len(result_str) < original_result_len:
                    logger.warning(
                        "[tool_node][%s] Tool result trimmed from %d to %d characters "
                        "before being added to internal_messages.",
                        tool_name, original_result_len, len(result_str),
                    )

            tm = ToolMessage(content=result_str, tool_call_id=tool_call_id, name=f"main_graph_tool_{tool_name}")

            event = make_tool_event(
                scope="main_graph",
                subgraph_key=None,
                step_index=None,
                tool_name=tool_name,
                tool_args=tool_args,
                success=True,
                output=result,
                error=None,
            )

            return {
                "internal_messages": [tm],
                "messages": [],
                "execution_trace": append_tool_event(dict(state), event),
                **({"main_tool_calls_made": main_tool_calls_made} if main_tool_calls_made is not None else {}),
            }

        except Exception as e:
            err_msg = f"Error while executing {tool_name}: {e}"
            tm = ToolMessage(content=err_msg, tool_call_id=tool_call_id, name=f"main_graph_tool_{tool_name}")

            event = make_tool_event(
                scope="main_graph",
                subgraph_key=None,
                step_index=None,
                tool_name=tool_name,
                tool_args=tool_args,
                success=False,
                output=None,
                error=err_msg,
            )

            return {
                "internal_messages": [tm],
                "messages": [],
                "execution_trace": append_tool_event(dict(state), event),
                **({"main_tool_calls_made": main_tool_calls_made} if main_tool_calls_made is not None else {}),
            }

    def debug_input_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Debug-only node that logs the incoming graph input (best-effort, never crashes),
        with emphasis on messages, file/attachment hints, and session info.
        Does not mutate state.
        """
        try:
            log_separator("DEBUG: graph input", width=28)

            msgs = state.get("messages") or []
            if not isinstance(msgs, list):
                logger.debug(f"Messages: <invalid type {type(msgs).__name__}>")
                msgs = []

            logger.debug(f"Messages: type=list, len={len(msgs)}")

            file_hints = []
            if msgs:
                last = msgs[-1]
                last_type = type(last).__name__
                last_content = getattr(last, "content", str(last))
                logger.debug(f"- Last message: {last_type}: {last_content!r}")

                if isinstance(last_content, str) and last_content.strip():
                    parsed = None
                    try:
                        parsed = json.loads(last_content)
                    except Exception:
                        parsed = None

                    try:
                        if parsed is not None:
                            file_hints = collect_file_hints(parsed)
                        else:
                            file_hints = collect_file_hints(last_content)
                    except Exception as e:
                        logger.debug(f"[debug_input_node] hint extraction failed: {e}")

                if file_hints:
                    logger.debug("File hints in last message:")
                    for i, h in enumerate(file_hints[:10], 1):
                        logger.debug(f"  {i}. {h}")
                    if len(file_hints) > 10:
                        logger.debug(f"  ... ({len(file_hints) - 10} more)")

            atts = state.get("attachments", None)
            logger.debug("Attachments in state:")

            if not atts:
                logger.debug("  - None")
            else:
                if not isinstance(atts, list):
                    logger.debug(f"  - <invalid type {type(atts).__name__}>")
                    atts = []

                if not atts:
                    logger.debug("  - []")
                else:
                    for i, att in enumerate(atts, 1):
                        if not isinstance(att, dict):
                            logger.debug(f"  {i}. <invalid attachment: {type(att).__name__}>")
                            continue

                        fname = att.get("filename") or att.get("name") or "unknown_file"
                        fpath = att.get("path") or "unknown_path"

                        size_raw = att.get("size_mb", att.get("size", None))
                        try:
                            size_str = f"{float(size_raw):.3f} MB" if isinstance(size_raw, (int, float, str)) else "N/A"
                        except Exception:
                            size_str = "N/A"

                        origin_path = att.get("origin_path")
                        if isinstance(origin_path, str) and origin_path and origin_path != fpath:
                            logger.debug(f"  {i}. {fname} | path={fpath} | size={size_str} | origin={origin_path}")
                        else:
                            logger.debug(f"  {i}. {fname} | path={fpath} | size={size_str}")

            sess = state.get("session")
            if isinstance(sess, dict) and sess.get("id"):
                sid = sess.get("id")
                spath = sess.get("path")
                if spath:
                    logger.debug(f"Session detected: id={sid} | path={spath}")
                else:
                    logger.debug(f"Session detected: id={sid}")
            else:
                logger.debug("Session detected: None/invalid (will be created next)")

            log_separator("DEBUG: graph input end", width=70)
        except Exception as e:
            logger.exception(f"[DEBUG-NODE ERROR] {e}")

        return {}

    def call_model(self, state: BiomedAgentState) -> dict[str, Any]:
        banner = state.get("session_banner", "--- SESSION: N/A ---")
        system = SystemMessage(content=self.system_prompt + "\n\n" + banner, name="main_graph_agent_system")

        internal = _as_message_list(state.get("internal_messages"))

        msgs = _as_message_list(state.get("messages"))
        if msgs:
            last_ui = msgs[-1]
            if last_ui is not None:
                if not internal or internal[-1] is not last_ui:
                    internal = internal + [last_ui]

        response = self.llm.invoke([system, *internal])
        self._log_truncation_to_langgraph(response, where="BiomedAgent.call_model")

        updates = {"internal_messages": [response]}

        is_tool_call = (
            (hasattr(response, "tool_calls") and bool(getattr(response, "tool_calls", None)))
            or parse_tool_json(getattr(response, "content", "")) is not None
        )

        if not is_tool_call:
            updates["messages"] = [response]
        else:
            logger.debug("[call_model] Tool call generated. Not adding to user-visible messages.")

        return updates

    def human_approval_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """Pause here for human approval; planner can set plan_approved later."""
        return {"plan_approved": False}

    def should_continue(self, state: BiomedAgentState) -> Literal["tools", "human_approval", "todo_planner"] | str:
        if not state.get("plan_approved", True):
            return "human_approval"

        internal = state.get("internal_messages", []) or []
        if not isinstance(internal, list) or not internal:
            return END

        last = internal[-1]
        content = getattr(last, "content", str(last))

        parsed_tool = parse_tool_json(content)
        wants_tool_call = parsed_tool is not None or (
            hasattr(last, "tool_calls") and bool(getattr(last, "tool_calls", None))
        )

        if wants_tool_call and self._main_tool_call_limit is not None:
            try:
                main_tool_calls_made = max(int(state.get("main_tool_calls_made", 0) or 0), 0)
            except Exception:
                main_tool_calls_made = 0

            if main_tool_calls_made >= self._main_tool_call_limit:
                logger.warning(
                    "[should_continue] Main-graph tool-call limit reached (%d/%d); "
                    "stopping the tool loop and routing to final_summary instead of crashing "
                    "on the graph recursion limit.",
                    main_tool_calls_made, self._main_tool_call_limit,
                )
                return END

        if parsed_tool is not None:
            if (
                parsed_tool.get("tool") == "write_todos"
                and state.get("route_decision") != "todo_planner"
            ):
                logger.warning(
                    "[should_continue] write_todos was requested outside todo_planner; "
                    "rerouting through todo_planner."
                )
                return "todo_planner"
            return "tools"

        if hasattr(last, "tool_calls") and bool(getattr(last, "tool_calls", None)):
            return "tools"

        return END


    def route_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """Decide which path to take next and explain the routing decision (structured output)."""

        route_descriptions = "".join(f"- {sg.key}: {sg.description()}\n" for sg in self.subgraphs)
        subgraph_keys = list(self.subgraph_by_key.keys())
        subgraph_call_limit = self._subgraph_call_limit

        try:
            subgraph_calls_made = max(int(state.get("subgraph_calls_made", 0) or 0), 0)
        except Exception:
            subgraph_calls_made = 0

        subgraph_calls_remaining = max(subgraph_call_limit - subgraph_calls_made, 0)

        allowed_routes = [*subgraph_keys, "react_free", "todo_planner", "final_answer", "final_summary"]

        router_state_view = self.views.router_view(state)
        route_history_view = self.views.route_history_view(state, limit=15)
        workspace_root = str(state.get("workspace_root") or DEFAULT_WORKSPACE_ROOT)

        router_instruction = f"""
You are the workflow router for a biomedical agent system.

<objective>
Choose the single best next route for progressing the user's request.
Routing is iterative: after a route completes, control returns to you.
Your job is not to answer the user directly unless a terminal route is clearly best.
</objective>

<available_routes>
You may choose exactly one route:
- one subgraph key from the provided route list
- react_free
- todo_planner
- final_answer
- final_summary
</available_routes>

<route_semantics>
- A subgraph route is the default when there is a semantically suitable next executable action.
- react_free is a last-resort route for turns that do not map cleanly to any available subgraph.
- Use react_free only when no semantically suitable subgraph exists for the next step, or when the turn is better handled directly without delegating to a subgraph.
- final_answer is only for explanation-only turns where no execution is needed.
- final_summary is only for recap/report requests.
- todo_planner is for multi-step work when creating or updating a plan is the highest-value next step.
</route_semantics>

<decision_policy>
Rules:
- You MUST respect each subgraph description. Do not ask a subgraph to perform tasks outside its scope.
- Treat subgraph capabilities as disjoint. If a subgraph explicitly advertises a capability, assume the other subgraphs almost certainly do not provide that capability unless their own descriptions explicitly say so.
- If the request contains multiple phases where a later phase depends on an artifact produced by an earlier phase, route the earliest unmet prerequisite phase first.
- Do not route to an analysis or evaluation subgraph until the artifact to be analyzed already exists in state, attachments, or the user-provided paths.
- If the user asks to generate a predicted mask and then evaluate it, route first to the segmentation-capable subgraph. Route to quantitative analysis only after the predicted mask has been produced.
- A user-provided raw or preprocessed image volume is not a predicted segmentation mask unless the user explicitly identifies it as a mask or segmentation.
- If the request asks for detected lacunae, lacunar counts, lacunar volumes, density, centroids, spatial bins, nearest-neighbor distances, or size filtering from an original/raw/preprocessed microscopy volume and no explicit mask/segmentation path is already available, route to imaging_manipulation first to produce the lacunar mask. Quantitative analysis can consume that generated mask afterward.
- For lacunar morphometry feature-extraction tasks, the canonical source of truth is `calculate_lacunae_parameters`. The workflow must call `calculate_lacunae_parameters` before deriving any lacunar morphometry values, counts, distributions, percentiles, filters, or summaries. Do not route to code_sandbox to compute surface area, volume filtering, or morphometry statistics directly from a segmentation mask unless `calculate_lacunae_parameters` has already succeeded and produced the canonical per-lacuna morphometry CSV. If `calculate_lacunae_parameters` fails, route for retry/reporting of that failure; do not replace it with a custom algorithm.
- If a subgraph reports that a task was only partially supported, treat the reported unsupported step(s) as remaining work. If no other subgraph explicitly advertises the needed capability and the step is a concrete computable file/data operation, route the remaining unsupported step(s) to code_sandbox.
- If a subgraph reports that an operation is unsupported for that subgraph, do not send the same unsupported operation back to the same subgraph unless new tools or new inputs have appeared.
- Never justify a route by assuming a subgraph can perform a capability that its description explicitly says it does not perform.
- If one subgraph is clearly the best semantic match for the next step, choose that subgraph.
- If the user's request spans multiple logically distinct phases, route only the highest-value next phase.
- Do not choose react_free merely because the request is incomplete, ambiguous, or underspecified.
- If a suitable subgraph still exists, choose the subgraph and encode any uncertainty, assumptions, or limitations in the reason and subgraph_task.
- Choose react_free only when there is no semantically suitable subgraph for the next step.
- If the user is asking for explanation only and no subgraph execution is needed, choose final_answer.
- If a TODO plan already contains relevant pending steps, prefer routing in a way that progresses those steps instead of creating conflicting or duplicate work.
- Respect the current structure: communicate uncertainty or missing context in the reason field, not via extra output fields.
</decision_policy>

<subgraph_task_contract>
If you choose a subgraph:
- subgraph_task must be single-purpose
- state the goal
- include the relevant inputs already available
- state the expected output or state change
- do not include low-level tool instructions
- do not bundle multiple phases into one task
- remember that the subgraph will see this task as its starting instruction, so include the context it needs
</subgraph_task_contract>

<file_rules>
- If a user-mentioned path matches an attachment origin_path, prefer the attachment path.
- The official input channel is the read-only session data link: use `data/...`
  paths for datasets placed under the configured workspace data folder.
- Subgraphs may read files under the session directory and the configured workspace root.
- Treat relative `data/...` paths as input data. When a routed task produces files,
  make the expected destination a writable session output location outside `data`.
- The workspace root is read-only and exists so generic QA/data tasks can use relative paths without being copied into the per-run session folder.
- Configured workspace root: {workspace_root}
- Never instruct a subgraph to operate outside the allowed roots provided in context.
- Do not invent attachments for folder/pattern instructions. Preserve explicit paths, and let the chosen subgraph inspect folders with filesystem tools when needed.
</file_rules>

<budget_rules>
- Respect the subgraph budget.
- If no subgraph calls remain, do not choose a subgraph.
- Low budget is a tie-breaker, not a default reason to choose react_free.
</budget_rules>

<output_contract>
Return only a JSON object with:
{{
  "route": "<one of: {', '.join(allowed_routes)}>",
  "reason": "<short reason explanation>",
  "subgraph_task": "<string or null>"
}}
</output_contract>
""".strip()

        router_runtime_prompt = f"""
{router_state_view}

SUBGRAPH BUDGET:
- max: {subgraph_call_limit}
- used: {subgraph_calls_made}
- remaining: {subgraph_calls_remaining}

AVAILABLE ROUTES:
{route_descriptions}

ROUTE TRACE (up to 15 most recent):
{route_history_view}

Choose the single best next route for this turn.
Important reminders:
- prefer a semantically suitable subgraph when one exists
- advertised subgraph capabilities are disjoint; route explicit capabilities to the subgraph that advertises them
- if the latest subgraph result identifies unsupported remaining step(s), route those step(s) to another explicit subgraph capability or to code_sandbox for concrete computable operations
- react_free is a safety net, not a preferred route
- if the next step can still be delegated meaningfully, choose the subgraph
- encode uncertainty, assumptions, or missing context in the reason
- if you choose a subgraph, provide one single-purpose subgraph_task
""".strip()

        full_history = state.get("internal_messages") or []
        if not isinstance(full_history, list):
            full_history = []

        router_token_budget = ROUTER_TOKEN_BUDGET
        trimmed_history = trim_messages(
            full_history,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=router_token_budget,
            start_on=None,
            end_on=None,
        )

        dropped_count = len(full_history) - len(trimmed_history)
        if dropped_count > 0:
            msg = (
                f"[ROUTER TRIM] Trimmed conversation for router: kept {len(trimmed_history)} messages, "
                f"dropped {dropped_count} older messages to respect ~{router_token_budget} token budget."
            )
            log_to_both(msg, level="warning")

        router_messages = (
            [SystemMessage(content=router_instruction, name="main_graph_router_system")]
            + trimmed_history
            + [HumanMessage(content=router_runtime_prompt, name="main_graph_router_runtime")]
        )

        RouteLiteral = Literal[tuple(allowed_routes)]  # type: ignore
        RouterDecision = create_model(
            "RouterDecision",
            route=(RouteLiteral, ...),
            reason=(str, ...),
            subgraph_task=(str | None, None),
        )

        route = None
        reason = None
        subgraph_task = None

        def _accept_router_payload(payload: Any) -> bool:
            nonlocal route, reason, subgraph_task
            if not isinstance(payload, dict):
                return False
            try:
                decision = RouterDecision.model_validate(payload)
            except Exception:
                return False
            route = getattr(decision, "route", None)
            reason = getattr(decision, "reason", None)
            subgraph_task = getattr(decision, "subgraph_task", None)
            return True

        raw_router_outputs: list[str] = []
        structured_error: Exception | None = None

        for attempt in range(1, 3):
            try:
                decision = self.llm.with_structured_output(RouterDecision).invoke(router_messages)
                route = getattr(decision, "route", None)
                reason = getattr(decision, "reason", None)
                subgraph_task = getattr(decision, "subgraph_task", None)
                break
            except Exception as e:
                if is_ollama_usage_limit_error(e):
                    raise
                structured_error = e
                llm_output = getattr(e, "llm_output", None)
                if isinstance(llm_output, str) and llm_output.strip():
                    raw_router_outputs.append(llm_output.strip())
                logger.warning(
                    "[router_node] structured routing failed on attempt %d/2: %r",
                    attempt,
                    e,
                )

        if not isinstance(route, str) or not route.strip():
            logger.warning(
                "[router_node] structured routing failed after retries: %r; falling back to raw JSON parse",
                structured_error,
            )

            parsed_from_exception = False
            for raw_output in raw_router_outputs:
                if _accept_router_payload(_extract_first_json_object(raw_output)):
                    parsed_from_exception = True
                    break

            if not parsed_from_exception:
                response = self.llm.invoke(router_messages)
                self._log_truncation_to_langgraph(response, where="BiomedAgent.route_node.raw_fallback")
                raw = (getattr(response, "content", str(response)) or "").strip()
                if raw:
                    raw_router_outputs.append(raw)
                _accept_router_payload(_extract_first_json_object(raw))

        if not isinstance(route, str) or not route.strip():
            invalid_output = raw_router_outputs[-1] if raw_router_outputs else ""
            repair_messages = [
                SystemMessage(
                    content=(
                        "You repair router outputs for a biomedical LangGraph router. "
                        "Return only a JSON object matching exactly this schema: "
                        '{"route": string, "reason": string, "subgraph_task": string or null}. '
                        f"The route must be one of: {', '.join(allowed_routes)}. "
                        "Do not add Markdown, prose, or code fences."
                    ),
                    name="main_graph_router_repair_system",
                ),
                HumanMessage(
                    content=(
                        "Convert this invalid router output into valid JSON. "
                        "Preserve the chosen route, reason, and subgraph task when present.\n\n"
                        f"{invalid_output}"
                    ),
                    name="main_graph_router_repair_user",
                ),
            ]
            try:
                repair_response = self.llm.invoke(repair_messages)
                self._log_truncation_to_langgraph(repair_response, where="BiomedAgent.route_node.repair")
                repair_raw = (getattr(repair_response, "content", str(repair_response)) or "").strip()
                if repair_raw:
                    raw_router_outputs.append(repair_raw)
                _accept_router_payload(_extract_first_json_object(repair_raw))
            except Exception as e:
                if is_ollama_usage_limit_error(e):
                    raise
                logger.warning("[router_node] router output repair failed: %r", e)

        if not isinstance(route, str) or not route.strip():
            for raw_output in raw_router_outputs:
                if _accept_router_payload(_salvage_router_decision_from_prose(raw_output, allowed_routes)):
                    logger.warning("[router_node] recovered route from non-JSON router prose")
                    break

        if not isinstance(route, str) or not route.strip():
            route = "react_free"
        if not isinstance(reason, str) or not reason.strip():
            reason = "Fallback routing due to invalid structured output."
        if route in subgraph_keys and not (isinstance(subgraph_task, str) and subgraph_task.strip()):
            subgraph_task = "Perform the single most relevant subgraph-specific task using available attachments and the current plan."

        route_str = str(route).strip()
        if route_str not in allowed_routes:
            route_str = "react_free"

        chosen_route_key = route_str

        if chosen_route_key not in subgraph_keys:
            subgraph_task = None
        else:
            if not (isinstance(subgraph_task, str) and subgraph_task.strip()):
                subgraph_task = "Perform the single most relevant subgraph-specific task using available attachments and the current plan."
            user_paths = _extract_paths_from_messages(list(state.get("messages") or []))
            if user_paths and not _task_mentions_any_path(subgraph_task, user_paths):
                subgraph_task = (
                    f"{subgraph_task}\n\n"
                    "User-provided paths to preserve verbatim for this subtask:\n"
                    + "\n".join(f"- {path}" for path in user_paths)
                )

        if chosen_route_key in subgraph_keys and subgraph_calls_remaining <= 0:
            chosen_route_key = "react_free"
            subgraph_task = None
            reason = (
                f"Subgraph limit reached: {subgraph_calls_made}/{subgraph_call_limit} "
                "subgraph calls already used in this run. Ask for clarification or respond without another subgraph."
            )

        new_subgraph_calls_made = subgraph_calls_made + (1 if chosen_route_key in subgraph_keys else 0)
        new_subgraph_calls_remaining = max(subgraph_call_limit - new_subgraph_calls_made, 0)

        prev_history = state.get("route_history") or []
        if not isinstance(prev_history, list):
            prev_history = []

        new_entry = {
            "route": chosen_route_key,
            "subgraph_task": subgraph_task,
            "subgraph_calls_used": new_subgraph_calls_made,
            "subgraph_calls_remaining": new_subgraph_calls_remaining,
            "timestamp": local_now_iso(),
        }
        route_history_out = prev_history + [new_entry]

        explanation_lines = [f"Routing decision: **{chosen_route_key}**"]
        explanation_lines.append(
            f"Subgraph budget: used {new_subgraph_calls_made}/{subgraph_call_limit}, remaining {new_subgraph_calls_remaining}"
        )
        if isinstance(reason, str) and reason.strip():
            explanation_lines.append(f"Reason: {reason}")
        if chosen_route_key in subgraph_keys:
            explanation_lines.append(f"Subgraph task: {subgraph_task or '(missing)'}")
        else:
            explanation_lines.append("Task: (terminal/control route)")

        explanation_msg = AIMessage(content="\n".join(explanation_lines), name="main_graph_router_explanation")

        return {
            "route_decision": chosen_route_key,
            "router_reason": reason,
            "router_subgraph_task": subgraph_task,
            "route_history": route_history_out,
            "subgraph_call_limit": subgraph_call_limit,
            "subgraph_calls_made": new_subgraph_calls_made,
            "internal_messages": [explanation_msg],
        }

    def todo_planner_node(self, state: BiomedAgentState) -> dict[str, Any]:

        user_msg = getattr(state["messages"][-1], "content", str(state["messages"][-1]))
        attachments, hidden_non_data = self.views.visible_attachments(
            state.get("attachments", []) or [],
            session=state.get("session"),
            workspace_root=state.get("workspace_root") or DEFAULT_WORKSPACE_ROOT,
        )
        attachment_note = (
            f"\nHidden non-data attachments: {hidden_non_data}"
            if hidden_non_data
            else ""
        )

        allowed_routes = list(self.subgraph_by_key.keys()) + ["react_free", "planner", "end"]

        _, TodoPlan = make_todo_models(allowed_routes)

        planner_prompt = f"""
    You create a TODO plan for the agent.
    Return ONLY structured data (no prose).

    Constraints:
    - Every todo MUST have a non-empty 'content'.
    - 'route' must be one of: {allowed_routes}
    - Keep tasks small and single-purpose.
    - Use depends_on to express ordering.
    - If attachments are needed but missing, add a todo asking the user to upload them.

    Input:
    User message: {user_msg}
    Visible attachments: {attachments}{attachment_note}
    """.strip()

        structured = self.llm.with_structured_output(TodoPlan)
        plan = structured.invoke([HumanMessage(content=planner_prompt, name="main_graph_todo_planner_user")])
        plan_payload = plan.model_dump() if isinstance(plan, BaseModel) else plan
        raw_todos = plan_payload.get("todos", []) if isinstance(plan_payload, dict) else []
        todos = [dict(todo) for todo in raw_todos if isinstance(todo, dict)]

        seen = set()
        for i, t in enumerate(todos, start=1):
            if t["id"] in seen or not t["id"].strip():
                t["id"] = f"step_{i:02d}"
            seen.add(t["id"])

            t["content"] = (t.get("content") or "").strip()
            if not t["content"]:
                t["content"] = f"Task {i}"

            t["depends_on"] = [d for d in (t.get("depends_on") or []) if d in seen]

        all_ids = {t["id"] for t in todos}
        for t in todos:
            t["depends_on"] = [d for d in (t.get("depends_on") or []) if d in all_ids]

        tool = self.main_graph_tools_by_name["write_todos"]
        tool_call_id = f"manual:write_todos:{uuid.uuid4().hex[:8]}"
        tool_call = {
            "type": "tool_call",
            "name": "write_todos",
            "args": {"todos": todos},
            "id": tool_call_id,
        }
        command = tool.invoke(tool_call)
        if isinstance(command, Command):
            return _sanitize_command_update(command.update)
        return {}




    def post_tool_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Reserved for main-graph tool post-processing.
        """
        return {}


    def final_summary_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Terminal node that produces the final user-facing response.

        Modes (selected by state.route_decision):
        - final_summary: summarize what has been done in this session
        - final_answer: answer a tool-free/explanatory user question using available context

        It reads a trimmed user-visible conversation snapshot and a detailed state view.
        """

        try:
            convo_text = self._build_router_conversation_text(state, max_chars=6000)
        except Exception as e:
            logger.exception("[final_summary_node] Failed to build conversation text: %s", e)
            convo_text = "Conversation history not available due to an internal error."

        try:
            state_view = self.views.llm_state_view(state)
            if not isinstance(state_view, str):
                state_view = str(state_view)
            original_state_view_len = len(state_view)
            state_view = truncate_middle(
                state_view,
                FINAL_SUMMARY_STATE_VIEW_MAX_CHARS,
                label="state view",
            )
            if len(state_view) < original_state_view_len:
                log_to_both(
                    "[final_summary_node] State view trimmed from "
                    f"{original_state_view_len:,} to {len(state_view):,} characters "
                    "before final LLM call.",
                    level="warning",
                )
        except Exception as e:
            logger.exception("[final_summary_node] Failed to build state view: %s", e)
            state_view = "{}"

        mode = state.get("route_decision") or "final_summary"
        mode = str(mode).strip().lower()

        if mode == "final_answer":
            system_msg = SystemMessage(
                content=(
                    "You are a biomedical AI assistant. Your task is to answer the user's question.\n\n"
                    "GUIDELINES:\n"
                    "- Answer clearly and directly.\n"
                    "- Use the conversation snapshot as the primary source of what the user asked.\n"
                    "- Use the state view only when it is relevant (attachments, session paths, computed results).\n"
                    "- Treat structured tool results, sandbox_result JSON, and artifact contents mentioned in state as authoritative. Copy arrays and numeric values exactly from those sources instead of reconstructing them from prose.\n"
                    "- Before returning requested JSON-like metrics, check simple invariants when possible: totals equal component sums, ratios match numerator/denominator, requested keys are present, and units are consistent with the tool output.\n"
                    "- Do NOT claim to have run tools unless the state view indicates results exist.\n"
                    "- If the user asks about files, prefer attachment session-local paths.\n"
                    "- Use light Markdown formatting when helpful.\n"
                    "- Do NOT include raw JSON unless the user explicitly requests it."
                ),
                name="main_graph_final_answer_system",
            )
        else:
            system_msg = SystemMessage(
                content=(
                    "You are a biomedical AI assistant. Your task is to write the FINAL "
                    "message to the user, summarizing what has been done in this session.\n\n"
                    "GUIDELINES:\n"
                    "- Write directly to the user in clear, concise language.\n"
                    "- Start with a short paragraph that reminds them of their goal.\n"
                    "- Then summarize the main steps performed.\n"
                    "- Highlight important results/findings/file outputs (paths or filenames) in readable bullets.\n"
                    "- Treat structured tool results, sandbox_result JSON, and artifact contents mentioned in state as authoritative. Copy arrays and numeric values exactly from those sources instead of reconstructing them from prose, and mention any obvious consistency issue instead of smoothing it over.\n"
                    "- If the user's latest request or the task output instructions ask to return a JSON object, make the final message exactly one valid JSON object: no Markdown fence, no prose before or after it, and no Python-style dictionaries.\n"
                    "- If there are remaining TODO items or limitations, mention them at the end.\n"
                    "- Use light Markdown, but DO NOT dump raw internal JSON/state keys."
                ),
                name="main_graph_final_summary_system",
            )

        user_prompt = HumanMessage(
            content=(
                "===== CONVERSATION SNAPSHOT (user-visible) =====\n"
                f"{convo_text}\n"
                "===== END SNAPSHOT =====\n\n"
                "===== STATE VIEW (internal, structured) =====\n"
                f"{state_view}\n"
                "===== END STATE VIEW =====\n\n"
                "Write a single final message for the user.\n"
                "If the requested output is a JSON object, the final message must be only that JSON object.\n"
                "Do NOT mention 'state' or 'internal messages'."
            ),
            name="main_graph_final_summary_user",
        )

        internal_messages = state.get("internal_messages") or []
        if not isinstance(internal_messages, list):
            internal_messages = []

        response = self.llm.invoke([system_msg, *internal_messages, user_prompt])
        self._log_truncation_to_langgraph(response, where="BiomedAgent.final_summary_node")

        final_msg = response if isinstance(response, AIMessage) else AIMessage(
            content=getattr(response, "content", str(response)),
            name="main_graph_final_summary_response",
        )

        return {
            "messages": [final_msg],
            "internal_messages": [final_msg],
        }


    def persist_execution_trace_node(self, state: BiomedAgentState) -> dict[str, Any]:
        try:
            persisted = persist_execution_trace(dict(state))
            return {
                "execution_trace": persisted["execution_trace"],
                "execution_trace_file": persisted["execution_trace_file"],
                "execution_trace_index_file": persisted["execution_trace_index_file"],
            }
        except Exception as e:
            logger.exception("[persist_execution_trace_node] Failed to persist execution trace: %s", e)
            return {
                "internal_messages": [
                    SystemMessage(
                        content=f"[RUN TRACE WARNING] Failed to persist execution trace: {e}",
                        name="main_graph_execution_trace_warning",
                    )
                ]
            }

    def inject_system_prompt_node(self, state: BiomedAgentState) -> dict[str, Any]:
        """
        Prepend the agent's system_prompt as a SystemMessage into internal_messages.

        Used by PI_RETRIES_STRONG: the code-sandbox spec-planner reads
        recent internal_messages for context, so injecting the strong constraint
        prompt here ensures the sandbox sees the pipeline ordering rules and
        tool-usage requirements before generating the spec and code.
        """
        constraint_msg = SystemMessage(
            content=self.system_prompt,
            name="main_graph_strong_constraint_injection",
        )
        return {"internal_messages": [constraint_msg]}

    def create_graph(self) -> MainCompiledGraph:
        if self.mode == AgentMode.LLM_ONLY:
            return self._create_llm_only_graph()
        if self.mode in (AgentMode.PYTHON_AGENT_ONESHOT, AgentMode.PYTHON_AGENT_RETRY_STANDARD):
            return self._create_pi_graph()
        if self.mode == AgentMode.PYTHON_AGENT_RETRY_STRICT:
            return self._create_pi_strong_graph()
        if self.mode in (AgentMode.DIRECT_TOOL_AGENT_STRICT, AgentMode.DIRECT_TOOL_AGENT_STANDARD):
            return self._create_single_agent_graph()
        return self._create_full_graph()  # MULTI_AGENT_STANDARD and MULTI_AGENT_STRICT

    def _create_llm_only_graph(self) -> MainCompiledGraph:
        workflow = StateGraph(BiomedAgentState)
        workflow.add_node("start_run",        self.start_run_node)
        workflow.add_node("ensure_session",   self.ensure_session_node)
        workflow.add_node("banner",           self.banner_node)
        workflow.add_node("agent",            self.call_model)
        workflow.add_node("final_summary",    self.final_summary_node)
        workflow.add_node("persist_execution_trace",self.persist_execution_trace_node)

        workflow.add_edge(START,              "start_run")
        workflow.add_edge("start_run",        "ensure_session")
        workflow.add_edge("ensure_session",   "banner")
        workflow.add_edge("banner",           "agent")
        workflow.add_edge("agent",            "final_summary")
        workflow.add_edge("final_summary",    "persist_execution_trace")
        workflow.add_edge("persist_execution_trace", END)
        return workflow.compile(interrupt_before=[])

    def _create_single_agent_graph(self) -> MainCompiledGraph:
        """
        Flat ReAct graph for the single-agent baseline.

        All domain tools are registered directly on the main agent; there are no
        subgraphs and no router.  The loop is:
            start_run → ensure_session → banner → agent ⟳ tools → final_summary → persist_execution_trace

        This mirrors what the full system accomplishes through subgraph routing but
        compresses everything into one agent that must choose tools and ordering itself.
        """
        workflow = StateGraph(BiomedAgentState)
        workflow.add_node("start_run",         self.start_run_node)
        workflow.add_node("ensure_session",    self.ensure_session_node)
        workflow.add_node("banner",            self.banner_node)
        workflow.add_node("agent",             self.call_model)
        workflow.add_node("tools",             self.tool_node)
        workflow.add_node("post_tool",         self.post_tool_node)
        workflow.add_node("final_summary",     self.final_summary_node)
        workflow.add_node("persist_execution_trace", self.persist_execution_trace_node)

        workflow.add_edge(START,               "start_run")
        workflow.add_edge("start_run",         "ensure_session")
        workflow.add_edge("ensure_session",    "banner")
        workflow.add_edge("banner",            "agent")

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools":          "tools",
                "human_approval": "final_summary",   # no interrupt; go straight to summary
                "todo_planner":   "final_summary",   # no planner; go straight to summary
                END:              "final_summary",
            },
        )

        workflow.add_edge("tools",             "post_tool")
        workflow.add_edge("post_tool",         "banner")

        workflow.add_edge("final_summary",     "persist_execution_trace")
        workflow.add_edge("persist_execution_trace", END)

        return workflow.compile(interrupt_before=[])

    def _create_pi_graph(self) -> MainCompiledGraph:
        """
        Linear graph for PI baselines: ensure_session → code_sandbox → final_summary.
        No router, no react_free path — the sandbox is always invoked exactly once
        (pi_single) or once with internal repair cycles (pi_retries).
        """
        assert self.subgraphs, "PI graph requires exactly one subgraph (code_sandbox)"
        sandbox = self.subgraphs[0]
        compiled_sandbox = sandbox.attach()

        workflow = StateGraph(BiomedAgentState)
        workflow.add_node("start_run",         self.start_run_node)
        workflow.add_node("ensure_session",    self.ensure_session_node)
        workflow.add_node("code_sandbox",      compiled_sandbox)
        workflow.add_node("final_summary",     self.final_summary_node)
        workflow.add_node("persist_execution_trace", self.persist_execution_trace_node)

        workflow.add_edge(START,               "start_run")
        workflow.add_edge("start_run",         "ensure_session")
        workflow.add_edge("ensure_session",    "code_sandbox")
        workflow.add_edge("code_sandbox",      "final_summary")
        workflow.add_edge("final_summary",     "persist_execution_trace")
        workflow.add_edge("persist_execution_trace", END)
        return workflow.compile(interrupt_before=[])

    def _create_pi_strong_graph(self) -> MainCompiledGraph:
        """
        Linear graph for the PI_RETRIES_STRONG baseline.

        Identical to _create_pi_graph() except that an inject_system_prompt node
        runs BEFORE the code sandbox.  This injects the strong constraint-rich
        system prompt into internal_messages so the sandbox spec-planner can read
        it as recent execution context, enforcing pipeline ordering and tool-usage
        rules that the weak PI prompt omits entirely.

        Graph:
            start_run → ensure_session → inject_system_prompt
            → code_sandbox → final_summary → persist_execution_trace
        """
        assert self.subgraphs, "PI_RETRIES_STRONG graph requires exactly one subgraph (code_sandbox)"
        sandbox = self.subgraphs[0]
        compiled_sandbox = sandbox.attach()

        workflow = StateGraph(BiomedAgentState)
        workflow.add_node("start_run",             self.start_run_node)
        workflow.add_node("ensure_session",        self.ensure_session_node)
        workflow.add_node("inject_system_prompt",  self.inject_system_prompt_node)
        workflow.add_node("code_sandbox",          compiled_sandbox)
        workflow.add_node("final_summary",         self.final_summary_node)
        workflow.add_node("persist_execution_trace",     self.persist_execution_trace_node)

        workflow.add_edge(START,                   "start_run")
        workflow.add_edge("start_run",             "ensure_session")
        workflow.add_edge("ensure_session",        "inject_system_prompt")
        workflow.add_edge("inject_system_prompt",  "code_sandbox")
        workflow.add_edge("code_sandbox",          "final_summary")
        workflow.add_edge("final_summary",         "persist_execution_trace")
        workflow.add_edge("persist_execution_trace",     END)
        return workflow.compile(interrupt_before=[])

    def _create_full_graph(self) -> MainCompiledGraph:
        workflow = StateGraph(BiomedAgentState)

        workflow.add_node("start_run", self.start_run_node)
        workflow.add_node("ensure_session", self.ensure_session_node)

        workflow.add_node("router", self.route_node)
        workflow.add_node("todo_planner", self.todo_planner_node)
        workflow.add_node("banner", self.banner_node)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("post_tool", self.post_tool_node)
        workflow.add_node("human_approval", self.human_approval_node)
        workflow.add_node("final_summary", self.final_summary_node)
        workflow.add_node("persist_execution_trace", self.persist_execution_trace_node)

        if self.debug:
            workflow.add_node("debug_input1", self.debug_input_node)
            workflow.add_node("debug_input2", self.debug_input_node)

        sub_graph_dict: dict[str, str] = {}

        for sub_graph in self.subgraphs:
            compiled_sub = sub_graph.attach()
            node_name = sub_graph.key
            workflow.add_node(node_name, compiled_sub)

            workflow.add_edge(node_name, "router")
            sub_graph_dict[sub_graph.key] = node_name

        workflow.add_edge(START, "start_run")

        if self.debug:
            workflow.add_edge("start_run", "debug_input1")
            workflow.add_edge("debug_input1", "ensure_session")
        else:
            workflow.add_edge("start_run", "ensure_session")

        if self.debug:
            workflow.add_edge("ensure_session", "debug_input2")
            workflow.add_edge("debug_input2", "router")
        else:
            workflow.add_edge("ensure_session", "router")

        router_destinations: dict[Hashable, str] = {
            "todo_planner": "todo_planner",
            "react_free": "banner",
            "final_summary": "final_summary",
            "final_answer": "final_summary",
        }
        router_destinations.update(sub_graph_dict)
        workflow.add_conditional_edges(
            "router",
            branch_from_router,
            router_destinations,
        )

        workflow.add_edge("todo_planner", "router")

        workflow.add_edge("banner", "agent")

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "human_approval": "human_approval",
                "todo_planner": "todo_planner",
                END: "persist_execution_trace",
            },
        )

        workflow.add_edge("tools", "post_tool")
        workflow.add_edge("post_tool", "banner")

        workflow.add_edge("final_summary", "persist_execution_trace")
        workflow.add_edge("persist_execution_trace", END)

        compiled = workflow.compile(
            interrupt_before=["human_approval"],
        )
        return compiled
