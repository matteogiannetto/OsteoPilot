from typing import Annotated, Literal
import hashlib
import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from ..graph_state import TodoItem

def compute_plan_hash(todos: list[TodoItem]) -> str:
    plan_content = json.dumps([t.content for t in todos], sort_keys=True)
    return hashlib.sha256(plan_content.encode("utf-8")).hexdigest()
@tool(parse_docstring=True)
def write_todos(
    todos: list[TodoItem],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Literal["human_approval"]]:

    """Create or replace the TODO list used to plan the task.
    
    Args:
        todos: Tasks with their content and status.
        tool_call_id: Identifier of the tool call.
    """
    new_hash = compute_plan_hash(todos)
    
    return Command(
        update={
            "todos": todos,
            "current_plan_hash": new_hash,
            "plan_approved": False,
            "internal_messages": [
                ToolMessage(
                    "TODO list aggiornato. Approvazione richiesta.",
                    tool_call_id=tool_call_id,
                    name="main_graph_tool_write_todos",
                )
            ],
        },
        goto="human_approval"
    )

@tool(parse_docstring=True)
def read_todos(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list.
    
    Args:
        tool_call_id: Identifier of the tool call.
    """
    return "Use write_todos to create the plan before reading it."
