# src/agent/tools/session_management.py

import os
from pathlib import Path
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict, NotRequired

from langchain_core.tools import tool, InjectedToolCallId

import ulid  # <- you'll need `ulid-py` in your dependencies

from ..time_utils import local_now_iso


def _workspace_root() -> Path:
    """Return the portable project workspace root."""
    configured_root = os.environ.get("BIOMED_WORKSPACE_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _workspace_data_dir() -> Path:
    return _workspace_root() / "data"


# --- ID helpers --------------------------------------------------------------


def _new_id(prefix: str) -> str:
    """
    Generate a type-prefixed ULID, e.g.:

        sess-01J5Z8K9FQZQJ3X2R9K6DMBE4T
        file-01J5Z8M2A4YQY6H1P9T4KQZ8D2
    """
    return f"{prefix}-{str(ulid.new())}"


# --- Data Structures ---------------------------------------------------------


class SessionInfo(TypedDict):
    """Information about the current user session."""
    id: str          # e.g. "sess-01J5Z8K9FQZQJ3X2R9K6DMBE4T"
    path: str
    timestamp: str
    data_path: NotRequired[str]
    data_read_only: NotRequired[bool]


class AttachmentInfo(TypedDict, total=False):
    """
    Canonical metadata for a file in a session.
    """

    # Core identity
    id: str
    filename: str
    path: str

    # Preserves original absolute path when a file is registered from data or a tool output.
    origin_path: NotRequired[str]

    # File-level metadata
    sha256: str
    size_mb: float
    mimetype: str | None

    # Provenance
    origin: NotRequired[Literal["user", "agent", "tool", "sandbox"]]
    parent: NotRequired[str | None]

    # Semantic / domain hints
    kind: NotRequired[str]
    role: NotRequired[str]
    semantic_target: NotRequired[str]
    file_format: NotRequired[str]
    label_space: NotRequired[str]

    # Free-form metadata
    metadata: NotRequired[dict[str, Any]]
    source: NotRequired[str]
    source_image_path: NotRequired[str]
    source_image_attachment_id: NotRequired[str | None]
    producer_subgraph: NotRequired[str]
    producer_tool: NotRequired[str]
    description: NotRequired[str]
    name: NotRequired[str]
    
    

# --- Session Tools -----------------------------------------------------------


def _link_workspace_data(base_path: str) -> str | None:
    """Expose the workspace data folder inside a newly created session."""
    workspace_data_path = _workspace_data_dir()
    workspace_data = os.fspath(workspace_data_path)
    if not os.path.isdir(workspace_data):
        raise FileNotFoundError(
            "Workspace data directory not found: "
            f"{workspace_data_path}. The graph no longer supports session/input "
            "uploads; place input datasets under <workspace_root>/data or set "
            "BIOMED_WORKSPACE_ROOT to the correct workspace."
        )

    link_path = os.path.join(base_path, "data")
    if os.path.lexists(link_path):
        return link_path

    target = os.path.relpath(workspace_data, start=base_path)
    os.symlink(target, link_path)
    return link_path


@tool(parse_docstring=True)
def create_session(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> SessionInfo:
    """Creates a new, isolated user session and directory structure on the server.
    
    This MUST be the first tool called if no session exists.
    The workspace data directory is required and is exposed as read-only
    session/data. Binary upload into session/input is not supported.

    Args:
        tool_call_id: The ID of the tool call, injected by LangGraph.

    Returns:
        A dictionary containing the new session's ID, base path, and creation timestamp.
    """
    # Use type-prefixed ULID for LLM-friendly, sortable IDs
    workspace_data = _workspace_data_dir()
    if not workspace_data.is_dir():
        raise FileNotFoundError(
            "Workspace data directory not found: "
            f"{workspace_data}. The graph no longer supports session/input "
            "uploads; place input datasets under <workspace_root>/data or set "
            "BIOMED_WORKSPACE_ROOT to the correct workspace."
        )

    session_id = _new_id("sess")
    base_path = os.path.join(os.getcwd(), "sessions", session_id)
    
    # Create session subdirectories. Inputs are exposed only through session/data.
    os.makedirs(os.path.join(base_path, "output"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
    data_path = _link_workspace_data(base_path)

    session: SessionInfo = {
        "id": session_id,
        "path": base_path,
        "timestamp": local_now_iso(),
    }
    if data_path:
        session["data_path"] = data_path
        session["data_read_only"] = True
    return session
