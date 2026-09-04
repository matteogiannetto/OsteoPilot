import logging
import os
import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

TOOL_KIND = "filesystem"
TOOL_NAME = "session_tree"
MAX_RENDERED_CHILDREN_PER_DIR = 200
MIN_SEQUENCE_RUN_LENGTH = 3
NUMBERED_FILE_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>\.[^.]+)$")


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "attachments": [],
        "mode": "single",
        "outputs": {},
        "metadata": {},
    }


class SessionTreeToolInput(BaseModel):
    """Arguments for listing a session directory as a structured tree."""

    model_config = ConfigDict(extra="forbid")

    session_path: str = Field(
        ...,
        description=(
            "INTERNAL: absolute path of the active session directory. The orchestrator "
            "injects and enforces this value, while preserving narrower paths inside "
            "the active session tree."
        ),
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum depth to traverse. Omit for full recursive traversal.",
    )
    include_files: bool = Field(
        default=True,
        description="Whether regular files should be included in the tree output.",
    )
    include_dirs: bool = Field(
        default=True,
        description="Whether directories should be included in the tree output.",
    )
    include_sizes: bool = Field(
        default=False,
        description="Whether file nodes should include their size in megabytes.",
    )
    output_format: str = Field(
        default="text",
        description=(
            "Output shape to return: 'text' for a readable compact tree, or 'struct' "
            "for the nested tree object. Defaults to 'text' to avoid duplicating the tree."
        ),
    )


class SessionTreeTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Inspect a session directory and return its filesystem structure as a compact readable text "
        "tree by default, or as a structured tree when output_format='struct'. Use this tool "
        "iteratively on narrow subdirectories instead of requesting a broad full-session view. "
        "Start with a shallow max_depth, inspect the relevant child path, then call again on that "
        "specific path if more detail is needed. Numbered file runs such as slice_0001.tif through "
        "slice_0100.tif are summarized as ranges, and very large directories are summarized with "
        "file/dir counts and a hint to inspect a narrower path. Use this tool when you need to "
        "explore the contents of a session, "
        "locate relevant files or folders, or understand how session outputs are organized before "
        "calling downstream tools. This tool works at session-filesystem level and reports the current "
        "directory structure without modifying any files. Calls are restricted to the active session "
        "tree: the orchestrator injects the session path unless the supplied path is a specific "
        "location inside that session tree. Directory symlinks inside the session tree can be explored."
    )
    args_schema: type[BaseModel] = SessionTreeToolInput

    def _build_tree(
        self,
        current_path: str,
        depth: int,
        max_depth: int | None,
        include_files: bool,
        include_dirs: bool,
        include_sizes: bool,
        visited_dirs: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if max_depth is not None and depth > max_depth:
            return None

        is_symlink = os.path.islink(current_path)
        is_dir = os.path.isdir(current_path)
        node: dict[str, Any] = {
            "name": os.path.basename(os.path.normpath(current_path)) or current_path,
            "path": current_path,
            "type": "symlink" if is_symlink else ("directory" if is_dir else "file"),
        }

        if is_symlink:
            node["target"] = os.readlink(current_path)

        if not is_dir:
            if include_sizes:
                node["size_mb"] = round(os.path.getsize(current_path) / (1024 * 1024), 6)
            return node

        if visited_dirs is None:
            visited_dirs = set()
        real_dir = os.path.realpath(current_path)
        if real_dir in visited_dirs:
            node["error"] = "cycle_detected"
            node["children"] = []
            return node
        visited_dirs.add(real_dir)

        children: list[dict[str, Any]] = []
        try:
            for entry in sorted(os.listdir(current_path)):
                full_path = os.path.join(current_path, entry)
                if os.path.islink(full_path):
                    if os.path.isdir(full_path):
                        if not include_dirs:
                            continue
                    elif os.path.isfile(full_path):
                        if not include_files:
                            continue
                    else:
                        continue
                elif os.path.isdir(full_path):
                    if not include_dirs:
                        continue
                elif os.path.isfile(full_path):
                    if not include_files:
                        continue
                else:
                    continue

                child = self._build_tree(
                    current_path=full_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                    include_files=include_files,
                    include_dirs=include_dirs,
                    include_sizes=include_sizes,
                    visited_dirs=visited_dirs,
                )
                if child is not None:
                    children.append(child)
        except PermissionError:
            node["error"] = "permission_denied"
        finally:
            visited_dirs.remove(real_dir)

        node["children"] = children
        return node

    def _node_label(self, node: dict[str, Any]) -> str:
        label = node.get("name")
        if not isinstance(label, str):
            raise TypeError("Session tree nodes must have a string name.")
        if node.get("type") == "symlink":
            label = f"{label} -> {node.get('target', '')}"
        if node.get("type") == "file" and "size_mb" in node:
            label = f"{label} ({node['size_mb']} MB)"
        return label

    def _summarize_child_counts(self, children: list[dict[str, Any]]) -> str:
        file_count = sum(1 for child in children if child.get("type") == "file")
        dir_count = sum(
            1
            for child in children
            if child.get("type") in {"directory", "symlink"}
            and child.get("children") is not None
        )
        other_count = max(len(children) - file_count - dir_count, 0)
        parts = [f"{file_count:,} files", f"{dir_count:,} dirs"]
        if other_count:
            parts.append(f"{other_count:,} other entries")
        return f"{', '.join(parts)} omitted; use narrower path to inspect"

    def _numbered_file_parts(self, node: dict[str, Any]) -> tuple[str, int, str, int] | None:
        if node.get("type") != "file":
            return None
        match = NUMBERED_FILE_RE.match(str(node.get("name", "")))
        if not match:
            return None
        number_text = match.group("number")
        return (
            match.group("prefix"),
            int(number_text),
            match.group("suffix"),
            len(number_text),
        )

    def _render_child_entries(self, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        index = 0
        while index < len(children):
            child = children[index]
            parts = self._numbered_file_parts(child)
            if parts is None:
                entries.append({"kind": "node", "node": child})
                index += 1
                continue

            prefix, start_number, suffix, width = parts
            run = [child]
            expected = start_number + 1
            lookahead = index + 1
            while lookahead < len(children):
                next_child = children[lookahead]
                next_parts = self._numbered_file_parts(next_child)
                if next_parts is None:
                    break
                next_prefix, next_number, next_suffix, next_width = next_parts
                if (
                    next_prefix != prefix
                    or next_suffix != suffix
                    or next_width != width
                    or next_number != expected
                ):
                    break
                run.append(next_child)
                expected += 1
                lookahead += 1

            if len(run) >= MIN_SEQUENCE_RUN_LENGTH:
                end_number = start_number + len(run) - 1
                entries.append(
                    {
                        "kind": "summary",
                        "label": (
                            f"{prefix}{start_number:0{width}d}{suffix}.."
                            f"{prefix}{end_number:0{width}d}{suffix} "
                            f"({len(run):,} files)"
                        ),
                    }
                )
                index = lookahead
                continue

            for run_child in run:
                entries.append({"kind": "node", "node": run_child})
            index = lookahead
        return entries

    def _render_tree_text(self, node: dict[str, Any], prefix: str = "", is_last: bool = True) -> list[str]:
        connector = "" if prefix == "" else ("└── " if is_last else "├── ")
        label = self._node_label(node)
        lines = [f"{prefix}{connector}{label}"]

        children = node.get("children", [])
        child_prefix = prefix + ("    " if is_last else "│   ") if prefix != "" else ""
        if len(children) > MAX_RENDERED_CHILDREN_PER_DIR:
            summary_connector = "└── " if child_prefix else ""
            lines.append(
                f"{child_prefix}{summary_connector}"
                f"{self._summarize_child_counts(children)}"
            )
            return lines

        rendered_entries = self._render_child_entries(children)
        for index, entry in enumerate(rendered_entries):
            entry_is_last = index == len(rendered_entries) - 1
            if entry["kind"] == "summary":
                summary_connector = "└── " if entry_is_last else "├── "
                lines.append(f"{child_prefix}{summary_connector}{entry['label']}")
                continue
            child = entry["node"]
            lines.extend(
                self._render_tree_text(
                    child,
                    prefix=child_prefix,
                    is_last=entry_is_last,
                )
            )
        return lines

    def _run(
        self,
        session_path: str,
        max_depth: int | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
        include_sizes: bool = False,
        output_format: str = "text",
    ) -> dict[str, Any]:
        try:
            if not os.path.isabs(session_path):
                return _error_payload(f"session_path must be an absolute path: {session_path}")
            if not os.path.exists(session_path):
                return _error_payload(f"Session path does not exist: {session_path}")
            if not os.path.isdir(session_path):
                return _error_payload(f"session_path is not a directory: {session_path}")
            if not include_files and not include_dirs:
                return _error_payload("At least one of include_files or include_dirs must be True.")
            if output_format not in {"text", "struct"}:
                return _error_payload("output_format must be either 'text' or 'struct'.")

            tree_struct = self._build_tree(
                current_path=session_path,
                depth=0,
                max_depth=max_depth,
                include_files=include_files,
                include_dirs=include_dirs,
                include_sizes=include_sizes,
            )
            if tree_struct is None:
                return _error_payload("Failed to build tree for the requested session path.")

            outputs = (
                {"tree_struct": tree_struct}
                if output_format == "struct"
                else {"tree_text": "\n".join(self._render_tree_text(tree_struct))}
            )
            return {
                "success": True,
                "tool_kind": TOOL_KIND,
                "error": None,
                "attachments": [],
                "mode": "single",
                "outputs": outputs,
                "metadata": {
                    "session_path": session_path,
                    "max_depth": max_depth,
                    "include_files": include_files,
                    "include_dirs": include_dirs,
                    "include_sizes": include_sizes,
                    "output_format": output_format,
                    "max_rendered_children_per_dir": MAX_RENDERED_CHILDREN_PER_DIR,
                    "min_sequence_run_length": MIN_SEQUENCE_RUN_LENGTH,
                },
            }
        except Exception as exc:
            logger.exception("Error while building session tree")
            return _error_payload(str(exc))

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


def session_tree_mcp_call(
    session_path: str,
    max_depth: int | None = None,
    include_files: bool = True,
    include_dirs: bool = True,
    include_sizes: bool = False,
    output_format: str = "text",
) -> dict[str, Any]:
    """Plain MCP-callable function mirroring the BaseTool behavior."""
    tool = SessionTreeTool()
    result = tool.invoke(
        {
            "session_path": session_path,
            "max_depth": max_depth,
            "include_files": include_files,
            "include_dirs": include_dirs,
            "include_sizes": include_sizes,
            "output_format": output_format,
        }
    )
    if not isinstance(result, dict):
        raise TypeError("Session tree tool must return a dictionary payload.")
    return result


def register_mcp_tools(mcp_server: Any) -> Any:
    """Register this tool onto an existing MCP server instance without starting it."""

    # The MCP server exposes its registration decorator dynamically.
    @mcp_server.tool(name=TOOL_NAME)  # type: ignore[untyped-decorator]
    def _registered_session_tree(
        session_path: str,
        max_depth: int | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
        include_sizes: bool = False,
        output_format: str = "text",
    ) -> dict[str, Any]:
        return session_tree_mcp_call(
            session_path=session_path,
            max_depth=max_depth,
            include_files=include_files,
            include_dirs=include_dirs,
            include_sizes=include_sizes,
            output_format=output_format,
        )

    return mcp_server


EXPORTED_TOOLS: dict[str, BaseTool] = {
    TOOL_NAME: SessionTreeTool(),
}
