from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.path_resolution import input_path_candidates

logger = logging.getLogger(__name__)

TOOL_NAME = "folder_listing_with_sizes"
TOOL_KIND = "folder_listing"


class FolderListingWithSizesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field(
        ...,
        description=(
            "Folder to inspect. May be an absolute path or a path relative to "
            "workspace_root or session_path."
        ),
    )
    session_path: str = Field(
        ...,
        description="INTERNAL: absolute path to the active session folder. Injected by the orchestrator.",
    )
    workspace_root: str | None = Field(
        default=None,
        description="INTERNAL: optional read-only base directory for generic relative paths.",
    )
    recursive: bool = Field(
        default=False,
        description="Whether to recursively list files and directories below folder_path.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum recursion depth. 0 means only folder_path entries.",
    )
    max_entries: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of entries to return.",
    )
    include_dirs: bool = Field(
        default=True,
        description="Whether directory entries should be included.",
    )
    include_files: bool = Field(
        default=True,
        description="Whether file entries should be included.",
    )
    extensions: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of file extensions to include, such as ['.tif', '.tiff']. "
            "Directories are not filtered by extension."
        ),
    )


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "attachments": [],
        "outputs": {},
        "metadata": {},
    }


def _resolve_folder(
    folder_path: str,
    session_path: str,
    workspace_root: str | None,
) -> Path:
    candidates = input_path_candidates(
        folder_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _normalize_extensions(extensions: list[str] | None) -> set[str] | None:
    if not extensions:
        return None
    normalized: set[str] = set()
    for ext in extensions:
        if not isinstance(ext, str) or not ext.strip():
            continue
        item = ext.strip().lower()
        normalized.add(item if item.startswith(".") else f".{item}")
    return normalized or None


class FolderListingWithSizesCore:
    def execute(self, args: FolderListingWithSizesArgs) -> dict[str, Any]:
        try:
            folder = _resolve_folder(args.folder_path, args.session_path, args.workspace_root)
            if not folder.exists():
                return _error_payload(f"Folder not found: {folder}")
            if not folder.is_dir():
                return _error_payload(f"Path is not a folder: {folder}")
            if not args.include_dirs and not args.include_files:
                return _error_payload("At least one of include_dirs or include_files must be True.")

            extension_filter = _normalize_extensions(args.extensions)
            entries: list[dict[str, Any]] = []
            total_files = 0
            total_dirs = 0
            total_size_bytes = 0
            truncated = False

            def visit(current: Path, depth: int) -> None:
                nonlocal total_files, total_dirs, total_size_bytes, truncated
                if len(entries) >= args.max_entries:
                    truncated = True
                    return
                if args.max_depth is not None and depth > args.max_depth:
                    return

                try:
                    children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                except PermissionError:
                    entries.append(
                        {
                            "name": current.name,
                            "path": str(current),
                            "relative_path": str(current.relative_to(folder)),
                            "type": "directory",
                            "error": "permission_denied",
                        }
                    )
                    return

                for child in children:
                    if len(entries) >= args.max_entries:
                        truncated = True
                        return

                    if child.is_dir():
                        total_dirs += 1
                        if args.include_dirs:
                            entries.append(
                                {
                                    "name": child.name,
                                    "path": str(child),
                                    "relative_path": str(child.relative_to(folder)),
                                    "type": "directory",
                                }
                            )
                        if args.recursive:
                            visit(child, depth + 1)
                        continue

                    if not child.is_file():
                        continue

                    suffix = child.suffix.lower()
                    if extension_filter and suffix not in extension_filter:
                        continue

                    total_files += 1
                    size_bytes = child.stat().st_size
                    total_size_bytes += size_bytes
                    if args.include_files:
                        entries.append(
                            {
                                "name": child.name,
                                "path": str(child),
                                "relative_path": str(child.relative_to(folder)),
                                "type": "file",
                                "extension": suffix,
                                "size_bytes": size_bytes,
                                "size_mb": round(size_bytes / (1024 * 1024), 6),
                            }
                        )

            visit(folder, 0)

            return {
                "success": True,
                "tool_kind": TOOL_KIND,
                "error": None,
                "attachments": [],
                "outputs": {
                    "folder_path": str(folder),
                    "entries": entries,
                    "n_entries_returned": len(entries),
                    "n_files_seen": total_files,
                    "n_dirs_seen": total_dirs,
                    "total_file_size_bytes": total_size_bytes,
                    "total_file_size_mb": round(total_size_bytes / (1024 * 1024), 6),
                    "truncated": truncated,
                },
                "metadata": {
                    "requested_folder_path": args.folder_path,
                    "recursive": args.recursive,
                    "max_depth": args.max_depth,
                    "max_entries": args.max_entries,
                    "include_dirs": args.include_dirs,
                    "include_files": args.include_files,
                    "extensions": sorted(extension_filter) if extension_filter else None,
                },
            }
        except Exception as exc:
            logger.exception("Folder listing failed")
            return _error_payload(f"{type(exc).__name__}: {exc}")


class FolderListingWithSizesTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Lists the contents of a folder and reports file sizes. Use this when a task "
        "provides a directory but the files inside it are not known yet, or when the "
        "user asks for list-like output with sizes. Accepts absolute paths or paths "
        "relative to workspace_root/session_path. Returns structured entries, counts, "
        "and total file size. It does not read image pixels, convert volumes, modify "
        "files, segment data, or perform quantitative biological analysis."
    )
    args_schema: type[BaseModel] = FolderListingWithSizesArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        args = FolderListingWithSizesArgs(**kwargs)
        return FolderListingWithSizesCore().execute(args)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


FolderListingWithSizesArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
FolderListingWithSizesTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "FolderListingWithSizesArgs": FolderListingWithSizesArgs,
    }
)


def folder_listing_with_sizes_mcp_call(
    folder_path: str,
    session_path: str,
    workspace_root: str | None = None,
    recursive: bool = False,
    max_depth: int | None = None,
    max_entries: int = 500,
    include_dirs: bool = True,
    include_files: bool = True,
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    args = FolderListingWithSizesArgs(
        folder_path=folder_path,
        session_path=session_path,
        workspace_root=workspace_root,
        recursive=recursive,
        max_depth=max_depth,
        max_entries=max_entries,
        include_dirs=include_dirs,
        include_files=include_files,
        extensions=extensions,
    )
    return FolderListingWithSizesCore().execute(args)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    TOOL_NAME: FolderListingWithSizesTool(),
}
