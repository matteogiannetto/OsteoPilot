import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.path_resolution import input_path_candidates

logger = logging.getLogger(__name__)

TOOL_NAME = "load_discovered_files_as_artifacts"
TOOL_KIND = "file"


class LoadDiscoveredFilesAsArtifactsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Existing file paths to register as graph artifacts. Paths may be absolute "
            "or relative to workspace_root/session_path."
        ),
    )
    folder_path: str | None = Field(
        default=None,
        description=(
            "Optional folder whose matching files should be registered as graph artifacts. "
            "Use this after a filesystem discovery step identifies a folder of relevant files."
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
    extensions: list[str] | None = Field(
        default=None,
        description=(
            "Optional file extensions to include when folder_path is used, such as "
            "['.tif', '.tiff', '.nii.gz']. Explicit paths are not filtered."
        ),
    )
    recursive: bool = Field(
        default=False,
        description="Whether to recursively collect files below folder_path.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum recursion depth for folder_path. 0 means direct files only.",
    )
    max_files: int = Field(
        default=120,
        ge=1,
        le=10000,
        description="Maximum number of relevant files to register from all inputs.",
    )
    artifact_kind: str = Field(
        default=TOOL_KIND,
        description=(
            "Semantic kind to assign to the registered artifacts, e.g. file, image, "
            "tiff_slice, nifti_volume, segmentation, or metadata."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Optional description to attach to every registered artifact.",
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


def _resolve_path(raw_path: str, session_path: str, workspace_root: str | None) -> Path:
    candidates = input_path_candidates(
        raw_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _normalize_extensions(extensions: list[str] | None) -> tuple[set[str], set[str]] | None:
    if not extensions:
        return None

    suffixes: set[str] = set()
    full_suffixes: set[str] = set()
    for ext in extensions:
        if not isinstance(ext, str) or not ext.strip():
            continue
        normalized = ext.strip().lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        suffixes.add(normalized)
        full_suffixes.add(normalized)

    return (suffixes, full_suffixes) if suffixes else None


def _matches_extension(path: Path, extension_filter: tuple[set[str], set[str]] | None) -> bool:
    if extension_filter is None:
        return True
    suffixes, full_suffixes = extension_filter
    simple_suffix = path.suffix.lower()
    combined_suffix = "".join(path.suffixes).lower()
    return simple_suffix in suffixes or combined_suffix in full_suffixes


def _iter_folder_files(
    folder: Path,
    *,
    recursive: bool,
    max_depth: int | None,
    extension_filter: tuple[set[str], set[str]] | None,
) -> Iterator[Path]:
    stack: list[tuple[Path, int]] = [(folder, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
        except PermissionError:
            continue

        for child in children:
            if child.is_file() and _matches_extension(child, extension_filter):
                yield child.resolve()
                continue

            if not recursive or not child.is_dir():
                continue
            if max_depth is not None and depth >= max_depth:
                continue
            stack.append((child, depth + 1))


class LoadDiscoveredFilesAsArtifactsTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Register relevant existing files discovered by filesystem inspection as graph artifacts "
        "so downstream subgraphs can see them as attachments. Use this after listing a folder "
        "or locating specific files that are significant for later tools; do not use it to "
        "register entire datasets just to make them accessible. It records file paths and "
        "lightweight metadata only; it does not copy files, read file contents, inspect image "
        "pixels, convert formats, or modify data."
    )
    args_schema: type[BaseModel] = LoadDiscoveredFilesAsArtifactsArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        try:
            args = LoadDiscoveredFilesAsArtifactsArgs(**kwargs)
            if not args.paths and not args.folder_path:
                return _error_payload("Provide at least one explicit file path or a folder_path.")

            extension_filter = _normalize_extensions(args.extensions)
            unique_files: list[Path] = []
            seen: set[str] = set()
            skipped: list[dict[str, str]] = []
            truncated = False

            def add_candidate(path: Path) -> bool:
                key = str(path)
                if key in seen:
                    return True
                if len(unique_files) >= args.max_files:
                    return False
                seen.add(key)
                unique_files.append(path)
                return True

            for item in args.paths:
                path = _resolve_path(item, args.session_path, args.workspace_root)
                if not path.exists():
                    skipped.append({"path": str(path), "reason": "not_found"})
                    continue
                if not path.is_file():
                    skipped.append({"path": str(path), "reason": "not_a_file"})
                    continue
                if not add_candidate(path):
                    truncated = True

            resolved_folder: Path | None = None
            if args.folder_path:
                resolved_folder = _resolve_path(args.folder_path, args.session_path, args.workspace_root)
                if not resolved_folder.exists():
                    skipped.append({"path": str(resolved_folder), "reason": "folder_not_found"})
                elif not resolved_folder.is_dir():
                    skipped.append({"path": str(resolved_folder), "reason": "not_a_folder"})
                else:
                    for path in _iter_folder_files(
                        resolved_folder,
                        recursive=args.recursive,
                        max_depth=args.max_depth,
                        extension_filter=extension_filter,
                    ):
                        if not add_candidate(path):
                            truncated = True
                            break

            if not unique_files:
                return {
                    "success": False,
                    "tool_kind": TOOL_KIND,
                    "error": "No existing files matched the requested inputs.",
                    "attachments": [],
                    "outputs": {
                        "registered_paths": [],
                        "skipped": skipped,
                    },
                    "metadata": {
                        "folder_path": str(resolved_folder) if resolved_folder else None,
                        "extensions": args.extensions,
                        "recursive": args.recursive,
                        "max_depth": args.max_depth,
                        "max_files": args.max_files,
                    },
                }

            description = args.description or "Relevant existing file registered for downstream use."
            attachments = [
                {
                    "filename": path.name,
                    "path": str(path),
                    "kind": args.artifact_kind or TOOL_KIND,
                    "description": description,
                    "origin": "user",
                    "size_mb": path.stat().st_size / (1024 * 1024),
                }
                for path in unique_files
            ]

            return {
                "success": True,
                "tool_kind": args.artifact_kind or TOOL_KIND,
                "error": None,
                "attachments": attachments,
                "outputs": {
                    "registered_paths": [str(path) for path in unique_files],
                    "n_registered": len(unique_files),
                    "skipped": skipped,
                    "truncated": truncated,
                },
                "metadata": {
                    "folder_path": str(resolved_folder) if resolved_folder else None,
                    "extensions": args.extensions,
                    "recursive": args.recursive,
                    "max_depth": args.max_depth,
                    "max_files": args.max_files,
                    "artifact_kind": args.artifact_kind,
                },
            }
        except Exception as exc:
            logger.exception("Failed to register discovered files as artifacts")
            return _error_payload(f"{type(exc).__name__}: {exc}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


def load_discovered_files_as_artifacts_mcp_call(
    paths: list[str] | None,
    session_path: str,
    folder_path: str | None = None,
    workspace_root: str | None = None,
    extensions: list[str] | None = None,
    recursive: bool = False,
    max_depth: int | None = None,
    max_files: int = 120,
    artifact_kind: str = TOOL_KIND,
    description: str | None = None,
) -> dict[str, Any]:
    tool = LoadDiscoveredFilesAsArtifactsTool()
    result = tool.invoke(
        {
            "paths": paths or [],
            "folder_path": folder_path,
            "session_path": session_path,
            "workspace_root": workspace_root,
            "extensions": extensions,
            "recursive": recursive,
            "max_depth": max_depth,
            "max_files": max_files,
            "artifact_kind": artifact_kind,
            "description": description,
        }
    )
    if not isinstance(result, dict):
        raise TypeError("Artifact loader must return a dictionary payload.")
    return result


EXPORTED_TOOLS: dict[str, BaseTool] = {
    TOOL_NAME: LoadDiscoveredFilesAsArtifactsTool(),
}
