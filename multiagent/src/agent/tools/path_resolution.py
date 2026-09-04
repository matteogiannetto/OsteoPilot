from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
BIOMED_AGENT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ResolvedInputPath:
    path: Path | None
    searched_paths: list[str]

    @property
    def found(self) -> bool:
        return self.path is not None


def input_path_candidates(
    raw_path: str | Path,
    *,
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> list[Path]:
    requested_path = Path(raw_path).expanduser()
    if requested_path.is_absolute():
        return [requested_path]

    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if session_path:
        add_candidate(Path(session_path).expanduser() / requested_path)
    if workspace_root:
        add_candidate(Path(workspace_root).expanduser() / requested_path)
    add_candidate(BIOMED_AGENT_ROOT / requested_path)
    add_candidate(requested_path)
    return candidates


def resolve_input_path(
    raw_path: str | Path,
    *,
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> ResolvedInputPath:
    """
    Resolve user-facing tool input paths against framework-owned context.

    Relative paths are interpreted from the active session first. This is the
    session contract used by runtime/evaluation sessions, where ``data`` is a
    session-local symlink to the shared project data folder. ``workspace_root``
    is optional context for tools that expose it, and the raw relative path is
    kept as a final fallback for direct/manual calls.
    """
    searched_paths: list[str] = []
    for candidate in input_path_candidates(
        raw_path,
        session_path=session_path,
        workspace_root=workspace_root,
    ):
        resolved = candidate.resolve(strict=False)
        searched_paths.append(str(resolved))
        if resolved.exists():
            return ResolvedInputPath(path=resolved, searched_paths=searched_paths)

    return ResolvedInputPath(path=None, searched_paths=searched_paths)


def missing_input_path_message(label: str, raw_path: str | Path, searched_paths: list[str]) -> str:
    searched = "; ".join(searched_paths) if searched_paths else "(none)"
    return f"{label} not found: {raw_path}. Searched: {searched}"
