# agent/subgraphs/subgraph_contract.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph

class SubgraphModule(ABC):
    """
    Contract every subgraph implements.
    """
    # Unique key the router/planner will use, e.g. "imaging_preprocessing"
    key: str

    # Human title (LLM-friendly text)
    title: str
    # Optional: tags to help planners/routers (fast string matches)
    tags: list[str] = []

    @abstractmethod
    def attach(self) -> CompiledStateGraph[Any, None, Any, Any]:
        """
        Build and return this module's compiled subgraph.

        The root graph owns integration: it registers the returned graph as a
        node while each module owns its internal nodes and edges.
        """
        ...

    @abstractmethod
    def description(self) -> str:
        """Return the short description used by the router and capability prompts."""
        ...

    def capability_text(self) -> str:
        """
        Human-friendly summary used directly in prompts.
        You can derive it from the JSON to avoid drift.
        """
        j = self.capability_json()
        lines = [f"- {self.title} ({self.key}): {self.description()}"]
        # Prefer short, bulleted capabilities
        caps = j.get("capabilities", [])
        if caps:
            for c in caps:
                lines.append(f"  • {c}")
        limits = j.get("limits", {})
        if limits:
            lines.append(f"  • Limits: {limits}")
        return "\n".join(lines)

    @abstractmethod
    def capability_json(self) -> dict[str, Any]:
        """Return the machine-readable capability description used to build prompts."""
        ...

    def can_route(self, user_text: str) -> Literal["yes", "no"] | None:
        """
        Optional fast pre-router hint. Return "yes"/"no" or None if undecided.
        Keep it simple (regex/keywords) to avoid an LLM call when obvious.
        """
        return None

    def tool_specs(self) -> list[dict[str, Any]]:
        """
        Return a list of tool specs:
        [{"name": "preprocess_microscopy_tiff",
          "description": "Preprocess microscopy .tif/.tiff",
          "args": {"tiff_path":"str","session_path":"str","output_prefix":"str"}}]
        """
        return []
