"""
Phase 3 — Knowledge Graph Loader
Loads the facet graph JSON and provides BFS-based neighbor expansion.
"""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache so the file is only read once per process
_GRAPH_CACHE: Optional[Dict[str, List[str]]] = None


def load_graph(graph_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load the facet adjacency graph.

    Args:
        graph_path: Optional explicit path; defaults to the configured location.

    Returns:
        Adjacency dict: { "Compassion": ["Empathy", "Kindness", ...], ... }
    """
    global _GRAPH_CACHE

    if _GRAPH_CACHE is not None and graph_path is None:
        return _GRAPH_CACHE

    if graph_path is None:
        cfg = load_config()
        graph_path = Path(cfg["data"]["knowledge_graph_dir"]) / "facet_graph.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {graph_path}")

    graph: Dict[str, List[str]] = json.loads(graph_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded knowledge graph with {len(graph)} nodes from {graph_path}")

    if graph_path is None:  # only cache default path
        _GRAPH_CACHE = graph

    return graph


def get_neighbors(
    facet_name: str,
    depth: int = 1,
    graph: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Return all facets reachable from *facet_name* within *depth* hops.

    Uses BFS so depth 1 = direct neighbours, depth 2 = neighbours of neighbours, etc.
    The starting node itself is excluded from the result.

    Args:
        facet_name: Name of the seed facet.
        depth: Maximum number of hops (1 or 2 are most common).
        graph: Optional pre-loaded adjacency dict; loaded from disk if None.

    Returns:
        Sorted list of reachable facet names (excluding the seed).
    """
    if depth < 1:
        return []

    if graph is None:
        graph = load_graph()

    visited = {facet_name}
    queue: deque = deque()

    # Seed the queue with direct neighbours
    for neighbour in graph.get(facet_name, []):
        if neighbour not in visited:
            visited.add(neighbour)
            queue.append((neighbour, 1))

    while queue:
        node, current_depth = queue.popleft()
        if current_depth < depth:
            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, current_depth + 1))

    result = sorted(visited - {facet_name})
    return result