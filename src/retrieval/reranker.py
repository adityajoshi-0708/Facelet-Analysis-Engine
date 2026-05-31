"""
Phase 9 — Knowledge Graph Expansion
After retrieval, expand each result's facet with its graph neighbours so that
"Compassion" also surfaces "Empathy", "Kindness" etc.

The expander de-duplicates, re-ranks by combining original retrieval score with
a small decay factor for graph-expanded neighbours, and preserves top_k.
"""

from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..knowledge_graph.graph_loader import load_graph, get_neighbors

logger = get_logger(__name__)

# Score decay applied to neighbours introduced by graph expansion
_EXPANSION_DECAY = 0.85


class GraphExpander:
    """
    Expand a list of retrieved facets using the knowledge graph.

    Args:
        depth:          BFS depth for neighbour expansion (1 or 2).
        top_k:          Maximum size of the expanded result list.
        decay:          Score multiplier applied to graph-expanded facets.
    """

    def __init__(
        self,
        depth: Optional[int] = None,
        top_k: Optional[int] = None,
        decay: float = _EXPANSION_DECAY,
    ):
        cfg = load_config()
        ret_cfg = cfg.get("retrieval", {})

        self._depth: int = depth or ret_cfg.get("graph_expansion_depth", 2)
        self._top_k: int = top_k or ret_cfg.get("top_k", 40)
        self._decay: float = decay
        self._graph: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_graph(self):
        if self._graph is None:
            self._graph = load_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, results: List[Dict]) -> List[Dict]:
        """
        Expand retrieved facets with knowledge-graph neighbours.

        Args:
            results: Ranked list from any retriever — each dict must have at
                     minimum keys ``facet_name``, ``facet_id``, ``score``.

        Returns:
            Expanded and re-ranked list (length ≤ top_k).
            Expanded entries carry an additional key ``"expanded": True``.
        """
        if not results:
            return results

        self._ensure_graph()

        # Index existing results by facet_name for fast lookup
        seen: Dict[str, Dict] = {r["facet_name"]: r for r in results}

        for result in list(results):  # iterate original list
            neighbours = get_neighbors(
                result["facet_name"], depth=self._depth, graph=self._graph
            )
            for neighbour in neighbours:
                if neighbour in seen:
                    continue  # already in results (with a better score)
                # Build a synthetic entry for the expanded neighbour
                seen[neighbour] = {
                    "facet_id":   neighbour.lower().replace(" ", "_"),
                    "facet_name": neighbour,
                    "category":   result.get("category", ""),
                    "score":      result["score"] * self._decay,
                    "rank":       None,  # re-ranked below
                    "expanded":   True,
                }

        # Re-rank all entries by score descending
        all_entries = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        all_entries = all_entries[: self._top_k]

        # Assign final ranks
        for rank, entry in enumerate(all_entries, start=1):
            entry["rank"] = rank

        original_count  = len(results)
        expanded_count  = sum(1 for e in all_entries if e.get("expanded"))
        logger.debug(
            f"GraphExpander: {original_count} → {len(all_entries)} results "
            f"({expanded_count} from graph expansion, depth={self._depth})"
        )
        return all_entries