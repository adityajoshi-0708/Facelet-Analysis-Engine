"""
Phase 3 — Knowledge Graph Builder
Reads enriched facets CSV and builds a bidirectional adjacency dict from the
related_facets and synonyms columns.  Saves to data/knowledge_graph/facet_graph.json.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_graph(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a facet relationship graph from an enriched DataFrame.

    Edges are added for:
    1. related_facets (explicit relationships defined in the data)
    2. synonyms (treated as equivalent / near-equivalent nodes)
    Both directions are stored (bidirectional graph).

    Args:
        df: Enriched DataFrame with columns facet_name, related_facets, synonyms.

    Returns:
        Adjacency dict: { "Compassion": ["Empathy", "Kindness", ...], ... }
    """
    cfg = load_config()
    graph_dir = Path(cfg["data"]["knowledge_graph_dir"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph: Dict[str, List[str]] = defaultdict(set)  # type: ignore[assignment]

    for _, row in df.iterrows():
        node: str = row["facet_name"]

        # Parse related_facets column (stored as JSON string)
        try:
            related: List[str] = json.loads(row.get("related_facets", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            related = []

        # Parse synonyms column
        try:
            synonyms: List[str] = json.loads(row.get("synonyms", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            synonyms = []

        # Add edges for related facets (bidirectional)
        for rel in related:
            if rel and rel != node:
                graph[node].add(rel)
                graph[rel].add(node)

        # Add edges for synonyms (bidirectional)
        for syn in synonyms:
            if syn and syn != node:
                # Capitalise synonym to match facet naming convention
                syn_cap = syn.title()
                graph[node].add(syn_cap)
                graph[syn_cap].add(node)

    # Convert sets to sorted lists for deterministic output
    adjacency: Dict[str, List[str]] = {
        node: sorted(neighbors)
        for node, neighbors in graph.items()
    }

    out_path = graph_dir / "facet_graph.json"
    out_path.write_text(json.dumps(adjacency, indent=2, ensure_ascii=False))
    logger.info(f"Saved knowledge graph to {out_path} ({len(adjacency)} nodes)")

    return adjacency