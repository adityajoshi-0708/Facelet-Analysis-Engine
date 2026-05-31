"""Phase 3 tests — knowledge graph build and load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data_pipeline.clean import clean_facets
from src.data_pipeline.enrich import enrich_facets
from src.knowledge_graph.graph_builder import build_graph
from src.knowledge_graph.graph_loader import load_graph, get_neighbors

SEED_FACETS = [
    "Risk Taking", "Compassion", "Honesty", "Naivety", "Adventure Seeking",
    "Assertiveness", "Empathy", "Statistical Reasoning", "Compassion Fatigue",
    "Democratic Leadership", "Moroseness", "Common Sense", "Kindness", "Warmth",
    "Courage", "Authenticity", "Integrity", "Pessimism", "Curiosity", "Gullibility",
]

# Build enriched CSV + graph once for all tests
_clean_df = clean_facets(SEED_FACETS)
_enriched_df = enrich_facets(_clean_df)
_graph = build_graph(_enriched_df)


def test_graph_built():
    assert isinstance(_graph, dict), "Graph must be a dict"
    assert len(_graph) > 0, "Graph must not be empty"
    print("✓ graph_built passed")


def test_graph_saved():
    assert Path("data/knowledge_graph/facet_graph.json").exists(), (
        "data/knowledge_graph/facet_graph.json not found"
    )
    print("✓ graph_saved passed")


def test_load_graph():
    graph = load_graph()
    assert isinstance(graph, dict)
    assert len(graph) > 0
    print("✓ load_graph passed")


def test_compassion_has_neighbors():
    neighbors = get_neighbors("Compassion", depth=1, graph=_graph)
    assert len(neighbors) > 0, "Compassion must have at least one neighbor"
    print("✓ compassion_has_neighbors passed")


def test_depth_2_more_than_depth_1():
    d1 = get_neighbors("Compassion", depth=1, graph=_graph)
    d2 = get_neighbors("Compassion", depth=2, graph=_graph)
    assert len(d2) >= len(d1), (
        f"Depth-2 neighbors ({len(d2)}) must be >= depth-1 ({len(d1)})"
    )
    print("✓ depth_2_more_than_depth_1 passed")


if __name__ == "__main__":
    test_graph_built()
    test_graph_saved()
    test_load_graph()
    test_compassion_has_neighbors()
    test_depth_2_more_than_depth_1()
    print("✅ All Phase 3 tests passed")