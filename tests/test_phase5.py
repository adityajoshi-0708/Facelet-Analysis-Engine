"""Phase 5 tests — FAISS index build and search."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data_pipeline.clean import clean_facets
from src.data_pipeline.enrich import enrich_facets
from src.models.embedding_client import EmbeddingClient
from src.retrieval.index_builder import IndexBuilder

SEED_FACETS = [
    "Risk Taking", "Compassion", "Honesty", "Naivety", "Adventure Seeking",
    "Assertiveness", "Empathy", "Statistical Reasoning", "Compassion Fatigue",
    "Democratic Leadership", "Moroseness", "Common Sense", "Kindness", "Warmth",
    "Courage", "Authenticity", "Integrity", "Pessimism", "Curiosity", "Gullibility",
]

_clean_df = clean_facets(SEED_FACETS)
_enriched_df = enrich_facets(_clean_df)

_client = EmbeddingClient()
_builder = IndexBuilder(client=_client)
_index = _builder.build(_enriched_df)


def test_index_built():
    assert _index is not None
    print("✓ index_built passed")


def test_index_total_matches_facets():
    assert _index.ntotal == len(_enriched_df), (
        f"Index has {_index.ntotal} vectors but DataFrame has {len(_enriched_df)} rows"
    )
    print("✓ index_total_matches_facets passed")


def test_search_returns_k_results():
    query_vec = _client.encode("I quit my job to start a startup").reshape(1, -1)
    D, I = _index.search(query_vec, k=5)
    assert len(I[0]) == 5, f"Expected 5 results, got {len(I[0])}"
    print("✓ search_returns_k_results passed")


def test_metadata_has_top_result():
    metadata = json.loads(Path("data/index/facet_metadata.json").read_text())
    query_vec = _client.encode("I quit my job to start a startup").reshape(1, -1)
    D, I = _index.search(query_vec, k=5)
    top_id = str(I[0][0])
    assert top_id in metadata, f"Top result ID {top_id} not found in metadata"
    top_facet = metadata[top_id]["facet_name"]
    print(f"  → Top result for startup query: {top_facet}")
    print("✓ metadata_has_top_result passed")


def test_index_files_saved():
    assert Path("data/index/facets.index").exists(), "facets.index not found"
    assert Path("data/index/facet_metadata.json").exists(), "facet_metadata.json not found"
    print("✓ index_files_saved passed")


if __name__ == "__main__":
    test_index_built()
    test_index_total_matches_facets()
    test_search_returns_k_results()
    test_metadata_has_top_result()
    test_index_files_saved()
    print("✅ All Phase 5 tests passed")