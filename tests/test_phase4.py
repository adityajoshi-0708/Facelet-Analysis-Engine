"""Phase 4 tests — embedding client."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.models.embedding_client import EmbeddingClient

_client = EmbeddingClient()


def test_single_encode_shape():
    emb = _client.encode("Risk Taking")
    assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"
    print("✓ single_encode_shape passed")


def test_single_encode_normalised():
    emb = _client.encode("Risk Taking")
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-5, f"Embedding not normalised: norm={norm}"
    print("✓ single_encode_normalised passed")


def test_batch_encode_shape():
    batch = _client.encode_batch(["Risk Taking", "Compassion", "Honesty"])
    assert batch.shape == (3, 384), f"Expected shape (3, 384), got {batch.shape}"
    print("✓ batch_encode_shape passed")


def test_semantic_similarity_ordering():
    risk_emb = _client.encode("Risk Taking")
    adventure_emb = _client.encode("Adventure Seeking")
    compassion_emb = _client.encode("Compassion")

    sim_related = float(np.dot(risk_emb, adventure_emb))
    sim_unrelated = float(np.dot(risk_emb, compassion_emb))

    assert sim_related > sim_unrelated, (
        f"Related pair similarity ({sim_related:.4f}) must exceed "
        f"unrelated pair ({sim_unrelated:.4f})"
    )
    print("✓ semantic_similarity_ordering passed")


if __name__ == "__main__":
    test_single_encode_shape()
    test_single_encode_normalised()
    test_batch_encode_shape()
    test_semantic_similarity_ordering()
    print("✅ All Phase 4 tests passed")