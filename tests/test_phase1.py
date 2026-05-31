import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_section
from src.utils.logger import get_logger
from src.utils.types import Facet, FacetScore, EvaluationResult


def test_load_config():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["top_k"] == 20
    assert "model" in cfg
    assert cfg["model"]["name"] == "qwen3-8b"
    print("✓ load_config passed")


def test_config_has_sub_sections():
    cfg = load_config()
    assert "retrieval" in cfg
    assert "scoring" in cfg
    assert "models" in cfg
    assert "logging" in cfg
    print("✓ sub-config merge passed")


def test_get_section():
    retrieval = get_section("retrieval")
    assert retrieval["dense_weight"] == 0.7
    assert retrieval["bm25_weight"] == 0.3
    print("✓ get_section passed")


def test_logger():
    logger = get_logger("test")
    logger.info("Logger working")
    assert logger is not None
    print("✓ logger passed")


def test_types():
    facet = Facet(
        facet_id="risk_taking",
        facet_name="Risk Taking",
        category="personality",
    )
    assert facet.facet_id == "risk_taking"

    score = FacetScore(
        facet_id="risk_taking",
        facet_name="Risk Taking",
        score=5,
        confidence=0.88,
        rationale="Strong willingness to accept uncertainty.",
    )
    assert 1 <= score.score <= 5
    assert 0.0 <= score.confidence <= 1.0
    print("✓ types passed")


if __name__ == "__main__":
    test_load_config()
    test_config_has_sub_sections()
    test_get_section()
    test_logger()
    test_types()
    print("\n✅ All Phase 1 tests passed")