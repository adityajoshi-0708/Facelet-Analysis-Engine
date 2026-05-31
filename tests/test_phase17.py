"""
tests/test_phase17_evaluation.py — Phase 17A: Evaluation Framework Tests

Run from project root:
    python tests/test_phase17_evaluation.py

All tests use mocked/synthetic data — no LLM calls, no real retrieval.
Each test_* function is standalone (no pytest fixtures required).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PASSED: List[str] = []
_FAILED: List[str] = []


def _ok(name: str):
    print(f"✓ {name} passed")
    _PASSED.append(name)


def _fail(name: str, reason: str):
    print(f"✗ {name} FAILED: {reason}")
    _FAILED.append(name)


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic data
# ---------------------------------------------------------------------------

_RETRIEVAL_DETAILS = [
    {
        "cid": "p01", "title": "Startup", "category": "personality",
        "difficulty": "easy",
        "expected_facets": ["Risktaking", "Determinedness"],
        "retrieved_facets": ["Risktaking", "Determinedness", "Curiosity"],
        "recall_at_5": 1.0, "recall_at_10": 1.0, "recall_at_20": 1.0,
        "rr": 1.0, "expected_rank": 1.5, "retrieval_ms": 42.0,
    },
    {
        "cid": "am02", "title": "Minimal signal", "category": "ambiguous",
        "difficulty": "hard",
        "expected_facets": [],
        "retrieved_facets": ["Curiosity"],
        "recall_at_5": 1.0, "recall_at_10": 1.0, "recall_at_20": 1.0,
        "rr": 0.0, "expected_rank": None, "retrieval_ms": 38.0,
    },
    {
        "cid": "r04", "title": "Suspicion", "category": "relational",
        "difficulty": "medium",
        "expected_facets": ["Suspicion"],
        "retrieved_facets": ["Courage", "Honesty"],  # miss
        "recall_at_5": 0.0, "recall_at_10": 0.0, "recall_at_20": 0.0,
        "rr": 0.0, "expected_rank": None, "retrieval_ms": 50.0,
    },
]

_SCORING_DETAILS = [
    {
        "cid": "p01", "facet": "Risktaking", "category": "personality",
        "predicted_score": 5, "expected_score": 5,
        "confidence": 0.9, "retrieved": True, "scoring_ms": 10.0,
    },
    {
        "cid": "p01", "facet": "Determinedness", "category": "personality",
        "predicted_score": 3, "expected_score": 4,
        "confidence": 0.7, "retrieved": True, "scoring_ms": 10.0,
    },
    {
        "cid": "e01", "facet": "Compassion", "category": "emotion",
        "predicted_score": 4, "expected_score": 5,
        "confidence": 0.6, "retrieved": True, "scoring_ms": 12.0,
    },
    {
        "cid": "e01", "facet": "Compassion Fatigue", "category": "emotion",
        "predicted_score": None, "expected_score": 4,
        "confidence": None, "retrieved": False, "scoring_ms": 12.0,
    },
]

_RETRIEVAL_METRICS = {
    "recall_at_5": 0.75,
    "recall_at_10": 0.80,
    "recall_at_20": 0.85,
    "mrr": 0.65,
    "mean_expected_rank": 3.2,
}

_SCORING_METRICS = {
    "accuracy": 0.333,
    "mae": 0.667,
    "rmse": 0.816,
    "per_category": {
        "personality": {"mae": 0.5, "accuracy": 0.5, "n": 2},
        "emotion": {"mae": 1.0, "accuracy": 0.0, "n": 1},
    },
    "n": 3,
}

_CONFIDENCE_METRICS = {
    "ece": 0.08,
    "avg_confidence": 0.73,
    "accuracy": 0.333,
    "n": 3,
    "bins": [
        {"bin_low": 0.5, "bin_high": 0.6, "acc": 0.0, "mean_conf": 0.6, "n": 1},
        {"bin_low": 0.6, "bin_high": 0.7, "acc": 0.5, "mean_conf": 0.7, "n": 2},
        {"bin_low": 0.8, "bin_high": 0.9, "acc": 1.0, "mean_conf": 0.9, "n": 1},
    ],
}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ===========================================================================
# conversation_bank tests
# ===========================================================================

def test_load_conversations_returns_list():
    name = "load_conversations_returns_list"
    try:
        from evaluation.conversation_bank import load_conversations
        convs = load_conversations()
        assert isinstance(convs, list)
        assert len(convs) > 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_conversations_have_required_keys():
    name = "conversations_have_required_keys"
    try:
        from evaluation.conversation_bank import load_conversations
        convs = load_conversations()
        for c in convs:
            for key in ("cid", "turns", "expected_facets"):
                assert key in c, f"Missing key '{key}' in conversation {c.get('cid')}"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_load_conversations_difficulty_filter():
    name = "load_conversations_difficulty_filter"
    try:
        from evaluation.conversation_bank import load_conversations
        easy = load_conversations(difficulty="easy")
        assert all(c["difficulty"] == "easy" for c in easy)
        assert len(easy) > 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_load_conversations_category_filter():
    name = "load_conversations_category_filter"
    try:
        from evaluation.conversation_bank import load_conversations
        emotion = load_conversations(category="emotion")
        assert all(c["category"] == "emotion" for c in emotion)
        assert len(emotion) > 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# metrics unit tests (pure functions — already in evaluation/metrics.py)
# ===========================================================================

def test_recall_at_k_perfect():
    name = "recall_at_k_perfect"
    try:
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["A", "B", "C"], ["A", "B"], k=5) == 1.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_recall_at_k_miss():
    name = "recall_at_k_miss"
    try:
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["X", "Y"], ["A"], k=5) == 0.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_recall_at_k_empty_expected():
    name = "recall_at_k_empty_expected"
    try:
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["A", "B"], [], k=5) == 1.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_reciprocal_rank_first():
    name = "reciprocal_rank_first"
    try:
        from evaluation.metrics import reciprocal_rank
        assert reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_reciprocal_rank_third():
    name = "reciprocal_rank_third"
    try:
        from evaluation.metrics import reciprocal_rank
        rr = reciprocal_rank(["X", "Y", "A"], ["A"])
        assert abs(rr - 1/3) < 1e-9
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mrr_aggregation():
    name = "mrr_aggregation"
    try:
        from evaluation.metrics import mean_reciprocal_rank
        mrr = mean_reciprocal_rank([1.0, 0.5])
        assert abs(mrr - 0.75) < 1e-9
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_scoring_metrics_perfect():
    name = "scoring_metrics_perfect"
    try:
        from evaluation.metrics import scoring_metrics
        preds = [
            {"facet_name": "A", "predicted_score": 4, "expected_score": 4, "category": "x"},
            {"facet_name": "B", "predicted_score": 2, "expected_score": 2, "category": "x"},
        ]
        m = scoring_metrics(preds)
        assert m["accuracy"] == 1.0
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_scoring_metrics_empty():
    name = "scoring_metrics_empty"
    try:
        from evaluation.metrics import scoring_metrics
        m = scoring_metrics([])
        assert m["accuracy"] == 0.0
        assert m["n"] == 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_confidence_metrics_runs():
    name = "confidence_metrics_runs"
    try:
        from evaluation.metrics import confidence_metrics
        preds = [
            {"confidence": 0.9, "predicted_score": 4, "expected_score": 4},
            {"confidence": 0.5, "predicted_score": 3, "expected_score": 5},
            {"confidence": 0.7, "predicted_score": 2, "expected_score": 2},
        ]
        m = confidence_metrics(preds)
        assert "ece" in m
        assert 0.0 <= m["ece"] <= 1.0
        assert 0.0 <= m["avg_confidence"] <= 1.0
        assert m["n"] == 3
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_coverage_metrics_runs():
    name = "coverage_metrics_runs"
    try:
        from evaluation.metrics import coverage_metrics
        all_f = ["A", "B", "C", "D"]
        tested = ["A", "B"]
        cat_map = {"A": "x", "B": "x", "C": "y", "D": "y"}
        m = coverage_metrics(all_f, tested, cat_map)
        assert m["total_facets"] == 4
        assert m["tested_facets"] == 2
        assert m["overall_coverage_pct"] == 50.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# retrieval eval — logic test with synthetic data
# ===========================================================================

def test_retrieval_eval_aggregate_shape():
    """aggregate_retrieval_metrics produces expected keys from synthetic records."""
    name = "retrieval_eval_aggregate_shape"
    try:
        from evaluation.metrics import (
            retrieval_metrics_for_conversation,
            aggregate_retrieval_metrics,
        )
        per_conv = [
            retrieval_metrics_for_conversation(r["retrieved_facets"], r["expected_facets"])
            for r in _RETRIEVAL_DETAILS
        ]
        agg = aggregate_retrieval_metrics(per_conv)
        for key in ("recall_at_5", "recall_at_10", "recall_at_20", "mrr"):
            assert key in agg, f"Missing key '{key}' in aggregate"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_retrieval_details_saved_and_loaded():
    name = "retrieval_details_saved_and_loaded"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "retrieval_details.json"
            _write_json(p, _RETRIEVAL_DETAILS)
            loaded = json.loads(p.read_text())
            assert len(loaded) == len(_RETRIEVAL_DETAILS)
            assert loaded[0]["cid"] == "p01"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# scoring eval — logic test with synthetic data
# ===========================================================================

def test_scoring_eval_only_retrieved_included():
    """Only records with retrieved=True should feed into scoring_metrics."""
    name = "scoring_eval_only_retrieved_included"
    try:
        from evaluation.metrics import scoring_metrics
        scored = [
            {
                "facet_name": r["facet"],
                "predicted_score": r["predicted_score"],
                "expected_score": r["expected_score"],
                "confidence": r["confidence"],
                "category": r["category"],
            }
            for r in _SCORING_DETAILS
            if r.get("retrieved") and r.get("predicted_score") is not None
        ]
        m = scoring_metrics(scored)
        assert m["n"] == 3  # one record has retrieved=False → excluded
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_scoring_metrics_file_written():
    name = "scoring_metrics_file_written"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "scoring_metrics.json"
            _write_json(p, _SCORING_METRICS)
            loaded = json.loads(p.read_text())
            assert "accuracy" in loaded
            assert "mae" in loaded
            assert "rmse" in loaded
            assert "per_category" in loaded
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# confidence eval — logic test
# ===========================================================================

def test_confidence_eval_reuses_scoring_details():
    """confidence eval filters retrieved=True records for calibration."""
    name = "confidence_eval_reuses_scoring_details"
    try:
        from evaluation.metrics import confidence_metrics
        predictions = [
            {
                "confidence": r["confidence"],
                "predicted_score": r["predicted_score"],
                "expected_score": r["expected_score"],
            }
            for r in _SCORING_DETAILS
            if r.get("retrieved") and r.get("confidence") is not None
            and r.get("predicted_score") is not None
        ]
        m = confidence_metrics(predictions)
        assert m["n"] == 3
        assert 0.0 <= m["ece"] <= 1.0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_confidence_metrics_file_schema():
    name = "confidence_metrics_file_schema"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "confidence_metrics.json"
            _write_json(p, _CONFIDENCE_METRICS)
            loaded = json.loads(p.read_text())
            for key in ("ece", "avg_confidence", "accuracy", "n", "bins"):
                assert key in loaded
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# report generation
# ===========================================================================

def test_report_generation_succeeds():
    """generate_report.run() writes REPORT.md given all three JSON inputs."""
    name = "report_generation_succeeds"
    try:
        import importlib, types

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "evaluation_results"
            out_dir.mkdir()

            _write_json(out_dir / "retrieval_metrics.json", _RETRIEVAL_METRICS)
            _write_json(out_dir / "scoring_metrics.json", _SCORING_METRICS)
            _write_json(out_dir / "confidence_metrics.json", _CONFIDENCE_METRICS)
            _write_json(out_dir / "retrieval_details.json", _RETRIEVAL_DETAILS)
            _write_json(out_dir / "scoring_details.json", _SCORING_DETAILS)

            # Patch OUTPUT_DIR inside generate_report dynamically
            import evaluation.generate_report as gr_mod
            orig_dir = gr_mod.OUTPUT_DIR
            gr_mod.OUTPUT_DIR = out_dir
            try:
                report_path = gr_mod.run()
            finally:
                gr_mod.OUTPUT_DIR = orig_dir

            assert report_path.exists(), "REPORT.md was not created"
            content = report_path.read_text(encoding="utf-8")
            for section in ("## Overview", "## Retrieval", "## Scoring",
                            "## Confidence", "## Coverage", "## Recommendations"):
                assert section in content, f"Missing section: {section}"

        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_report_contains_retrieval_numbers():
    name = "report_contains_retrieval_numbers"
    try:
        import evaluation.generate_report as gr_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "evaluation_results"
            out_dir.mkdir()
            _write_json(out_dir / "retrieval_metrics.json", _RETRIEVAL_METRICS)
            _write_json(out_dir / "scoring_metrics.json", _SCORING_METRICS)
            _write_json(out_dir / "confidence_metrics.json", _CONFIDENCE_METRICS)

            orig_dir = gr_mod.OUTPUT_DIR
            gr_mod.OUTPUT_DIR = out_dir
            try:
                report_path = gr_mod.run()
            finally:
                gr_mod.OUTPUT_DIR = orig_dir

            content = report_path.read_text(encoding="utf-8")
            assert "0.8000" in content or "0.80" in content  # Recall@10
            assert "MRR" in content

        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_report_missing_files_handled_gracefully():
    """Report generation must not crash when some JSON files are absent."""
    name = "report_missing_files_handled_gracefully"
    try:
        import evaluation.generate_report as gr_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "evaluation_results"
            out_dir.mkdir()
            # Only retrieval_metrics.json present; others missing

            _write_json(out_dir / "retrieval_metrics.json", _RETRIEVAL_METRICS)

            orig_dir = gr_mod.OUTPUT_DIR
            gr_mod.OUTPUT_DIR = out_dir
            try:
                report_path = gr_mod.run()
            finally:
                gr_mod.OUTPUT_DIR = orig_dir

            assert report_path.exists()

        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 17A — Evaluation Framework")
    print("=" * 60)

    print("\n── Conversation Bank ──")
    test_load_conversations_returns_list()
    test_conversations_have_required_keys()
    test_load_conversations_difficulty_filter()
    test_load_conversations_category_filter()

    print("\n── Metrics (pure functions) ──")
    test_recall_at_k_perfect()
    test_recall_at_k_miss()
    test_recall_at_k_empty_expected()
    test_reciprocal_rank_first()
    test_reciprocal_rank_third()
    test_mrr_aggregation()
    test_scoring_metrics_perfect()
    test_scoring_metrics_empty()
    test_confidence_metrics_runs()
    test_coverage_metrics_runs()

    print("\n── Retrieval Eval ──")
    test_retrieval_eval_aggregate_shape()
    test_retrieval_details_saved_and_loaded()

    print("\n── Scoring Eval ──")
    test_scoring_eval_only_retrieved_included()
    test_scoring_metrics_file_written()

    print("\n── Confidence Eval ──")
    test_confidence_eval_reuses_scoring_details()
    test_confidence_metrics_file_schema()

    print("\n── Report Generation ──")
    test_report_generation_succeeds()
    test_report_contains_retrieval_numbers()
    test_report_missing_files_handled_gracefully()

    print()
    if _FAILED:
        print(f"✗ {len(_FAILED)} test(s) failed: {_FAILED}")
        sys.exit(1)
    else:
        print(f"✅ All Phase 17A tests passed ({len(_PASSED)} tests)")