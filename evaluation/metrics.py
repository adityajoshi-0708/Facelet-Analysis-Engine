"""
evaluation/metrics.py — Phase 17A metric computations.

Pure functions — no pipeline imports.
All functions take plain Python dicts/lists matching pipeline output shapes.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_facet_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return normalized


def _canonicalize_label(value: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    norm = _normalize_facet_name(value)
    if alias_map is None:
        return norm
    return alias_map.get(norm, norm)


def recall_at_k(
    retrieved: List[str],
    expected: List[str],
    k: int,
    alias_map: Optional[Dict[str, str]] = None,
) -> float:
    """
    Recall@k = |relevant ∩ top-k retrieved| / |relevant|

    retrieved : ordered list of facet_names from the retriever
    expected  : list of ground-truth facet_names
    """
    if not expected:
        return 1.0  # nothing expected → vacuously satisfied
    top_k = {
        _canonicalize_label(r, alias_map)
        for r in retrieved[:k] if r
    }
    expected_norm = [
        _canonicalize_label(e, alias_map)
        for e in expected if e
    ]
    hits = sum(1 for e in expected_norm if e in top_k)
    return hits / len(expected_norm)


def reciprocal_rank(
    retrieved: List[str],
    expected: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> float:
    """
    Reciprocal rank of the FIRST relevant item in the retrieved list.
    Returns 0 if none found.
    """
    expected_set = {
        _canonicalize_label(e, alias_map) for e in expected if e
    }
    for i, name in enumerate(retrieved, start=1):
        if _canonicalize_label(name, alias_map) in expected_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rr_list: List[float]) -> float:
    if not rr_list:
        return 0.0
    return sum(rr_list) / len(rr_list)


def expected_rank(
    retrieved: List[str],
    expected: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """
    Average rank of ALL relevant items found in the retrieved list.
    Returns None if none found.
    """
    expected_set = {
        _canonicalize_label(e, alias_map) for e in expected if e
    }
    ranks = [
        i for i, n in enumerate(retrieved, start=1)
        if _canonicalize_label(n, alias_map) in expected_set
    ]
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def retrieval_metrics_for_conversation(
    retrieved: List[str],
    expected: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Full retrieval metric bundle for a single conversation."""
    return {
        "recall_at_5":  recall_at_k(retrieved, expected, 5, alias_map),
        "recall_at_10": recall_at_k(retrieved, expected, 10, alias_map),
        "recall_at_20": recall_at_k(retrieved, expected, 20, alias_map),
        "rr":           reciprocal_rank(retrieved, expected, alias_map),
        "expected_rank": expected_rank(retrieved, expected, alias_map),
        "n_expected":   len(expected),
        "n_retrieved":  len(retrieved),
    }


def aggregate_retrieval_metrics(
    per_conv: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate per-conversation retrieval metrics → global means."""
    keys = ["recall_at_5", "recall_at_10", "recall_at_20", "rr"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in per_conv if r[k] is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    out["mrr"] = out.pop("rr")  # rename for clarity
    ranks = [r["expected_rank"] for r in per_conv if r["expected_rank"] is not None]
    out["mean_expected_rank"] = sum(ranks) / len(ranks) if ranks else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scoring metrics
# ─────────────────────────────────────────────────────────────────────────────

def scoring_metrics(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    predictions: list of dicts with keys:
        facet_name, predicted_score (int), expected_score (int), category (str)

    Returns accuracy, MAE, RMSE, per-category breakdown.
    """
    if not predictions:
        return {"accuracy": 0.0, "mae": 0.0, "rmse": 0.0, "per_category": {}, "n": 0}

    errors  = [abs(p["predicted_score"] - p["expected_score"]) for p in predictions]
    correct = [e == 0 for e in errors]

    mae   = sum(errors) / len(errors)
    rmse  = math.sqrt(sum(e**2 for e in errors) / len(errors))
    acc   = sum(correct) / len(correct)

    # per-category
    cat_buckets: Dict[str, list] = defaultdict(list)
    for p in predictions:
        cat_buckets[p.get("category", "unknown")].append(
            abs(p["predicted_score"] - p["expected_score"])
        )
    per_cat = {
        cat: {
            "mae":   sum(errs) / len(errs),
            "accuracy": sum(1 for e in errs if e == 0) / len(errs),
            "n":     len(errs),
        }
        for cat, errs in cat_buckets.items()
    }

    return {
        "accuracy": round(acc, 4),
        "mae":      round(mae, 4),
        "rmse":     round(rmse, 4),
        "per_category": per_cat,
        "n": len(predictions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence / calibration metrics
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: List[float],
    correct_flags: List[bool],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Returns (ece_score, bin_details).
    bin_details: list of dicts per bin: {bin_low, bin_high, acc, mean_conf, n}
    """
    n = len(confidences)
    if n == 0:
        return 0.0, []

    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, flag in zip(confidences, correct_flags):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, flag))

    ece = 0.0
    details = []
    for i, b in enumerate(bins):
        if not b:
            details.append({
                "bin_low": i / n_bins, "bin_high": (i + 1) / n_bins,
                "acc": None, "mean_conf": None, "n": 0,
            })
            continue
        acc       = sum(f for _, f in b) / len(b)
        mean_conf = sum(c for c, _ in b) / len(b)
        ece      += (len(b) / n) * abs(acc - mean_conf)
        details.append({
            "bin_low":   round(i / n_bins, 2),
            "bin_high":  round((i + 1) / n_bins, 2),
            "acc":       round(acc, 4),
            "mean_conf": round(mean_conf, 4),
            "n":         len(b),
        })

    return round(ece, 4), details


def confidence_metrics(
    predictions: List[Dict[str, Any]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    predictions: list of dicts with:
        confidence (float), predicted_score (int), expected_score (int)
    """
    if not predictions:
        return {"ece": 0.0, "avg_confidence": 0.0, "n": 0, "bins": []}

    confs   = [p["confidence"] for p in predictions]
    correct = [p["predicted_score"] == p["expected_score"] for p in predictions]

    ece, bins = expected_calibration_error(confs, correct, n_bins)

    return {
        "ece":            ece,
        "avg_confidence": round(sum(confs) / len(confs), 4),
        "accuracy":       round(sum(correct) / len(correct), 4),
        "n":              len(predictions),
        "bins":           bins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Latency metrics
# ─────────────────────────────────────────────────────────────────────────────

def latency_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    records: list of dicts with optional keys:
        retrieval_ms, scoring_ms, total_ms
    """
    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(vals)
        n = len(s)
        p50_idx = int(n * 0.50)
        p95_idx = min(int(n * 0.95), n - 1)
        return {
            "mean": round(sum(s) / n, 2),
            "p50":  round(s[p50_idx], 2),
            "p95":  round(s[p95_idx], 2),
            "min":  round(s[0], 2),
            "max":  round(s[-1], 2),
        }

    return {
        "retrieval": _stats([r["retrieval_ms"] for r in records if "retrieval_ms" in r]),
        "scoring":   _stats([r["scoring_ms"]   for r in records if "scoring_ms"   in r]),
        "total":     _stats([r["total_ms"]      for r in records if "total_ms"      in r]),
        "n": len(records),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coverage metrics
# ─────────────────────────────────────────────────────────────────────────────

def coverage_metrics(
    all_facets: List[str],
    tested_facets: List[str],
    category_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    all_facets    : full 399-facet catalog names
    tested_facets : facet names that appear in benchmark expected_facets
    category_map  : facet_name → category
    """
    all_set    = set(all_facets)
    tested_set = set(tested_facets)
    untested   = sorted(all_set - tested_set)

    # per-category coverage
    cat_total:  Dict[str, int] = defaultdict(int)
    cat_tested: Dict[str, int] = defaultdict(int)
    for f in all_facets:
        cat = category_map.get(f, "unknown")
        cat_total[cat] += 1
    for f in tested_facets:
        cat = category_map.get(f, "unknown")
        cat_tested[cat] += 1

    per_cat = {
        cat: {
            "total":   cat_total[cat],
            "tested":  cat_tested.get(cat, 0),
            "coverage_pct": round(
                100 * cat_tested.get(cat, 0) / cat_total[cat], 1
            ) if cat_total[cat] else 0.0,
        }
        for cat in sorted(cat_total)
    }

    return {
        "total_facets":   len(all_set),
        "tested_facets":  len(tested_set),
        "untested_facets": len(untested),
        "overall_coverage_pct": round(100 * len(tested_set) / len(all_set), 1) if all_set else 0.0,
        "per_category": per_cat,
        "untested_list": untested,
    }