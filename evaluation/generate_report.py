"""
evaluation/generate_report.py — Phase 17A: Final Markdown Report Generator

Run from project root:
    python evaluation/generate_report.py

Reads evaluation_results/{retrieval,scoring,confidence}_metrics.json
and optional detail files, then writes evaluation_results/REPORT.md.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.conversation_bank import load_conversations
from evaluation.metrics import coverage_metrics
from src.utils.logger import get_logger

log = get_logger("generate_report")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(filename: str) -> dict | list | None:
    path = OUTPUT_DIR / filename
    if not path.exists():
        log.warning(f"  {filename} not found — section will be skipped")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ---------------------------------------------------------------------------
# Recommendations generator
# ---------------------------------------------------------------------------

def _generate_recommendations(
    retrieval: dict | None,
    scoring: dict | None,
    confidence: dict | None,
) -> list[str]:
    recs = []

    if retrieval:
        r10 = retrieval.get("recall_at_10", 0)
        mrr = retrieval.get("mrr", 0)
        if r10 < 0.5:
            recs.append(
                f"Recall@10 is {r10:.2f} — below 0.50. Consider expanding the "
                "seed facet list, tuning BM25 weights, or increasing `top_k`."
            )
        if mrr < 0.3:
            recs.append(
                f"MRR is {mrr:.2f}. The most-relevant facet is rarely ranked "
                "first. Review the re-ranking/graph-expansion decay factor."
            )
        if r10 >= 0.8:
            recs.append(
                "Recall@10 is strong (≥0.80). Focus optimisation effort on "
                "scoring accuracy rather than retrieval coverage."
            )

    if scoring:
        mae = scoring.get("mae", 0)
        acc = scoring.get("accuracy", 0)
        if mae > 1.0:
            recs.append(
                f"MAE is {mae:.2f} (>1.0). Enable LLM-path scoring "
                "(`use_llm=True`) or lower the `confidence_threshold` in "
                "ScoringEngine to reduce large score errors."
            )
        if acc < 0.4:
            recs.append(
                f"Exact-match accuracy is {acc:.2f}. Review rubric anchors in "
                "the enriched facets CSV — score-1 and score-5 anchors may "
                "need tightening."
            )
        per_cat = scoring.get("per_category", {})
        for cat, cm in per_cat.items():
            if cm.get("mae", 0) > 1.5:
                recs.append(
                    f"Category '{cat}' has high MAE ({cm['mae']:.2f}). "
                    "Consider adding domain-specific rubrics for this category."
                )

    if confidence:
        ece = confidence.get("ece", 0)
        avg_conf = confidence.get("avg_confidence", 0)
        if ece > 0.2:
            recs.append(
                f"ECE is {ece:.2f} (>0.20) — model is poorly calibrated. "
                "Enable logprob-based confidence (Phase 15 method A) or apply "
                "temperature scaling."
            )
        if avg_conf > 0.85:
            recs.append(
                f"Average confidence is {avg_conf:.2f} — the model may be "
                "overconfident. Inspect high-confidence, high-error predictions."
            )

    if not recs:
        recs.append(
            "All core metrics look healthy. Continue expanding the conversation "
            "bank with harder / more ambiguous examples to stress-test the pipeline."
        )

    return recs


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def run() -> Path:
    retrieval  = _load("retrieval_metrics.json")
    scoring    = _load("scoring_metrics.json")
    confidence = _load("confidence_metrics.json")
    ret_details = _load("retrieval_details.json")
    scr_details = _load("scoring_details.json")

    conversations = load_conversations()

    # Coverage
    all_facets = []
    tested_facets = []
    category_map: dict[str, str] = {}
    for conv in conversations:
        cat = conv.get("category", "unknown")
        for f in conv.get("expected_facets", []):
            all_facets.append(f)
            category_map[f] = cat
        for f in conv.get("expected_facets", []):
            tested_facets.append(f)

    # De-duplicate all_facets to represent the tested catalog
    unique_all = list(dict.fromkeys(all_facets))
    unique_tested = list(dict.fromkeys(tested_facets))
    cov = coverage_metrics(unique_all, unique_tested, category_map)

    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Ahoum Evaluation Report",
        f"*Generated: {ts}*",
        "",
    ]

    # ── Overview ────────────────────────────────────────────────────────────
    total_with_scores = sum(1 for c in conversations if c.get("expected_scores"))
    lines += [
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total benchmark conversations | {len(conversations)} |",
        f"| Conversations with expected scores | {total_with_scores} |",
        f"| Unique facets tested | {cov['tested_facets']} |",
        f"| Overall facet coverage | {cov['overall_coverage_pct']}% |",
        "",
    ]

    # ── Retrieval ────────────────────────────────────────────────────────────
    lines += ["## Retrieval", ""]
    if retrieval:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Recall@5  | {_fmt(retrieval.get('recall_at_5'))} |",
            f"| Recall@10 | {_fmt(retrieval.get('recall_at_10'))} |",
            f"| Recall@20 | {_fmt(retrieval.get('recall_at_20'))} |",
            f"| MRR       | {_fmt(retrieval.get('mrr'))} |",
            f"| Mean Expected Rank | {_fmt(retrieval.get('mean_expected_rank'), 2)} |",
            "",
        ]
    else:
        lines += ["*retrieval_metrics.json not found — run run_retrieval_eval.py first.*", ""]

    # ── Scoring ──────────────────────────────────────────────────────────────
    lines += ["## Scoring", ""]
    if scoring:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy (exact match) | {_fmt(scoring.get('accuracy'))} |",
            f"| MAE | {_fmt(scoring.get('mae'))} |",
            f"| RMSE | {_fmt(scoring.get('rmse'))} |",
            f"| Predictions | {scoring.get('n', 'n/a')} |",
            "",
            "### Per-Category Scoring",
            "",
            "| Category | Accuracy | MAE | n |",
            "|----------|----------|-----|---|",
        ]
        for cat, cm in sorted(scoring.get("per_category", {}).items()):
            lines.append(
                f"| {cat} | {_fmt(cm.get('accuracy'))} | {_fmt(cm.get('mae'))} | {cm.get('n', '')} |"
            )
        lines.append("")
    else:
        lines += ["*scoring_metrics.json not found — run run_scoring_eval.py first.*", ""]

    # ── Confidence ───────────────────────────────────────────────────────────
    lines += ["## Confidence Calibration", ""]
    if confidence:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| ECE | {_fmt(confidence.get('ece'))} |",
            f"| Average Confidence | {_fmt(confidence.get('avg_confidence'))} |",
            f"| Calibrated Accuracy | {_fmt(confidence.get('accuracy'))} |",
            f"| Predictions | {confidence.get('n', 'n/a')} |",
            "",
            "### Calibration Bins",
            "",
            "| Bin | Accuracy | Mean Conf | n |",
            "|-----|----------|-----------|---|",
        ]
        for b in confidence.get("bins", []):
            if b.get("n", 0) > 0:
                lines.append(
                    f"| {b['bin_low']:.1f}–{b['bin_high']:.1f} "
                    f"| {_fmt(b.get('acc'))} "
                    f"| {_fmt(b.get('mean_conf'))} "
                    f"| {b['n']} |"
                )
        lines.append("")
    else:
        lines += ["*confidence_metrics.json not found — run run_confidence_eval.py first.*", ""]

    # ── Coverage ─────────────────────────────────────────────────────────────
    lines += [
        "## Coverage",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total facets in bank | {cov['total_facets']} |",
        f"| Tested facets | {cov['tested_facets']} |",
        f"| Untested facets | {cov['untested_facets']} |",
        f"| Overall coverage | {cov['overall_coverage_pct']}% |",
        "",
        "### Per-Category Coverage",
        "",
        "| Category | Total | Tested | Coverage |",
        "|----------|-------|--------|----------|",
    ]
    for cat, cm in sorted(cov["per_category"].items()):
        lines.append(
            f"| {cat} | {cm['total']} | {cm['tested']} | {cm['coverage_pct']}% |"
        )
    lines.append("")

    # ── Retrieval Failures ────────────────────────────────────────────────────
    lines += ["## Top Retrieval Failures", ""]
    if ret_details:
        # Conversations where expected facets exist but recall@10 = 0
        misses = [
            r for r in ret_details
            if r.get("expected_facets") and r.get("recall_at_10", 1) == 0
        ]
        misses_sorted = sorted(misses, key=lambda x: x.get("rr", 1))[:10]
        if misses_sorted:
            lines += [
                "| cid | Title | Expected | RR |",
                "|-----|-------|----------|-----|",
            ]
            for m in misses_sorted:
                exp = ", ".join(m["expected_facets"][:3])
                lines.append(
                    f"| {m['cid']} | {m.get('title','')[:40]} | {exp} | {_fmt(m.get('rr'), 3)} |"
                )
        else:
            lines.append("*No complete retrieval failures (recall@10 > 0 for all).*")
    else:
        lines.append("*retrieval_details.json not found.*")
    lines.append("")

    # ── Scoring Failures ──────────────────────────────────────────────────────
    lines += ["## Top Scoring Errors", ""]
    if scr_details:
        scoreable = [
            r for r in scr_details
            if r.get("retrieved") and r.get("predicted_score") is not None
        ]
        scoreable_sorted = sorted(
            scoreable,
            key=lambda x: abs((x.get("predicted_score") or 0) - (x.get("expected_score") or 0)),
            reverse=True,
        )[:10]
        if scoreable_sorted:
            lines += [
                "| cid | Facet | Predicted | Expected | Error |",
                "|-----|-------|-----------|----------|-------|",
            ]
            for r in scoreable_sorted:
                err = abs((r.get("predicted_score") or 0) - (r.get("expected_score") or 0))
                lines.append(
                    f"| {r['cid']} | {r['facet']} | {r.get('predicted_score')} "
                    f"| {r.get('expected_score')} | {err} |"
                )
        else:
            lines.append("*No scoring errors to display.*")
    else:
        lines.append("*scoring_details.json not found.*")
    lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────────
    lines += ["## Recommendations", ""]
    recs = _generate_recommendations(retrieval, scoring, confidence)
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    # ── Write ────────────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report written → {report_path}")
    print(f"\n✅ Report generated → {report_path}")
    return report_path


if __name__ == "__main__":
    run()