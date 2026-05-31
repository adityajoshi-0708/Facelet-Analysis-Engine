"""
evaluation/run_confidence_eval.py — Phase 17A: Confidence Calibration Evaluation

Run from project root:
    python evaluation/run_confidence_eval.py

Reuses scoring_details.json if already present; otherwise runs the full
scoring pipeline to collect (confidence, predicted_score, expected_score).

Saves:
    evaluation_results/confidence_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.metrics import confidence_metrics
from src.utils.logger import get_logger

log = get_logger("confidence_eval")

SCORING_DETAILS_PATH = OUTPUT_DIR / "scoring_details.json"


def _load_or_run_scoring() -> list[dict]:
    """Return scoring details records, running scoring eval if needed."""
    if SCORING_DETAILS_PATH.exists():
        log.info(f"Reusing existing {SCORING_DETAILS_PATH}")
        return json.loads(SCORING_DETAILS_PATH.read_text(encoding="utf-8"))

    log.info("scoring_details.json not found — running scoring eval first …")
    # Import here to avoid circular issues at module level
    from evaluation.run_scoring_eval import run as run_scoring
    run_scoring()
    if SCORING_DETAILS_PATH.exists():
        return json.loads(SCORING_DETAILS_PATH.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Scoring eval did not produce {SCORING_DETAILS_PATH}"
    )


def run(n_bins: int = 10) -> dict:
    records = _load_or_run_scoring()

    # Keep only records where scoring was performed (retrieved=True)
    scored = [
        r for r in records
        if r.get("retrieved") and r.get("confidence") is not None
        and r.get("predicted_score") is not None
        and r.get("expected_score") is not None
    ]

    log.info(f"  {len(scored)} scored facet predictions available for calibration")

    predictions = [
        {
            "confidence": r["confidence"],
            "predicted_score": r["predicted_score"],
            "expected_score": r["expected_score"],
        }
        for r in scored
    ]

    metrics = confidence_metrics(predictions, n_bins=n_bins)

    # Save
    (OUTPUT_DIR / "confidence_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 55)
    print("CONFIDENCE CALIBRATION SUMMARY")
    print("=" * 55)
    print(f"  Predictions evaluated : {metrics['n']}")
    print(f"  ECE                   : {metrics['ece']:.4f}")
    print(f"  Average Confidence    : {metrics['avg_confidence']:.4f}")
    print(f"  Accuracy              : {metrics['accuracy']:.4f}")
    print("\n  Calibration bins (non-empty):")
    for b in metrics["bins"]:
        if b["n"] > 0:
            print(
                f"    [{b['bin_low']:.1f}-{b['bin_high']:.1f}]  "
                f"acc={b['acc']:.2f}  conf={b['mean_conf']:.2f}  n={b['n']}"
            )
    print(f"\n  Saved → {OUTPUT_DIR / 'confidence_metrics.json'}")

    return metrics


if __name__ == "__main__":
    run()