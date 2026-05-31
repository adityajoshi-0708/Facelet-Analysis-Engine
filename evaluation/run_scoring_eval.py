"""
evaluation/run_scoring_eval.py — Phase 17A: Scoring Evaluation

Run from project root:
    python evaluation/run_scoring_eval.py

Evaluates ScorePipeline score predictions against expected_scores in
the conversation bank. Uses Ollama/Qwen when use_llm=True and raises if
Ollama is unavailable, to prevent accidentally scoring with MockLLMClient.

Saves:
    evaluation_results/scoring_metrics.json
    evaluation_results/scoring_details.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.conversation_bank import load_conversations
from evaluation.metrics import scoring_metrics
from src.models.llm_client import MockLLMClient, get_llm_client
from src.scoring.score_pipeline import ScorePipeline
from src.utils.logger import get_logger

log = get_logger("scoring_eval")


def _build_text(conv: dict) -> str:
    return " ".join(
        t["text"] for t in conv["turns"] if t.get("speaker") == "user"
    )


def run(use_llm: bool = True) -> dict:
    log.info("Loading conversation bank …")
    conversations = load_conversations()

    # Filter to those with expected_scores
    scoreable = [c for c in conversations if c.get("expected_scores")]
    log.info(f"  {len(scoreable)}/{len(conversations)} conversations have expected_scores")

    if use_llm:
        llm_client = get_llm_client(provider="ollama")
        if isinstance(llm_client, MockLLMClient):
            raise RuntimeError(
                "LLM scoring requested with use_llm=True, but Ollama is unavailable. "
                "Start Ollama (`ollama serve`) or set use_llm=False to run mock scoring."
            )
    else:
        llm_client = get_llm_client(provider="mock")

    log.info(
        f"Using LLM client: {llm_client.__class__.__name__} "
        f"(model={getattr(llm_client, 'model_name', 'unknown')})"
    )
    pipeline = ScorePipeline(llm_client=llm_client, use_llm=use_llm)

    all_predictions = []
    details = []

    for conv in scoreable:
        cid = conv["cid"]
        expected_scores: dict = conv["expected_scores"]
        category = conv.get("category", "unknown")
        text = _build_text(conv)

        t0 = time.perf_counter()
        try:
            result = pipeline.score_text(text, turn_id=cid)
        except Exception as exc:
            log.warning(f"  [{cid}] scoring failed: {exc}")
            continue
        scoring_ms = (time.perf_counter() - t0) * 1000

        # Build a map from facet_name → FacetScore for this result
        scored_map = {fs.facet_name: fs for fs in result.facet_scores}

        for facet_name, expected_score in expected_scores.items():
            fs = scored_map.get(facet_name)
            if fs is None:
                # Facet was not retrieved/scored — skip for metric computation
                # but record as a miss in details
                details.append({
                    "cid": cid,
                    "facet": facet_name,
                    "category": category,
                    "predicted_score": None,
                    "expected_score": expected_score,
                    "confidence": None,
                    "retrieved": False,
                    "scoring_ms": round(scoring_ms, 2),
                })
                continue

            all_predictions.append({
                "facet_name": facet_name,
                "predicted_score": fs.score,
                "expected_score": expected_score,
                "confidence": fs.confidence,
                "category": category,
            })

            details.append({
                "cid": cid,
                "facet": facet_name,
                "category": category,
                "predicted_score": fs.score,
                "expected_score": expected_score,
                "confidence": fs.confidence,
                "retrieved": True,
                "scoring_ms": round(scoring_ms, 2),
            })

            log.info(
                f"  [{cid}] {facet_name}: pred={fs.score} exp={expected_score} "
                f"conf={fs.confidence:.2f}"
            )

    metrics = scoring_metrics(all_predictions)

    # Save
    (OUTPUT_DIR / "scoring_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "scoring_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 55)
    print("SCORING EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Conversations with scores : {len(scoreable)}")
    print(f"  Facet predictions made    : {metrics['n']}")
    print(f"  Accuracy (exact)          : {metrics['accuracy']:.4f}")
    print(f"  MAE                       : {metrics['mae']:.4f}")
    print(f"  RMSE                      : {metrics['rmse']:.4f}")
    print("\n  Per-category breakdown:")
    for cat, cat_m in sorted(metrics["per_category"].items()):
        print(f"    {cat:20s}  acc={cat_m['accuracy']:.2f}  mae={cat_m['mae']:.2f}  n={cat_m['n']}")
    print(f"\n  Saved → {OUTPUT_DIR / 'scoring_metrics.json'}")
    print(f"  Saved → {OUTPUT_DIR / 'scoring_details.json'}")

    return metrics


if __name__ == "__main__":
    run()