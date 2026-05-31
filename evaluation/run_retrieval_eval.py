"""
evaluation/run_retrieval_eval.py — Phase 17A: Retrieval Evaluation

Run from project root:
    python evaluation/run_retrieval_eval.py

Evaluates HybridRetriever against the conversation bank.
Saves results under evaluation_results/.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.conversation_bank import load_conversations
from evaluation.metrics import (
    retrieval_metrics_for_conversation,
    aggregate_retrieval_metrics,
)
from src.data_pipeline.clean import clean_facets
from src.data_pipeline.enrich import enrich_facets
from src.models.embedding_client import EmbeddingClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.index_builder import IndexBuilder
from src.utils.logger import get_logger

log = get_logger("retrieval_eval")


def _build_query(conv: dict) -> str:
    """Concatenate all user turns into a single retrieval query."""
    return " ".join(
        t["text"] for t in conv["turns"] if t.get("speaker") == "user"
    )


def _normalize_facet_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _audit_expected_facets(expected: list, alias_map: dict) -> tuple:
    missing = [facet for facet in expected
               if _normalize_facet_name(facet) not in alias_map]
    return missing


def _build_alias_map(enriched_df: pd.DataFrame) -> dict:
    alias_map = {}
    for _, row in enriched_df.iterrows():
        canonical = _normalize_facet_name(str(row["facet_name"]))
        alias_map[canonical] = canonical
        alias_map[_normalize_facet_name(str(row["facet_id"]))] = canonical
        for col in ("synonyms", "related_facets"):
            raw = row.get(col, "[]")
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                items = []
            for item in items:
                if item:
                    alias_map[_normalize_facet_name(str(item))] = canonical
    return alias_map


def _ensure_index_built():
    enriched_path = ROOT / "data" / "processed" / "facets_enriched.csv"

    if not enriched_path.exists():
        raise FileNotFoundError(
            f"Missing {enriched_path}"
        )

    return pd.read_csv(enriched_path)


def run(top_k: int = 20) -> dict:
    log.info("Loading conversation bank …")
    conversations = load_conversations()
    log.info(f"  {len(conversations)} conversations loaded")

    enriched_df = _ensure_index_built()
    alias_map = _build_alias_map(enriched_df)

    client = EmbeddingClient()
    retriever = HybridRetriever(client=client)
    

    per_conv = []
    details = []

    for conv in conversations:
        cid = conv["cid"]
        expected = conv.get("expected_facets", [])
        query = _build_query(conv)

        t0 = time.perf_counter()
        try:
            results = retriever.retrieve(query, top_k=top_k)
            retrieved_names = [r["facet_name"] for r in results]
        except Exception as exc:
            log.warning(f"  [{cid}] retrieval failed: {exc}")
            retrieved_names = []
        retrieval_ms = (time.perf_counter() - t0) * 1000

        missing_expected = _audit_expected_facets(expected, alias_map)
        if missing_expected:
            log.warning(
                f"  [{cid}] {len(missing_expected)} expected facet(s) not found in current catalog: {missing_expected}"
            )

        metrics = retrieval_metrics_for_conversation(
            retrieved_names,
            expected,
            alias_map=alias_map,
        )

        per_conv.append(metrics)
        details.append({
            "cid": cid,
            "title": conv.get("title", ""),
            "category": conv.get("category", ""),
            "difficulty": conv.get("difficulty", ""),
            "expected_facets": expected,
            "missing_expected": missing_expected,
            "retrieved_facets": retrieved_names[:top_k],
            "recall_at_5": metrics["recall_at_5"],
            "recall_at_10": metrics["recall_at_10"],
            "recall_at_20": metrics["recall_at_20"],
            "rr": metrics["rr"],
            "expected_rank": metrics["expected_rank"],
            "retrieval_ms": round(retrieval_ms, 2),
        })

        status = "✓" if metrics["recall_at_10"] > 0 or not expected else "✗"
        log.info(
            f"  [{cid}] {status}  R@10={metrics['recall_at_10']:.2f}  "
            f"RR={metrics['rr']:.2f}  expected={expected}"
        )

    aggregate = aggregate_retrieval_metrics(per_conv)

    # Save
    (OUTPUT_DIR / "retrieval_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "retrieval_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Conversations : {len(conversations)}")
    print(f"  Recall@5      : {aggregate['recall_at_5']:.4f}")
    print(f"  Recall@10     : {aggregate['recall_at_10']:.4f}")
    print(f"  Recall@20     : {aggregate['recall_at_20']:.4f}")
    print(f"  MRR           : {aggregate['mrr']:.4f}")
    mean_rank = aggregate.get("mean_expected_rank")
    print(f"  Mean Exp Rank : {mean_rank:.2f}" if mean_rank else "  Mean Exp Rank : n/a")

    total_missing = sum(len(item.get("missing_expected", [])) for item in details)
    if total_missing:
        print(f"  Missing expectations: {total_missing} labels not found in current catalog.")

    print(f"\n  Saved → {OUTPUT_DIR / 'retrieval_metrics.json'}")
    print(f"  Saved → {OUTPUT_DIR / 'retrieval_details.json'}")

    return aggregate


if __name__ == "__main__":
    run()