"""
Phase 7 — BM25 Retriever
Keyword-based retrieval using rank_bm25 (BM25Okapi).
Each facet is represented as a bag-of-words document built from:
    facet_name + description + positive_indicators + synonyms
This gives the sparse retriever good lexical coverage.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall of in on at to for with by "
    "and or not but from that this it its".split()
)


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _listify(raw: object) -> List[str]:
    if raw is None or pd.isna(raw):
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item is not None]
    if isinstance(raw, str):
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            items = None
        if isinstance(items, list):
            return [str(item) for item in items if item is not None]
        return [raw]
    return [str(raw)]


def _build_document(row: pd.Series) -> List[str]:
    """Concatenate all textual fields into one token list for a facet."""
    # retrieval_text is the richest single source — use it when present
    rt = str(row.get("retrieval_text", "")).strip()
    parts = [rt] if rt else [
        str(row.get("facet_name", "")),
        str(row.get("description", "")),
    ]
    # Always append keywords and examples regardless — they add BM25 lexical coverage
    kw = row.get("keywords", "")
    if kw and not (isinstance(kw, float)):
        parts.extend(_listify(kw))
    for col in ("positive_indicators", "negative_indicators", "synonyms", "related_facets", "examples"):
        parts.extend(_listify(row.get(col, [])))

    return _tokenise(" ".join(parts))


class BM25Retriever:
    """
    Retrieve facets via BM25 keyword similarity.

    Args:
        top_k: Maximum results to return; falls back to config.
    """

    def __init__(self, top_k: Optional[int] = None):
        cfg = load_config()
        ret_cfg = cfg.get("retrieval", {})
        self._top_k: int = top_k or ret_cfg.get("top_k", 40)

        processed_dir = Path(cfg["data"]["processed_dir"])
        self._enriched_path = processed_dir / "facets_enriched.csv"

        # Lazy-build
        self._bm25 = None
        self._facets: Optional[List[Dict]] = None  # list of {facet_id, facet_name, category}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_built(self):
        if self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi

        if not self._enriched_path.exists():
            raise FileNotFoundError(
                f"Enriched facets CSV not found at {self._enriched_path}. "
                "Run Phase 2 (enrich_facets) first."
            )

        df = pd.read_csv(self._enriched_path)
        corpus = [_build_document(row) for _, row in df.iterrows()]
        self._bm25 = BM25Okapi(corpus)
        self._facets = [
            {
                "facet_id":   row["facet_id"],
                "facet_name": row["facet_name"],
                "category":   row.get("category", ""),
            }
            for _, row in df.iterrows()
        ]
        logger.info(f"BM25Retriever: indexed {len(corpus)} facets")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve facets matching *query* by BM25 score.

        Returns:
            List of dicts ordered by descending BM25 score::

                [
                  {
                    "facet_id":   "risk_taking",
                    "facet_name": "Risk Taking",
                    "category":   "personality",
                    "score":      12.4,
                    "rank":       1,
                  },
                  ...
                ]
        """
        self._ensure_built()
        k = top_k or self._top_k

        query_tokens = _tokenise(query)
        if not query_tokens:
            logger.warning("BM25Retriever: empty token list after tokenisation")
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with facet metadata and sort descending
        scored = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for rank, (idx, score) in enumerate(scored, start=1):
            meta = self._facets[idx]
            results.append(
                {
                    "facet_id":   meta["facet_id"],
                    "facet_name": meta["facet_name"],
                    "category":   meta["category"],
                    "score":      float(score),
                    "rank":       rank,
                }
            )

        logger.debug(
            f"BM25Retriever: '{query[:60]}' → {len(results)} results "
            f"(top: {results[0]['facet_name'] if results else 'none'})"
        )
        return results