"""
Phase 8 — Hybrid Retriever
Combines dense (FAISS) and BM25 scores using Reciprocal Rank Fusion (RRF)
weighted by the configurable dense_weight / bm25_weight parameters.

RRF score for facet i  =  dense_weight  * 1/(rank_dense_i  + K)
                        + bm25_weight   * 1/(rank_bm25_i   + K)

where K=60 is the standard RRF constant that dampens the influence of very
high-ranked but poorly-scoring results.
"""

from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..models.embedding_client import EmbeddingClient
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever

logger = get_logger(__name__)

_RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """
    Fuse dense and BM25 rankings via weighted Reciprocal Rank Fusion.

    Args:
        dense_weight: Weight for dense retriever contribution (default: config value).
        bm25_weight:  Weight for BM25 retriever contribution (default: config value).
        top_k:        Number of results to return after fusion.
        client:       Optional shared EmbeddingClient.
    """

    def __init__(
        self,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[EmbeddingClient] = None,
    ):
        cfg = load_config()
        ret_cfg = cfg.get("retrieval", {})

        self._dense_weight: float = dense_weight if dense_weight is not None else ret_cfg.get("dense_weight", 0.95)
        self._bm25_weight:  float = bm25_weight  if bm25_weight  is not None else ret_cfg.get("bm25_weight",  0.05)
        self._top_k: int = top_k or ret_cfg.get("top_k", 40)

        _client = client or EmbeddingClient()
        # Give sub-retrievers a larger candidate pool to improve fusion coverage.
        pool = max(self._top_k * 4, 80)
        self._dense = DenseRetriever(client=_client, top_k=pool)
        self._bm25  = BM25Retriever(top_k=pool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_score(rank: int, weight: float) -> float:
        return weight / (rank + _RRF_K)

    def _fuse(
        self,
        dense_results: List[Dict],
        bm25_results:  List[Dict],
    ) -> List[Dict]:
        """Merge two ranked lists via weighted RRF."""
        scores: Dict[str, float] = {}
        meta:   Dict[str, Dict]  = {}

        for item in dense_results:
            fid = item["facet_id"]
            scores[fid] = scores.get(fid, 0.0) + self._rrf_score(item["rank"], self._dense_weight)
            meta.setdefault(fid, item)

        for item in bm25_results:
            fid = item["facet_id"]
            scores[fid] = scores.get(fid, 0.0) + self._rrf_score(item["rank"], self._bm25_weight)
            meta.setdefault(fid, item)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self._top_k]

        results = []
        for rank, (fid, rrf_score) in enumerate(fused, start=1):
            entry = dict(meta[fid])
            entry["score"] = rrf_score
            entry["rank"]  = rank
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve the top-k facets for *query* using hybrid fusion.

        Args:
            query:  Free-text query string.
            top_k:  Override the instance default.

        Returns:
            List of facet dicts ordered by descending hybrid RRF score::

                [
                  {
                    "facet_id":   "risk_taking",
                    "facet_name": "Risk Taking",
                    "category":   "personality",
                    "score":      0.0112,
                    "rank":       1,
                  },
                  ...
                ]
        """
        dense_results = self._dense.retrieve(query)
        bm25_results  = self._bm25.retrieve(query)
        fused         = self._fuse(dense_results, bm25_results)

        if top_k and top_k < len(fused):
            fused = fused[:top_k]

        logger.debug(
            f"HybridRetriever: '{query[:60]}' → {len(fused)} fused results "
            f"(top: {fused[0]['facet_name'] if fused else 'none'})"
        )
        return fused