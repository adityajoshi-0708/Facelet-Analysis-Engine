"""
Phase 6 — Dense Retriever
Encodes the query string and searches the FAISS index for the top-k most
similar facets using inner-product (≡ cosine because vectors are L2-normalised).
"""

import json
from pathlib import Path
from pydoc import text
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..models.embedding_client import EmbeddingClient
from .index_builder import IndexBuilder

logger = get_logger(__name__)


class DenseRetriever:
    """
    Retrieve the top-k facets for an arbitrary query string using dense
    (embedding-based) similarity.

    Args:
        client: Optional pre-built EmbeddingClient; created from config if None.
        top_k:  Maximum number of results to return; falls back to config value.
    """

    def __init__(
        self,
        client: Optional[EmbeddingClient] = None,
        top_k: Optional[int] = None,
    ):
        cfg = load_config()
        ret_cfg = cfg.get("retrieval", {})

        self._top_k: int = top_k or ret_cfg.get("top_k", 40)
        self._client = client or EmbeddingClient()
        self._builder = IndexBuilder(client=self._client)

        # Lazy-load index + metadata
        self._index = None
        self._metadata: Optional[Dict[str, Dict]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._index is not None:
            return
        self._index = self._builder.load()
        meta_path = Path(load_config()["data"]["index_dir"]) / "facet_metadata.json"
        self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(f"DenseRetriever: loaded index ({self._index.ntotal} vectors)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search the FAISS index for the top-k facets matching *query*.

        Args:
            query:  Free-text query (e.g. a conversation turn).
            top_k:  Override the instance default.

        Returns:
            List of dicts ordered by descending score::

                [
                  {
                    "facet_id":   "risk_taking",
                    "facet_name": "Risk Taking",
                    "category":   "personality",
                    "score":      0.91,         # inner-product ∈ [-1, 1]
                    "rank":       1,
                  },
                  ...
                ]
        """
        self._ensure_loaded()
        k = min(top_k or self._top_k, self._index.ntotal)
        
        query_vec = self._client.encode(query).reshape(1, -1)
        distances, indices = self._index.search(query_vec, k=k)

        results = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0].tolist(), indices[0].tolist()), start=1
        ):
            if idx < 0:  # FAISS sentinel for "not enough results"
                continue
            meta = self._metadata.get(str(idx), {})
            if not meta:
                logger.warning(f"DenseRetriever: no metadata for index {idx}, skipping")
                continue
            results.append(
                {
                    "facet_id":   meta.get("facet_id", ""),
                    "facet_name": meta.get("facet_name", ""),
                    "category":   meta.get("category", ""),
                    "score":      float(dist),
                    "rank":       rank,
                }
            )
        def encode_query(self, text: str):
            return self.encode(
                f"Represent this conversation for personality trait retrieval: {text}"
            )    

        logger.debug(
            f"DenseRetriever: '{query[:60]}' → {len(results)} results "
            f"(top: {results[0]['facet_name'] if results else 'none'})"
        )
        return results