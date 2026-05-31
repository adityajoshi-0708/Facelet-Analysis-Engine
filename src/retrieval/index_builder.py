"""
Phase 5 — FAISS Index Builder
Encodes all facets from the enriched CSV and stores them in a FAISS IndexFlatIP
(exact inner product; valid because embeddings are L2-normalised → cosine sim == dot product).
Also persists a metadata JSON mapping integer index → facet_id / facet_name / category.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..models.embedding_client import EmbeddingClient

logger = get_logger(__name__)


class IndexBuilder:
    """
    Build and persist a FAISS index from an enriched facets DataFrame.

    The text encoded for each facet is built from name, description, and
    enrichment fields such as indicators, synonyms and related facets.
    This gives much richer retrieval signal than the bare name.
    """

    def __init__(self, client: Optional[EmbeddingClient] = None):
        cfg = load_config()
        self._index_dir = Path(cfg["data"]["index_dir"])
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / "facets.index"
        self._metadata_path = self._index_dir / "facet_metadata.json"
        self._client = client or EmbeddingClient()

    def _listify(self, raw: object) -> list:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw if item is not None]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item is not None]
            return [raw]
        return [str(raw)]

    def _build_texts(self, df: pd.DataFrame) -> list:
        """Concatenate facet_name, description, examples, and enrichment fields for richer embeddings."""
        texts = []
        for _, row in df.iterrows():
            rt = str(row.get("retrieval_text", "")).strip()
            if rt:
                texts.append(rt)
                continue
            # fallback: manual concatenation for rows without retrieval_text
            parts = [
                str(row.get("facet_name", "")),
                str(row.get("category", "")),
                str(row.get("description", "")),
            ]
            for col in (
                "positive_indicators", "negative_indicators",
                "synonyms", "related_facets", "examples",
            ):
                parts.extend(self._listify(row.get(col, [])))
            texts.append(" ".join([p for p in parts if p]))
        return texts

    def build(self, df: pd.DataFrame):
        """
        Encode all facets and build a FAISS index.

        Args:
            df: Enriched DataFrame with facet_id, facet_name, category, description.

        Returns:
            faiss.IndexFlatIP populated with all facet vectors.
        """
        import faiss  # imported here so module is loadable without faiss installed

        texts = self._build_texts(df)
        logger.info(f"Encoding {len(texts)} facets …")
        vectors: np.ndarray = self._client.encode_batch(texts)  # (N, 384)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

        # Persist index
        faiss.write_index(index, str(self._index_path))
        logger.info(f"Saved FAISS index to {self._index_path}")

        # Build and persist metadata: int_id → {facet_id, facet_name, category}
        metadata: Dict[str, Dict] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            metadata[str(i)] = {
                "facet_id": row.get("facet_id", ""),
                "facet_name": row.get("facet_name", ""),
                "category": row.get("category", ""),
            }

        self._metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved metadata to {self._metadata_path}")

        return index

    def load(self):
        """
        Load a previously saved FAISS index from disk.

        Returns:
            faiss.IndexFlatIP loaded from data/index/facets.index.
        """
        import faiss

        if not self._index_path.exists():
            raise FileNotFoundError(f"No saved index at {self._index_path}")

        index = faiss.read_index(str(self._index_path))
        logger.info(f"Loaded FAISS index: {index.ntotal} vectors from {self._index_path}")
        return index