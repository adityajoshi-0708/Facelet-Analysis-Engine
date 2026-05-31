"""
Phase 16 — Full Pipeline
Orchestrates all phases into a single pipeline.score() call.

Flow
----
ConversationTurn
    ↓  Phase 10 — FeaturePipeline
FeatureBundle (speaker, entities, sentiment)
    ↓  Phase 11 — RoutingPipeline
categories: List[str]
    ↓  Phase 8  — HybridRetriever  (filtered by categories)
retrieved: List[Dict]
    ↓  Phase 13 — EvidenceExtractor  (per facet, parallel)
evidences: List[Evidence]
    ↓  Phase 14 — ScoringEngine      (per facet, parallel)
raw_scores: List[FacetScore]
    ↓  Phase 15 — ConfidenceEngine   (per facet)
calibrated_scores: List[FacetScore]
    ↓
EvaluationResult

All components share a single LLMClient and EmbeddingClient instance to
avoid redundant model loads.

Performance features:
    - Parallel evidence extraction  (ThreadPoolExecutor, max_workers=6)
    - Parallel facet scoring        (ThreadPoolExecutor, max_workers=6)
    - Disk cache                    (data/cache/scores/, keyed by model+facet+text)
    - No GraphExpander              (removed — dense+BM25 hybrid only)

Config knobs (all from configs/):
    retrieval.top_k               — facets retrieved per query
    scoring.confidence_threshold  — minimum for Stage 1 scoring
    top_k (root)                  — final number of facet scores to return
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.types import (
    Conversation, ConversationTurn, EvaluationResult, Evidence, FacetScore,
)
from ..models.embedding_client import EmbeddingClient
from ..models.llm_client import LLMClient, get_llm_client
from ..features.feature_pipeline import FeaturePipeline
from ..routing.routing_pipeline import RoutingPipeline
from ..retrieval.hybrid_retriever import HybridRetriever
from ..evidence.evidence_extractor import EvidenceExtractor
from .rubric_engine import ScoringEngine
from ..confidence.confidence_engine import ConfidenceEngine

logger = get_logger(__name__)

_ALL_CATS = {"emotion", "personality", "cognitive", "social", "safety"}


class ScorePipeline:
    """
    End-to-end facet evaluation pipeline.

    Args:
        llm_client:       Shared LLMClient.  Created from config if None.
        embedding_client: Shared EmbeddingClient.  Created from config if None.
        use_llm:          Master switch — set False to run keyword-only mode
                          (no LLM calls; useful for testing without Ollama).
        top_k:            Number of top facet scores to include in output.
        parallel_workers: Thread count for parallel evidence + scoring (default 6).
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        use_llm: bool = True,
        top_k: Optional[int] = None,
        parallel_workers: int = 6,
    ):
        cfg = load_config()
        self._top_k: int = top_k or cfg.get("top_k", 20)
        self._use_llm = use_llm
        self._workers = parallel_workers

        # Shared clients — instantiated once, reused across all phases
        self._llm   = llm_client       or get_llm_client()
        self._embed = embedding_client or EmbeddingClient()

        # Phase components
        self._features  = FeaturePipeline()
        self._router    = RoutingPipeline()
        self._retriever = HybridRetriever(client=self._embed)
        self._extractor = EvidenceExtractor(client=self._llm, use_llm=use_llm)
        self._scorer    = ScoringEngine(client=self._llm, use_llm=use_llm)
        self._confidence = ConfidenceEngine(client=self._llm, use_llm=use_llm)

        # Disk cache
        self._cache_dir = Path("data/cache/scores")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_tag = getattr(self._llm, "model_name", "llm")

        logger.info(
            f"ScorePipeline initialised: use_llm={use_llm}, "
            f"top_k={self._top_k}, workers={self._workers}"
        )

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str, facet_id: str) -> str:
        h = hashlib.sha1()
        h.update(self._model_tag.encode())
        h.update(facet_id.encode())
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, text: str, facet_id: str) -> Optional[FacetScore]:
        p = self._cache_dir / f"{self._cache_key(text, facet_id)}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            ev = None
            if d.get("evidence"):
                ev = Evidence(
                    span=d["evidence"]["span"],
                    start_char=d["evidence"]["start_char"],
                    end_char=d["evidence"]["end_char"],
                    turn_id=d["evidence"]["turn_id"],
                    confidence=d["evidence"]["confidence"],
                )
            return FacetScore(
                facet_id=d["facet_id"],
                facet_name=d["facet_name"],
                score=d["score"],
                confidence=d["confidence"],
                evidence=ev,
                rationale=d.get("rationale", ""),
                metadata=d.get("metadata", {}),
            )
        except Exception:
            return None

    def _cache_put(self, text: str, fs: FacetScore) -> None:
        ev_dict = None
        if fs.evidence:
            ev_dict = {
                "span": fs.evidence.span,
                "start_char": fs.evidence.start_char,
                "end_char": fs.evidence.end_char,
                "turn_id": fs.evidence.turn_id,
                "confidence": fs.evidence.confidence,
            }
        p = self._cache_dir / f"{self._cache_key(text, fs.facet_id)}.json"
        p.write_text(json.dumps({
            "facet_id":   fs.facet_id,
            "facet_name": fs.facet_name,
            "score":      fs.score,
            "confidence": fs.confidence,
            "rationale":  fs.rationale,
            "metadata":   fs.metadata,
            "evidence":   ev_dict,
        }), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, text: str, categories: List[str]) -> List[Dict]:
        """Hybrid retrieval filtered by routed categories."""
        retrieved = self._retriever.retrieve(text, top_k=self._top_k * 2)

        # Filter to routed categories; fall back to unfiltered if empty
        if set(categories) != _ALL_CATS:
            filtered = [
                r for r in retrieved if r.get("category", "") in categories
            ]
            retrieved = filtered or retrieved

        return retrieved[: self._top_k * 2]

    def _extract_evidences_parallel(
        self,
        text: str,
        facets: List[Dict],
        turn_id: str,
        mentioned_entities: List[str],
    ) -> List[Evidence]:
        """Extract evidence for all facets in parallel."""
        results: List[Optional[Evidence]] = [None] * len(facets)

        def _one(args):
            i, facet = args
            return i, self._extractor.extract(
                text, facet, turn_id, mentioned_entities
            )

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(_one, (i, f)): i
                for i, f in enumerate(facets)
            }
            for fut in as_completed(futures):
                i, ev = fut.result()
                results[i] = ev

        return results

    def _score_facets_parallel(
        self,
        text: str,
        facets: List[Dict],
        evidences: List[Evidence],
        speaker: str,
    ) -> List[FacetScore]:
        """Score all facets in parallel, using cache where available."""
        results: List[Optional[FacetScore]] = [None] * len(facets)
        uncached_jobs = []  # (original_index, facet, evidence)

        # Check cache first
        for i, facet in enumerate(facets):
            cached = self._cache_get(text, facet.get("facet_id", ""))
            if cached is not None:
                results[i] = cached
            else:
                uncached_jobs.append((i, facet, evidences[i]))

        if not uncached_jobs:
            return results

        # Score uncached facets in parallel
        def _score_one(args):
            i, facet, ev = args
            return i, self._scorer.score(text, facet, ev, speaker)

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(_score_one, job): job[0]
                for job in uncached_jobs
            }
            for fut in as_completed(futures):
                i, fs = fut.result()
                results[i] = fs

        return results

    def _score_facets(
        self,
        text: str,
        facets: List[Dict],
        speaker: str,
        mentioned_entities: List[str],
        turn_id: str = "t0",
    ) -> List[FacetScore]:
        """Run Phase 13 → 14 → 15 for each retrieved facet."""

        # Phase 13 — evidence extraction (parallel)
        evidences = self._extract_evidences_parallel(
            text=text,
            facets=facets,
            turn_id=turn_id,
            mentioned_entities=mentioned_entities,
        )

        # Phase 14 — scoring (parallel, cache-aware)
        raw_scores = self._score_facets_parallel(
            text=text,
            facets=facets,
            evidences=evidences,
            speaker=speaker,
        )

        # Phase 15 — confidence calibration
        calibrated_confidences = self._confidence.calibrate_batch(
            facet_scores=raw_scores,
            text=text,
            facets=facets,
            evidences=evidences,
            speaker=speaker,
        )

        # Assemble final scores and write new ones to cache
        final_scores = []
        for i, (fs, cal_conf) in enumerate(zip(raw_scores, calibrated_confidences)):
            scored = FacetScore(
                facet_id=fs.facet_id,
                facet_name=fs.facet_name,
                score=fs.score,
                confidence=round(cal_conf, 4),
                evidence=fs.evidence,
                rationale=fs.rationale,
                metadata=fs.metadata,
            )
            # Only write to cache if this was not a cache hit
            facet_id = facets[i].get("facet_id", "")
            if self._cache_get(text, facet_id) is None:
                self._cache_put(text, scored)
            final_scores.append(scored)

        return final_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        turn: ConversationTurn,
        top_k: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Score a single ConversationTurn across all relevant facets.

        Args:
            turn:  A ConversationTurn with turn_id, speaker, text.
            top_k: Override default number of facet scores to return.

        Returns:
            EvaluationResult with facet_scores sorted by confidence desc.
        """
        k = top_k or self._top_k
        t0 = time.perf_counter()

        # Phase 10 — features
        bundle = self._features.extract(turn)

        # Phase 11 — routing
        categories, _ = self._router.run(turn)

        # Phase 8 — retrieval (no graph expansion)
        retrieved = self._retrieve(turn.text, categories)

        if not retrieved:
            logger.warning(
                f"ScorePipeline: no facets retrieved for turn {turn.turn_id}"
            )
            return EvaluationResult(
                conversation_id="",
                turn_id=turn.turn_id,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        # Phase 13 + 14 + 15 — evidence, scoring, confidence
        facet_scores = self._score_facets(
            text=turn.text,
            facets=retrieved,
            speaker=bundle.speaker,
            mentioned_entities=bundle.mentioned_entities,
            turn_id=turn.turn_id,
        )

        # Sort by confidence desc, take top_k
        facet_scores.sort(key=lambda fs: fs.confidence, reverse=True)
        facet_scores = facet_scores[:k]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = EvaluationResult(
            conversation_id="",
            turn_id=turn.turn_id,
            facet_scores=facet_scores,
            retrieved_facets=[r["facet_name"] for r in retrieved],
            latency_ms=latency_ms,
        )

        logger.info(
            f"ScorePipeline [{turn.turn_id}]: "
            f"{len(facet_scores)} facets scored in {latency_ms:.0f}ms"
        )
        return result

    def score_text(
        self,
        text: str,
        turn_id: str = "t0",
        speaker: str = "user",
        top_k: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Convenience wrapper — accepts raw text instead of ConversationTurn.

        Args:
            text:    Conversation turn text.
            turn_id: Identifier for this turn.
            speaker: Speaker label.
            top_k:   Number of top scores to return.

        Returns:
            EvaluationResult.
        """
        turn = ConversationTurn(turn_id=turn_id, speaker=speaker, text=text)
        return self.score(turn, top_k=top_k)

    def score_conversation(
        self,
        conversation: Conversation,
        top_k: Optional[int] = None,
    ) -> List[EvaluationResult]:
        """
        Score every turn in a Conversation.

        Args:
            conversation: Conversation dataclass with turns list.
            top_k:        Number of top scores per turn.

        Returns:
            List of EvaluationResult, one per turn.
        """
        results = []
        for turn in conversation.turns:
            result = self.score(turn, top_k=top_k)
            result.conversation_id = conversation.conversation_id
            results.append(result)
        return results