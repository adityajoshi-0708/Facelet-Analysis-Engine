"""
Phase 14 — Scoring Engine
Takes a (turn_text, facet, evidence) triple and produces an ordinal score 1–5.

Two-stage strategy (mirrors Phase 13's keyword → LLM pattern):
--------------------------------------------------------------
Stage 1 — Rubric keyword scorer (always runs first):
    Uses the facet's score_anchor strings from facets_enriched.csv.
    Checks which anchor level best matches the evidence span + full text.
    Returns a score if confidence ≥ threshold from config (default 0.6).
    Fast, deterministic, no LLM call.

Stage 2 — LLM scorer (runs when Stage 1 confidence is below threshold):
    Sends a structured rubric prompt to the LLM.
    Prompt includes: facet description, score anchors 1–5, evidence span,
    full turn text.  LLM returns JSON {score, rationale, evidence_span}.
    Score is validated to be in [1, 5].

Output: FacetScore dataclass from src/utils/types.py
    FacetScore(facet_id, facet_name, score, confidence, evidence, rationale)

Design notes:
- Score anchors come from the enriched CSV columns:
  score_1_anchor … score_5_anchor
- The rubric prompt is strict: chain-of-thought is internal to the LLM,
  the output MUST be JSON only (no markdown, no preamble).
- confidence here is a raw rubric-match score, NOT the final calibrated
  confidence — that is computed in Phase 15 (confidence_engine.py).
"""


import json
import re
from typing import Dict, List, Optional, Tuple

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.types import Evidence, FacetScore
from ..models.llm_client import LLMClient, get_llm_client

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rubric prompt template
# ---------------------------------------------------------------------------

_RUBRIC_SYSTEM = (
    "You are a precise psychological facet scorer. "
    "You will be given a facet definition, a 1–5 rubric, an evidence span, "
    "and the full conversation text. "
    "Score the SPEAKER only — not any person mentioned in the text. "
    "Return ONLY a JSON object with exactly three keys: "
    "\"score\" (integer 1-5), "
    "\"rationale\" (one sentence explaining the score), "
    "\"evidence_span\" (the verbatim substring most responsible for your score). "
    "No markdown. No preamble. JSON only."
)

def _safe_parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from LLM output.
    Handles surrounding text, markdown fences, and nested braces."""
    # Strip think tags first with DOTALL
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find outermost { ... }
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError(f"No valid JSON found in: {text[:200]!r}")

def _build_rubric_prompt(
    text: str,
    facet: Dict,
    evidence: Optional[Evidence],
    speaker: str = "user",
) -> str:
    anchors = facet.get("score_anchors", {})

    anchor_lines = []
    for level in range(1, 6):
        anchor_text = (
            anchors.get(level)
            or anchors.get(str(level))
            or f"Level {level}"
        )
        anchor_lines.append(f"  {level}: {anchor_text}")

    evidence_str = (
        f'"{evidence.span}"' if evidence and evidence.span
        else "No specific span identified"
    )

    return (
        f"Facet: {facet.get('facet_name', '')}\n"
        f"Description: {facet.get('description', '')}\n\n"
        f"Scoring Rubric:\n" + "\n".join(anchor_lines) + "\n\n"
        f"Speaker: {speaker}\n"
        f"Evidence span: {evidence_str}\n"
        f"Full text: \"{text}\"\n\n"
        "Score the speaker on this facet. Return JSON only."
    )


# ---------------------------------------------------------------------------
# Stage 1 — Rubric keyword scorer
# ---------------------------------------------------------------------------

def _anchor_keyword_score(
    text: str,
    facet: Dict,
    evidence: Optional[Evidence],
) -> Tuple[int, float]:
    """
    Match text against score anchors using keyword overlap.

    Returns:
        (score 1-5, confidence 0.0-1.0)
        confidence=0.0 means no signal found.
    """
    anchors = facet.get("score_anchors", {})
    if not anchors:
        return 3, 0.0  # no anchors → neutral, zero confidence

    search_text = (
        (evidence.span if evidence and evidence.span else "") + " " + text
    ).lower()

    best_score = 3
    best_overlap = 0
    total_tokens = max(len(search_text.split()), 1)

    for level in range(1, 6):
        anchor_text = (
            anchors.get(level)
            or anchors.get(str(level))
            or ""
        )
        if not anchor_text:
            continue

        anchor_tokens = set(re.findall(r"[a-z]+", anchor_text.lower()))
        text_tokens   = set(re.findall(r"[a-z]+", search_text))
        overlap = len(anchor_tokens & text_tokens)

        if overlap > best_overlap:
            best_overlap = overlap
            best_score = level

    confidence = min(1.0, best_overlap / max(total_tokens * 0.05, 1))
    return best_score, confidence


# ---------------------------------------------------------------------------
# Stage 2 — LLM scorer
# ---------------------------------------------------------------------------

def _llm_score(
    text: str,
    facet: Dict,
    evidence: Optional[Evidence],
    speaker: str,
    client: LLMClient,
) -> Tuple[int, str, float]:
    """
    Ask the LLM to score the facet.

    Returns:
        (score 1-5, rationale str, confidence float)
        Falls back to (3, "LLM failed", 0.3) on any error.
    """
    prompt = _build_rubric_prompt(text, facet, evidence, speaker)

    try:
        resp = client.generate(prompt=prompt, system_prompt=_RUBRIC_SYSTEM)
        data = _safe_parse_json(resp.raw_text)

        score = int(data.get("score", 3))
        score = max(1, min(5, score))   # clamp to [1, 5]

        rationale     = str(data.get("rationale", "")).strip()
        evidence_span = str(data.get("evidence_span", "")).strip()

        # LLM confidence: 0.8 base, boosted slightly if evidence_span found
        confidence = 0.8 if not evidence_span else 0.85

        logger.debug(
            f"LLM scorer: score={score}, conf={confidence}, "
            f"rationale='{rationale[:60]}'"
        )
        return score, rationale, confidence

    except Exception as exc:
        logger.warning(f"ScoringEngine: LLM scoring failed ({exc}) — using neutral fallback")
        return 3, "LLM scoring failed; neutral score assigned.", 0.3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Score a facet for a given conversation turn.

    Args:
        client:              LLMClient.  If None, created via get_llm_client().
        use_llm:             Whether to call LLM when Stage 1 confidence is low.
        confidence_threshold: Minimum Stage 1 confidence to skip LLM (from config).
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        use_llm: bool = True,
        confidence_threshold: Optional[float] = None,
    ):
        cfg = load_config()
        scoring_cfg = cfg.get("scoring", {})

        self._client = client or get_llm_client()
        self._use_llm = use_llm
        self._threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else float(scoring_cfg.get("confidence_threshold", 0.6))
        )

    def score(
        self,
        text: str,
        facet: Dict,
        evidence: Optional[Evidence] = None,
        speaker: str = "user",
    ) -> FacetScore:
        """
        Produce a FacetScore for *facet* given *text* and optional *evidence*.

        Args:
            text:     The speaker's conversation turn text.
            facet:    Dict with facet_id, facet_name, description, score_anchors.
            evidence: Evidence dataclass from Phase 13 (optional).
            speaker:  Speaker identifier (for rubric prompt context).

        Returns:
            FacetScore with score (1-5), confidence, evidence, rationale.
        """
        facet_id   = facet.get("facet_id",   "unknown")
        facet_name = facet.get("facet_name", "Unknown")

        # Stage 1 — rubric keyword
        kw_score, kw_conf = _anchor_keyword_score(text, facet, evidence)

        if kw_conf >= self._threshold or not self._use_llm:
            logger.debug(
                f"ScoringEngine: Stage 1 sufficient for '{facet_name}' "
                f"(score={kw_score}, conf={kw_conf:.2f})"
            )
            return FacetScore(
                facet_id=facet_id,
                facet_name=facet_name,
                score=kw_score,
                confidence=kw_conf,
                evidence=evidence,
                rationale="Scored via rubric keyword match.",
                metadata={"method": "keyword"},
            )

        # Stage 2 — LLM
        logger.debug(
            f"ScoringEngine: Stage 1 confidence {kw_conf:.2f} < {self._threshold} "
            f"for '{facet_name}' — calling LLM"
        )
        llm_score, rationale, llm_conf = _llm_score(
            text, facet, evidence, speaker, self._client
        )

        return FacetScore(
            facet_id=facet_id,
            facet_name=facet_name,
            score=llm_score,
            confidence=llm_conf,
            evidence=evidence,
            rationale=rationale,
            metadata={"method": "llm"},
        )

    # AFTER
    def score_batch(
        self,
        text: str,
        facets: List[Dict],
        evidences: Optional[List[Optional[Evidence]]] = None,
        speaker: str = "user",
    ) -> List[FacetScore]:
        if evidences is None:
            evidences = [None] * len(facets)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(facets)

        def _score_one(args):
            i, facet, ev = args
            return i, self.score(text, facet, ev, speaker)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(_score_one, (i, f, ev)): i
                for i, (f, ev) in enumerate(zip(facets, evidences))
            }
            for fut in as_completed(futures):
                i, fs = fut.result()
                results[i] = fs

        return results