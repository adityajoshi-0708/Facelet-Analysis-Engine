"""
Phase 15 — Confidence Engine
Calibrates the raw confidence from Phase 14 into a final, well-grounded
confidence score using one of three methods depending on what is available:

Method A — Logprob entropy (best, when LLM returns logprobs):
    Read the per-token logprobs at the score-token position.
    Build a distribution over the 5 score literals ("1".."5").
    confidence = 1 - H_5(p) / log(5)    ∈ [0, 1]
    where H_5 is Shannon entropy over 5 classes.

Method B — Self-consistency (when logprobs unavailable):
    Run the LLM `self_consistency_runs` times at temperature 0.7.
    Collect the score from each run and build an empirical distribution.
    confidence = 1 - H_5(empirical_p) / log(5)
    Configured via configs/scoring.yaml → self_consistency_runs (default 3).

Method C — Pass-through (when LLM not used / mock):
    When the score came from Stage 1 keyword matching or the mock client,
    the raw confidence from Phase 14 is used directly after clamping to [0,1].

This module is ONLY responsible for confidence.  It does not change the score.

Output: float ∈ [0.0, 1.0]
"""

import math
from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.types import Evidence, FacetScore
from ..models.llm_client import LLMClient, LLMResponse, get_llm_client
from ..scoring.rubric_engine import _build_rubric_prompt, _RUBRIC_SYSTEM

logger = get_logger(__name__)

_LOG5 = math.log(5)   # normalisation constant for 5-class entropy


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _entropy(probs: List[float]) -> float:
    """Shannon entropy H(p) in nats."""
    return -sum(p * math.log(p + 1e-12) for p in probs)


def _normalised_entropy(probs: List[float]) -> float:
    """H(p) / log(5) ∈ [0, 1].  0 = certain, 1 = maximally uncertain."""
    return _entropy(probs) / _LOG5


def _confidence_from_probs(probs: List[float]) -> float:
    """1 - normalised_entropy, clamped to [0, 1]."""
    return float(max(0.0, min(1.0, 1.0 - _normalised_entropy(probs))))


# ---------------------------------------------------------------------------
# Method A — logprob entropy
# ---------------------------------------------------------------------------

_SCORE_TOKENS = {"1", "2", "3", "4", "5"}


def _confidence_from_logprobs(
    logprobs: List[Dict],
    predicted_score: int,
) -> Optional[float]:
    """
    Extract confidence from logprobs returned by the LLM.

    Looks for the first token in the logprob list that is one of "1".."5"
    (the score token).  Builds a distribution by softmax over its top-k
    logprobs restricted to score tokens.

    Returns None if the logprob structure is unrecognisable.
    """
    try:
        # Ollama returns logprobs as list of {token, logprob} dicts
        # We find the score token position
        score_logits: Dict[str, float] = {}

        for entry in logprobs:
            tok = str(entry.get("token", "")).strip()
            lp  = float(entry.get("logprob", 0.0))
            if tok in _SCORE_TOKENS:
                score_logits[tok] = lp

        if not score_logits:
            return None

        # Fill missing score tokens with a very low logprob
        min_lp = min(score_logits.values()) - 10.0
        full_logits = [score_logits.get(str(i), min_lp) for i in range(1, 6)]
        probs = _softmax(full_logits)

        conf = _confidence_from_probs(probs)
        logger.debug(f"ConfidenceEngine: logprob method → conf={conf:.3f}")
        return conf

    except Exception as exc:
        logger.warning(f"ConfidenceEngine: logprob parsing failed ({exc})")
        return None


# ---------------------------------------------------------------------------
# Method B — self-consistency
# ---------------------------------------------------------------------------

def _confidence_from_self_consistency(
    text: str,
    facet: Dict,
    evidence: Optional[Evidence],
    speaker: str,
    client: LLMClient,
    runs: int,
) -> float:
    """
    Run `runs` independent LLM generations and compute empirical score
    distribution.  Uses temperature 0.7 for diversity.
    """
    import re, json

    prompt = _build_rubric_prompt(text, facet, evidence, speaker)
    counts = {i: 0 for i in range(1, 6)}
    successful = 0

    # We need a way to run with higher temperature — patch temporarily
    original_temp = getattr(client, "temperature", None)
    if original_temp is not None:
        client.temperature = 0.7  # type: ignore

    for i in range(runs):
        try:
            resp: LLMResponse = client.generate(
                prompt=prompt, system_prompt=_RUBRIC_SYSTEM
            )
            raw = resp.raw_text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            data = json.loads(raw)
            s = int(data.get("score", 3))
            s = max(1, min(5, s))
            counts[s] += 1
            successful += 1
        except Exception:
            pass

    # Restore temperature
    if original_temp is not None:
        client.temperature = original_temp  # type: ignore

    if successful == 0:
        logger.warning("ConfidenceEngine: all self-consistency runs failed")
        return 0.3

    total = sum(counts.values())
    probs = [counts[i] / total for i in range(1, 6)]
    conf = _confidence_from_probs(probs)

    logger.debug(
        f"ConfidenceEngine: self-consistency ({runs} runs) → "
        f"counts={counts}, conf={conf:.3f}"
    )
    return conf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Multi-source confidence combiner
# ---------------------------------------------------------------------------

def _combine_confidence(
    llm_conf: float,
    evidence: Optional["Evidence"],
    facet_score: "FacetScore",
) -> float:
    """
    Combine three independent confidence signals into a single calibrated score.

    Formula:
        final = 0.5 * llm_conf + 0.3 * evidence_conf + 0.2 * retrieval_conf

    - llm_conf:       from logprobs / self-consistency / passthrough (primary)
    - evidence_conf:  from Evidence.confidence (keyword=0.85, llm=0.75,
                      sentence=0.55, entity/none=0.30)
    - retrieval_conf: RRF score stored in FacetScore.metadata["retrieval_score"],
                      normalised to [0,1]. Defaults to 0.5 if absent.
    """
    # Evidence confidence — use 0.5 as neutral when no evidence
    evidence_conf = evidence.confidence if (evidence and evidence.span) else 0.5

    # Retrieval score — RRF scores are typically 0.001–0.02; normalise to [0,1]
    raw_retrieval = float(facet_score.metadata.get("retrieval_score", 0.0))
    # Clamp: RRF scores above 0.02 are very strong; scale so 0.02 → 1.0
    retrieval_conf = min(1.0, raw_retrieval / 0.02) if raw_retrieval > 0 else 0.5

    combined = (
        0.5 * llm_conf
        + 0.3 * evidence_conf
        + 0.2 * retrieval_conf
    )
    return float(max(0.0, min(1.0, combined)))
class ConfidenceEngine:
    """
    Calibrate confidence for a FacetScore produced by Phase 14.

    Args:
        client:    LLMClient.  If None, created via get_llm_client().
        use_llm:   Whether to attempt logprob / self-consistency methods.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        use_llm: bool = True,
    ):
        cfg = load_config()
        scoring_cfg = cfg.get("scoring", {})

        self._client   = client or get_llm_client()
        self._use_llm  = use_llm
        self._sc_runs: int = int(scoring_cfg.get("self_consistency_runs", 3))
        self._use_logprobs: bool = bool(scoring_cfg.get("use_logprobs", True))
        self._use_entropy:  bool = bool(scoring_cfg.get("use_entropy",  True))

    # AFTER
    def calibrate(
        self,
        facet_score: FacetScore,
        llm_response: Optional[LLMResponse] = None,
        text: str = "",
        facet: Optional[Dict] = None,
        evidence: Optional[Evidence] = None,
        speaker: str = "user",
    ) -> float:
        # Method A — logprobs
        if (
            self._use_logprobs
            and llm_response is not None
            and llm_response.logprobs
        ):
            llm_conf = _confidence_from_logprobs(
                llm_response.logprobs, facet_score.score
            )
            if llm_conf is not None:
                combined = _combine_confidence(llm_conf, evidence, facet_score)
                logger.debug(f"ConfidenceEngine: logprob+combine → {combined:.3f}")
                return combined

        # Method B — self-consistency (only when no logprobs AND evidence is weak)
        evidence_conf = evidence.confidence if (evidence and evidence.span) else 0.0
        run_self_consistency = (
            self._use_llm
            and self._use_entropy
            and text
            and facet is not None
            and evidence_conf < 0.75   # skip if evidence is already strong
        )
        if run_self_consistency:
            llm_conf = _confidence_from_self_consistency(
                text=text,
                facet=facet,
                evidence=evidence,
                speaker=speaker,
                client=self._client,
                runs=self._sc_runs,
            )
            combined = _combine_confidence(llm_conf, evidence, facet_score)
            logger.debug(f"ConfidenceEngine: self-consistency+combine → {combined:.3f}")
            return combined

        # Method C — pass-through + combine
        llm_conf = float(max(0.0, min(1.0, facet_score.confidence)))
        combined = _combine_confidence(llm_conf, evidence, facet_score)
        logger.debug(f"ConfidenceEngine: passthrough+combine → {combined:.3f}")
        return combined

    def calibrate_batch(
        self,
        facet_scores: List[FacetScore],
        llm_responses: Optional[List[Optional[LLMResponse]]] = None,
        text: str = "",
        facets: Optional[List[Dict]] = None,
        evidences: Optional[List[Optional[Evidence]]] = None,
        speaker: str = "user",
    ) -> List[float]:
        """
        Calibrate confidence for a list of FacetScores.

        Returns:
            List of confidence floats, one per FacetScore.
        """
        n = len(facet_scores)
        if llm_responses is None:
            llm_responses = [None] * n
        if facets is None:
            facets = [{}] * n
        if evidences is None:
            evidences = [None] * n

        return [
            self.calibrate(
                facet_score=fs,
                llm_response=lr,
                text=text,
                facet=f,
                evidence=ev,
                speaker=speaker,
            )
            for fs, lr, f, ev in zip(facet_scores, llm_responses, facets, evidences)
        ]