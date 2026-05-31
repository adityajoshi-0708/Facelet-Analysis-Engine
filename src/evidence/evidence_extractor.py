"""
Phase 13 — Evidence Extractor
Finds the exact text span inside a conversation turn that best supports
scoring a given facet.

Two-stage strategy
------------------
Stage 1 — Fast keyword pass (always runs):
    Scan the turn text for the facet's positive_indicators, negative_indicators,
    synonyms, and facet_name tokens.  If a match is found, return it immediately
    without an LLM call.  Covers ~60-70% of cases cheaply.

Stage 2 — LLM extraction (runs only when Stage 1 finds nothing):
    Ask the LLM to return the most relevant substring.  The returned span is
    validated: it MUST be a substring of the original text (LLMs hallucinate
    spans).  If validation fails we fall back to the full sentence containing
    the highest keyword density.

Design rule 2 — Speaker attribution:
    The extractor receives the FeatureBundle from Phase 10.  Any span that
    refers exclusively to a *mentioned entity* (not the speaker) is demoted —
    the evidence confidence is set to 0.3 to signal weak attribution.

Output: Evidence dataclass from src/utils/types.py
    Evidence(span, start_char, end_char, turn_id, confidence)
"""


import re
import json
from typing import Dict, List, Optional, Tuple

from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.types import Evidence
from ..models.llm_client import LLMClient, get_llm_client

logger = get_logger(__name__)

# Confidence assigned to spans found by each method
_CONF_KEYWORD   = 0.85
_CONF_LLM       = 0.75
_CONF_SENTENCE  = 0.55   # fallback: best sentence by keyword density
_CONF_FULL_TEXT = 0.30   # last resort: entire turn text
_CONF_ENTITY    = 0.30   # span refers to mentioned entity, not speaker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

def _normalise(text: str) -> str:
    return text.lower().strip()


def _find_span(text: str, fragment: str) -> Optional[Tuple[int, int]]:
    """
    Case-insensitive substring search.
    Returns (start_char, end_char) or None.
    """
    idx = text.lower().find(fragment.lower())
    if idx == -1:
        return None
    return (idx, idx + len(fragment))


def _sentences(text: str) -> List[str]:
    """Split text into sentences on . ? ! or newline."""
    return [s.strip() for s in re.split(r"[.?!\n]+", text) if s.strip()]


def _keyword_score(sentence: str, keywords: List[str]) -> int:
    """Count how many keywords appear in sentence (case-insensitive)."""
    sl = sentence.lower()
    return sum(1 for kw in keywords if kw.lower() in sl)


# AFTER
def _build_keywords(facet: Dict) -> List[str]:
    kws: List[str] = []

    name = facet.get("facet_name", "")
    kws.append(name)
    kws.extend(name.lower().split())

    # Add meaningful words from description (skip stopwords under 4 chars)
    description = facet.get("description", "")
    if description:
        desc_tokens = re.findall(r"[a-z]{4,}", description.lower())
        kws.extend(desc_tokens)

    for field in ("positive_indicators", "negative_indicators", "synonyms", "related_facets"):
        raw = facet.get(field, [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        kws.extend(str(x) for x in raw)

    return [k for k in kws if len(k) > 2]


# ---------------------------------------------------------------------------
# Stage 1 — keyword pass
# ---------------------------------------------------------------------------

def _keyword_extract(
    text: str,
    keywords: List[str],
    turn_id: str,
) -> Optional[Evidence]:
    """
    Search for the longest keyword that appears verbatim in the text.
    Prefer multi-word phrases over single tokens.
    """
    # Sort by length descending so multi-word phrases win
    sorted_kws = sorted(set(keywords), key=len, reverse=True)

    for kw in sorted_kws:
        span = _find_span(text, kw)
        if span:
            return Evidence(
                span=text[span[0]:span[1]],
                start_char=span[0],
                end_char=span[1],
                turn_id=turn_id,
                confidence=_CONF_KEYWORD,
            )
    return None


# ---------------------------------------------------------------------------
# Stage 2 — LLM extraction
# ---------------------------------------------------------------------------

_EVIDENCE_SYSTEM = (
    "You are an evidence extraction assistant. "
    "Given a facet definition and a text, return ONLY a JSON object with one key: "
    "\"span\" — the shortest substring of the text that best supports scoring "
    "the facet. The span MUST be copied verbatim from the text. "
    "If no relevant span exists, set span to an empty string. "
    "Return ONLY the JSON object, no explanation, no markdown."
)

def _extract_json_safe(raw: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Handles: extra text around JSON, truncated responses, single-quoted keys.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Find the outermost { ... } block
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Last resort: extract span value directly with regex
    span_match = re.search(r'"span"\s*:\s*"([^"]*)"', raw)
    if span_match:
        return {"span": span_match.group(1)}

    return None

def _llm_extract(
    text: str,
    facet: Dict,
    turn_id: str,
    client: LLMClient,
) -> Optional[Evidence]:
    """
    Ask the LLM for an evidence span.  Validates that the returned span
    is a true substring of text before accepting it.
    """
    prompt = (
        f"Facet: {facet.get('facet_name', '')}\n"
        f"Description: {facet.get('description', '')}\n"
        f"Text: {text}\n\n"
        "Return the JSON object with key \"span\"."
    )

    try:
        resp = client.generate(prompt=prompt, system_prompt=_EVIDENCE_SYSTEM)
        raw = resp.raw_text.strip()

        try:
            data = _safe_parse_json(raw)
        except ValueError as parse_err:
            logger.warning(f"EvidenceExtractor: {parse_err}")
            return None
        if data is None:
            logger.warning(
                f"EvidenceExtractor: could not parse JSON. Raw response={raw[:300]}"
            )
            return None

        span_text = data.get("span", "").strip()

        if not span_text:
            return None

        # Validate — span MUST exist verbatim in original text
        loc = _find_span(text, span_text)
        if loc is None:
            logger.warning(
                f"EvidenceExtractor: LLM returned hallucinated span "
                f"'{span_text[:60]}' — falling back to sentence method"
            )
            return None

        return Evidence(
            span=text[loc[0]:loc[1]],
            start_char=loc[0],
            end_char=loc[1],
            turn_id=turn_id,
            confidence=_CONF_LLM,
        )

    except Exception as exc:
        logger.warning(f"EvidenceExtractor: LLM extraction failed ({exc})")
        return None


# ---------------------------------------------------------------------------
# Sentence fallback
# ---------------------------------------------------------------------------

def _sentence_fallback(
    text: str,
    keywords: List[str],
    turn_id: str,
) -> Evidence:
    """
    Return the sentence with the highest keyword density.
    If text has no sentence boundaries, return the full text.
    """
    sents = _sentences(text)
    if not sents:
        return Evidence(
            span=text[:200],
            start_char=0,
            end_char=min(len(text), 200),
            turn_id=turn_id,
            confidence=_CONF_FULL_TEXT,
        )

    best_sent = max(sents, key=lambda s: _keyword_score(s, keywords))
    loc = _find_span(text, best_sent)
    if loc is None:
        loc = (0, min(len(text), 200))

    return Evidence(
        span=text[loc[0]:loc[1]],
        start_char=loc[0],
        end_char=loc[1],
        turn_id=turn_id,
        confidence=_CONF_SENTENCE,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EvidenceExtractor:
    """
    Extract the exact evidence span for a (turn, facet) pair.

    Args:
        client: LLMClient instance.  If None, created via get_llm_client().
        use_llm: Whether to use Stage 2 LLM extraction when Stage 1 fails.
                 Set False in tests to avoid LLM calls.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        use_llm: bool = True,
    ):
        self._client = client or get_llm_client()
        self._use_llm = use_llm

    def extract(
        self,
        text: str,
        facet: Dict,
        turn_id: str = "t0",
        mentioned_entities: Optional[List[str]] = None,
    ) -> Evidence:
        """
        Find the best evidence span for *facet* inside *text*.

        Args:
            text:               Raw turn text (speaker's words only).
            facet:              Dict with facet_name, description,
                                positive_indicators, negative_indicators, synonyms.
            turn_id:            Turn identifier carried into Evidence.
            mentioned_entities: Entity spans from Phase 10 that are NOT
                                the speaker — used to demote confidence.

        Returns:
            Evidence dataclass with span, char offsets, turn_id, confidence.
        """
        if not text.strip():
            return Evidence(
                span="", start_char=0, end_char=0,
                turn_id=turn_id, confidence=0.0,
            )

        keywords = _build_keywords(facet)

        # Stage 1 — keyword pass
        evidence = _keyword_extract(text, keywords, turn_id)

        # Stage 2 — LLM (only if Stage 1 failed)
        

        # Stage 3 — sentence fallback
        if evidence is None:
            evidence = _sentence_fallback(text, keywords, turn_id)

        # Speaker attribution check — demote if span is about a mentioned entity
        if mentioned_entities and evidence.span:
            span_lower = evidence.span.lower()
            for ent in mentioned_entities:
                if ent.lower() in span_lower and len(ent) > 3:
                    logger.debug(
                        f"EvidenceExtractor: span '{evidence.span[:40]}' "
                        f"refers to mentioned entity '{ent}' — demoting confidence"
                    )
                    evidence = Evidence(
                        span=evidence.span,
                        start_char=evidence.start_char,
                        end_char=evidence.end_char,
                        turn_id=evidence.turn_id,
                        confidence=_CONF_ENTITY,
                    )
                    break

        logger.debug(
            f"EvidenceExtractor [{turn_id}] facet='{facet.get('facet_name', '')}': "
            f"span='{evidence.span[:50]}', conf={evidence.confidence}"
        )
        return evidence

    # AFTER
    def extract_batch(
        self,
        text: str,
        facets: List[Dict],
        turn_id: str = "t0",
        mentioned_entities: Optional[List[str]] = None,
    ) -> List[Evidence]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(facets)

        def _extract_one(args):
            i, facet = args
            return i, self.extract(text, facet, turn_id, mentioned_entities)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(_extract_one, (i, f)): i
                for i, f in enumerate(facets)
            }
            for fut in as_completed(futures):
                i, ev = fut.result()
                results[i] = ev

        return results