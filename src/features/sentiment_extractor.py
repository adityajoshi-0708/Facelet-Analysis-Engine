"""
Phase 10 — Sentiment Extractor
Produces a lightweight sentiment signal for a conversation turn:
  - polarity  : float in [-1.0, +1.0]  (negative → positive)
  - magnitude : float in [0.0,  1.0]   (how strong the sentiment is)
  - label     : "positive" | "negative" | "neutral"

Implementation uses a curated lexicon + valence shifters (negation, intensifiers).
No external ML model is required, keeping the feature pipeline fully offline.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    polarity: float   # [-1, +1]
    magnitude: float  # [0, 1]
    label: str        # positive | negative | neutral


# ── Lexicon ────────────────────────────────────────────────────────────────────
# (word, base_score) where score ∈ [-1, +1]

_LEXICON: Dict[str, float] = {
    # Strongly positive
    "love": 0.8, "amazing": 0.85, "excellent": 0.85, "fantastic": 0.9,
    "wonderful": 0.8, "brilliant": 0.8, "outstanding": 0.85, "great": 0.7,
    "happy": 0.7, "joy": 0.75, "excited": 0.7, "proud": 0.65,
    "grateful": 0.7, "blessed": 0.65, "optimistic": 0.6, "hopeful": 0.55,
    "kind": 0.55, "caring": 0.6, "compassionate": 0.65, "generous": 0.6,
    "brave": 0.6, "courageous": 0.65, "honest": 0.55, "authentic": 0.5,
    "passionate": 0.6, "motivated": 0.55, "confident": 0.55,
    "good": 0.5, "nice": 0.45, "better": 0.4, "best": 0.7,
    "success": 0.7, "successful": 0.7, "win": 0.6, "won": 0.6,
    "opportunity": 0.5, "growth": 0.5, "progress": 0.5,
    # Mildly positive
    "okay": 0.15, "fine": 0.15, "decent": 0.2, "reasonable": 0.2,
    # Strongly negative
    "hate": -0.85, "terrible": -0.85, "awful": -0.85, "horrible": -0.85,
    "disgusting": -0.8, "despise": -0.75, "furious": -0.75, "rage": -0.75,
    "angry": -0.65, "sad": -0.65, "depressed": -0.7, "devastated": -0.8,
    "miserable": -0.75, "hopeless": -0.75, "worthless": -0.75,
    "anxious": -0.55, "scared": -0.6, "afraid": -0.6, "worried": -0.5,
    "quit": -0.3, "failed": -0.65, "failure": -0.65, "wrong": -0.5,
    "bad": -0.55, "worse": -0.5, "worst": -0.75, "problem": -0.35,
    "mistake": -0.5, "regret": -0.6, "blame": -0.55, "shame": -0.6,
    "toxic": -0.7, "abusive": -0.8, "manipulative": -0.7,
    # Mildly negative
    "difficult": -0.3, "hard": -0.2, "tough": -0.25, "uncertain": -0.25,
    "risk": -0.2, "dangerous": -0.5, "unsafe": -0.45,
}

_NEGATORS = frozenset(["not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"])
_INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "incredibly": 1.5, "absolutely": 1.4, "quite": 1.15, "rather": 1.1, "somewhat": 0.8, "slightly": 0.7, "a bit": 0.75, "so": 1.2, "really": 1.25, "truly": 1.2, "deeply": 1.3, "totally": 1.3}


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def extract_sentiment(text: str) -> SentimentResult:
    """
    Compute sentiment polarity, magnitude and label for *text*.

    Args:
        text: Raw conversation text.

    Returns:
        SentimentResult with polarity ∈ [-1, 1], magnitude ∈ [0, 1], and label.
    """
    tokens = _tokenise(text)
    if not tokens:
        return SentimentResult(polarity=0.0, magnitude=0.0, label="neutral")

    scores: List[float] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in _LEXICON:
            score = _LEXICON[token]

            # Check preceding window (up to 3 tokens) for negators / intensifiers
            window = tokens[max(0, i - 3): i]
            negated = any(w in _NEGATORS for w in window)
            intensity = 1.0
            for w, factor in _INTENSIFIERS.items():
                if w in " ".join(window):
                    intensity = max(intensity, factor)

            if negated:
                score = -score * 0.8   # flip and dampen
            score *= intensity
            score = max(-1.0, min(1.0, score))
            scores.append(score)

        i += 1

    if not scores:
        return SentimentResult(polarity=0.0, magnitude=0.0, label="neutral")

    polarity  = float(sum(scores) / len(scores))
    magnitude = float(min(1.0, sum(abs(s) for s in scores) / max(len(scores), 1)))

    if polarity >= 0.1:
        label = "positive"
    elif polarity <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    logger.debug(f"Sentiment: '{text[:60]}' → {label} ({polarity:.3f})")
    return SentimentResult(polarity=round(polarity, 4), magnitude=round(magnitude, 4), label=label)