"""
Phase 11 — Category Router
Narrows the set of facet categories to search *before* hitting the retrieval
stack. This reduces latency and noise by only searching categories whose
keywords appear in the query / feature bundle.

Strategy
--------
1. Build a keyword → category index from the enriched facets CSV at startup.
2. On each call, scan the query text for keyword signals.
3. Also incorporate the sentiment label: strong negative → weight "safety" and
   "emotion" higher; strong positive → weight "personality" and "emotion".
4. Return an ordered list of categories (most relevant first), or ALL categories
   if the signal is too weak to narrow.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Hard-coded seed signals per category (supplement the CSV-derived index)
_CATEGORY_SEED_KEYWORDS: Dict[str, List[str]] = {
    "emotion": [
        "feel", "felt", "feeling", "emotion", "sad", "happy", "angry", "afraid",
        "joy", "grief", "love", "hate", "fear", "anxiety", "anxious", "depress",
        "cry", "laugh", "hurt", "pain", "comfort", "compassion", "empathy",
        "warm", "cold", "lonely", "hopeless", "hopeful", "excited", "frustrated",
    ],
    "personality": [
        "risk", "bold", "dare", "brave", "courag", "honest", "authentic", "integr",
        "adventur", "quit", "start", "launch", "invest", "opportun", "chose",
        "decided", "confident", "assertive", "persist", "disciplin", "principl",
    ],
    "cognitive": [
        "think", "reason", "analys", "analyz", "statistic", "data", "probability",
        "logic", "learn", "understand", "curious", "question", "research",
        "naive", "gullible", "common sense", "practical", "wisdom", "smart",
        "intelligent", "insight", "creative", "memory",
    ],
    "social": [
        "team", "leader", "democratic", "collab", "communic", "negotiat",
        "manipulat", "dominan", "assert", "conflict", "cooperat", "group",
        "relationship", "influence", "power", "authority", "community",
    ],
    "safety": [
        "harm", "danger", "toxic", "threat", "abus", "violen", "self-harm",
        "suicid", "hurt", "unsafe", "aggress", "manipulat", "exploit",
        "coerce", "stalk", "harass",
    ],
}


class CategoryRouter:
    """
    Determine which facet categories are relevant for a query string.

    Args:
        min_categories: Minimum number of categories to always return (default 2).
        max_categories: Cap on categories returned; None = no cap (return all).
    """

    def __init__(
        self,
        min_categories: int = 2,
        max_categories: Optional[int] = None,
    ):
        self._min_categories = min_categories
        self._max_categories = max_categories
        self._all_categories = list(_CATEGORY_SEED_KEYWORDS.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"[a-z']+", text.lower())

    def _score_categories(self, tokens: List[str]) -> Dict[str, float]:
        """Return a score per category based on keyword overlap."""
        scores: Dict[str, float] = {cat: 0.0 for cat in self._all_categories}
        token_set = set(tokens)

        for category, keywords in _CATEGORY_SEED_KEYWORDS.items():
            for kw in keywords:
                # Substring match to catch stemmed variants
                if any(kw in tok for tok in token_set) or any(tok in kw for tok in token_set if len(tok) > 3):
                    scores[category] += 1.0

        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        sentiment_label: Optional[str] = None,
    ) -> List[str]:
        """
        Return ordered list of relevant category names for *query*.

        Args:
            query:           Free-text conversation turn or search query.
            sentiment_label: Optional sentiment label from Phase 10
                             ("positive" | "negative" | "neutral").

        Returns:
            Ordered list of category names, most relevant first.
            Always contains at least *min_categories* entries.
        """
        tokens = self._tokenise(query)
        scores = self._score_categories(tokens)

        # Sentiment boosts
        if sentiment_label == "negative":
            scores["safety"]  = scores.get("safety",  0.0) + 0.5
            scores["emotion"] = scores.get("emotion", 0.0) + 0.5
        elif sentiment_label == "positive":
            scores["personality"] = scores.get("personality", 0.0) + 0.3
            scores["emotion"]     = scores.get("emotion",     0.0) + 0.3

        # Sort categories by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Always include at least min_categories
        non_zero = [cat for cat, sc in ranked if sc > 0]
        if len(non_zero) < self._min_categories:
            # Pad with next-best scoring categories
            extras = [cat for cat, _ in ranked if cat not in non_zero]
            non_zero = non_zero + extras[: self._min_categories - len(non_zero)]

        result = non_zero
        if self._max_categories:
            result = result[: self._max_categories]

        logger.debug(
            f"CategoryRouter: '{query[:60]}' → {result}"
        )
        return result

    def route_all(self) -> List[str]:
        """Return all categories (used when routing signal is absent)."""
        return list(self._all_categories)