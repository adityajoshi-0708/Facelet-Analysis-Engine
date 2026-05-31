"""
Phase 10 — Speaker Detector
Determines who the *speaker* of a ConversationTurn is and which entities are
merely *mentioned* in the text.  The scoring system should evaluate the
speaker, not mentioned entities (Design Rule 2).

Detection strategy:
  1. Explicit speaker tag on the turn (turn.speaker field) is authoritative.
  2. Fallback heuristics for raw free-text without metadata.
"""

import re
from typing import List, Optional

from ..utils.logger import get_logger
from ..utils.types import ConversationTurn

logger = get_logger(__name__)

# Pronouns that imply the speaker is talking about themselves
_FIRST_PERSON = frozenset(
    ["i", "i'm", "i've", "i'll", "i'd", "me", "my", "myself", "mine", "we", "our", "us"]
)

# Named-entity mention patterns (simple heuristic — real NER is in entity_extractor.py)
_THIRD_PERSON_INDICATORS = re.compile(
    r"\b(he|she|they|his|her|their|him|them|my\s+\w+|a\s+friend|the\s+team)\b",
    re.IGNORECASE,
)


def detect_speaker(turn: ConversationTurn) -> str:
    """
    Return the canonical speaker identifier for a conversation turn.

    Priority order:
    1. turn.speaker if it is a non-empty, non-generic string → use as-is.
    2. First-person pronoun heuristic → "user"
    3. Default → turn.speaker or "unknown"

    Args:
        turn: A ConversationTurn dataclass instance.

    Returns:
        Speaker string, e.g. "user", "assistant", "system", or a custom name.
    """
    explicit = (turn.speaker or "").strip().lower()
    if explicit and explicit not in ("", "unknown"):
        return explicit

    # Heuristic: count first-person tokens in the text
    tokens = re.findall(r"\b\w[\w']*\b", turn.text.lower())
    first_person_count = sum(1 for t in tokens if t in _FIRST_PERSON)
    if first_person_count > 0:
        return "user"

    return "unknown"


def extract_mentioned_entities(text: str) -> List[str]:
    """
    Return a list of third-person entity mentions in the text.
    These are *not* the speaker and should *not* be scored.

    This is a lightweight heuristic.  Phase 10's full entity extractor
    (entity_extractor.py) handles richer NER.

    Args:
        text: Raw conversation text.

    Returns:
        List of matched mention spans.
    """
    matches = _THIRD_PERSON_INDICATORS.findall(text)
    entities = [m if isinstance(m, str) else m[0] for m in matches]
    return list(dict.fromkeys(entities))  # deduplicate, preserve order