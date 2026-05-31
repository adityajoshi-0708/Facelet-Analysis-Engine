"""
Phase 10 — Entity Extractor
Identifies named entities mentioned in conversation text.
Uses rule-based patterns (no external NLP model required) to detect:
  - Person names (proper noun patterns)
  - Organisations
  - Roles / relational entities (my boss, a colleague, the team)
  - Possessive references (my friend, his manager)

These are *mentioned* entities — they are NOT the speaker and must NOT be scored.
"""

import re
from dataclasses import dataclass, field
from typing import List

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    text: str
    label: str          # PERSON | ORG | ROLE | OTHER
    start_char: int
    end_char: int


# ── Compiled patterns ──────────────────────────────────────────────────────────

# Relational / role mentions ("my friend", "her boss", "the team")
_RELATIONAL = re.compile(
    r"\b(?:my|his|her|their|our|the|a|an)\s+"
    r"(?:friend|colleague|coworker|boss|manager|partner|spouse|husband|wife|"
    r"child|son|daughter|parent|mother|father|sister|brother|team|group|"
    r"client|customer|employee|staff|teacher|doctor|therapist|mentor)\b",
    re.IGNORECASE,
)

# Simple capitalised name heuristic: "John", "Sarah Connors" etc.
# Excludes sentence-start false positives by requiring ≥2 chars + not after period
_PROPER_NAME = re.compile(r"(?<![.?!]\s)\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})?)\b")

# Organisation-ish tokens (ALL CAPS acronyms ≥ 2 chars, or "Corp/Inc/Ltd" suffixes)
_ORG = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:\s+(?:Corp|Inc|Ltd|LLC|Co|Group|Solutions|Technologies))?)\b"
)

# Third-person pronouns used to refer to someone else
_PRONOUNS = re.compile(r"\b(he|she|they|him|her|them|his|their)\b", re.IGNORECASE)


def extract_entities(text: str) -> List[Entity]:
    """
    Extract mentioned entities from *text*.

    Args:
        text: Raw conversation text from a single turn.

    Returns:
        List of Entity objects (possibly empty).
    """
    entities: List[Entity] = []

    # Relational mentions (highest precision)
    for m in _RELATIONAL.finditer(text):
        entities.append(Entity(
            text=m.group(),
            label="ROLE",
            start_char=m.start(),
            end_char=m.end(),
        ))

    # Proper names
    for m in _PROPER_NAME.finditer(text):
        # Skip if already covered by a relational match
        if not any(e.start_char <= m.start() < e.end_char for e in entities):
            entities.append(Entity(
                text=m.group(),
                label="PERSON",
                start_char=m.start(),
                end_char=m.end(),
            ))

    # De-duplicate by span
    seen_spans = set()
    unique: List[Entity] = []
    for e in sorted(entities, key=lambda x: x.start_char):
        span = (e.start_char, e.end_char)
        if span not in seen_spans:
            seen_spans.add(span)
            unique.append(e)

    logger.debug(f"EntityExtractor: found {len(unique)} entities in '{text[:60]}'")
    return unique