"""
Phase 10 — Feature Pipeline
Orchestrates speaker detection, entity extraction, and sentiment extraction
into a single FeatureBundle per ConversationTurn.

This output feeds Phase 11 (Category Router) and Phase 13 (Evidence Extraction).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.types import ConversationTurn
from .speaker_detector import detect_speaker, extract_mentioned_entities
from .entity_extractor import Entity, extract_entities
from .sentiment_extractor import SentimentResult, extract_sentiment

logger = get_logger(__name__)


@dataclass
class FeatureBundle:
    """All features extracted from one ConversationTurn."""
    turn_id:            str
    speaker:            str
    mentioned_entities: List[str]          # entity text spans (not the speaker)
    entities:           List[Entity]       # rich entity objects
    sentiment:          SentimentResult
    token_count:        int
    metadata:           Dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """
    Extract all Phase 10 features from a ConversationTurn in one call.

    Usage::

        pipeline = FeaturePipeline()
        bundle = pipeline.extract(turn)
        print(bundle.speaker, bundle.sentiment.label)
    """

    def extract(self, turn: ConversationTurn) -> FeatureBundle:
        """
        Run all feature extractors on *turn*.

        Args:
            turn: A ConversationTurn dataclass instance.

        Returns:
            FeatureBundle containing speaker, entities, and sentiment.
        """
        text = turn.text or ""

        speaker            = detect_speaker(turn)
        mentioned          = extract_mentioned_entities(text)
        entities           = extract_entities(text)
        sentiment          = extract_sentiment(text)
        token_count        = len(text.split())

        bundle = FeatureBundle(
            turn_id=turn.turn_id,
            speaker=speaker,
            mentioned_entities=mentioned,
            entities=entities,
            sentiment=sentiment,
            token_count=token_count,
        )

        logger.debug(
            f"FeaturePipeline [{turn.turn_id}]: speaker={speaker}, "
            f"entities={len(entities)}, sentiment={sentiment.label}, "
            f"tokens={token_count}"
        )
        return bundle

    def extract_batch(self, turns: List[ConversationTurn]) -> List[FeatureBundle]:
        """Extract features for a list of turns."""
        return [self.extract(t) for t in turns]