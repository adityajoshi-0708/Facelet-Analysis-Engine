"""
Phase 11 — Routing Pipeline
Connects the FeatureBundle (Phase 10) to the CategoryRouter so that the full
pipeline can call a single method to get routed categories for a turn.
"""

from typing import List, Optional

from ..utils.logger import get_logger
from ..utils.types import ConversationTurn
from ..features.feature_pipeline import FeatureBundle, FeaturePipeline
from .category_router import CategoryRouter

logger = get_logger(__name__)


class RoutingPipeline:
    """
    Extract features from a ConversationTurn and route to relevant categories.

    Usage::

        pipeline = RoutingPipeline()
        categories, bundle = pipeline.run(turn)
        # categories → ["personality", "emotion"]
        # bundle     → full FeatureBundle
    """

    def __init__(
        self,
        min_categories: int = 2,
        max_categories: Optional[int] = None,
    ):
        self._feature_pipeline = FeaturePipeline()
        self._router = CategoryRouter(
            min_categories=min_categories,
            max_categories=max_categories,
        )

    def run(self, turn: ConversationTurn):
        """
        Run feature extraction + category routing for a single turn.

        Args:
            turn: A ConversationTurn instance.

        Returns:
            Tuple (categories: List[str], bundle: FeatureBundle)
        """
        bundle = self._feature_pipeline.extract(turn)
        categories = self._router.route(
            query=turn.text,
            sentiment_label=bundle.sentiment.label,
        )
        logger.debug(
            f"RoutingPipeline [{turn.turn_id}]: "
            f"speaker={bundle.speaker}, categories={categories}"
        )
        return categories, bundle

    def run_text(self, text: str, turn_id: str = "t0", speaker: str = "user"):
        """
        Convenience wrapper — accepts raw text instead of a ConversationTurn.

        Returns:
            Tuple (categories: List[str], bundle: FeatureBundle)
        """
        turn = ConversationTurn(turn_id=turn_id, speaker=speaker, text=text)
        return self.run(turn)