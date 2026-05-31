from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Facet:
    facet_id: str
    facet_name: str
    category: str
    description: str = ""
    positive_indicators: List[str] = field(default_factory=list)
    negative_indicators: List[str] = field(default_factory=list)
    score_anchors: Dict[int, str] = field(default_factory=dict)  # {1: "...", 5: "..."}
    synonyms: List[str] = field(default_factory=list)
    related_facets: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    turn_id: str
    speaker: str  # "user" | "assistant" | "system"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    span: str
    start_char: int
    end_char: int
    turn_id: str
    confidence: float = 1.0


@dataclass
class FacetScore:
    facet_id: str
    facet_name: str
    score: int  # 1-5
    confidence: float  # 0.0-1.0
    evidence: Optional[Evidence] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    conversation_id: str
    turn_id: str
    facet_scores: List[FacetScore] = field(default_factory=list)
    retrieved_facets: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)