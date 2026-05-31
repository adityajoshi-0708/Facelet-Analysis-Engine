"""
server.py — Ahoum FastAPI Backend
==================================
Endpoints:
    GET  /health                  liveness + backend tag
    GET  /facets                  list every enriched facet (paginated)
    GET  /facets/{facet_id}       full record for one facet
    POST /retrieve                top-K facets for a free-text query
    POST /score                   score a full conversation
    POST /score/turn              score a single turn

Run:
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ahoum — Personality Facet Scoring API",
    description=(
        "Retrieve and score psychological / behavioural facets "
        "from conversational text using dense + BM25 hybrid retrieval "
        "and an ordinal LLM scorer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first request
# ---------------------------------------------------------------------------

_pipeline = None
_retriever = None
_facets_df: Optional[pd.DataFrame] = None
_ENRICHED_PATH = Path("data/processed/facets_enriched.csv")
_START_TIME = time.time()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.scoring.score_pipeline import ScorePipeline
        _pipeline = ScorePipeline(use_llm=True)
    return _pipeline


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.models.embedding_client import EmbeddingClient
        from src.retrieval.hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever(client=EmbeddingClient())
    return _retriever


def _get_facets_df() -> pd.DataFrame:
    global _facets_df
    if _facets_df is None:
        if not _ENRICHED_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Enriched facets not found at {_ENRICHED_PATH}. "
                       "Run the data pipeline first.",
            )
        _facets_df = pd.read_csv(_ENRICHED_PATH)
    return _facets_df


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ConversationTurnIn(BaseModel):
    speaker: str = Field("user", description="'user' | 'assistant'")
    text: str = Field(..., description="Raw turn text")


class ConversationIn(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation ID")
    turns: List[ConversationTurnIn]


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Free-text query")
    top_k: int = Field(20, ge=1, le=100)


class ScoreTurnRequest(BaseModel):
    text: str = Field(..., description="Single turn text to score")
    turn_id: str = Field("t0")
    speaker: str = Field("user")
    top_k: int = Field(20, ge=1, le=100)


class FacetOut(BaseModel):
    facet_id: str
    facet_name: str
    category: str
    description: str
    score: Optional[float] = None
    confidence: Optional[float] = None
    evidence_span: Optional[str] = None
    rationale: Optional[str] = None


class HealthOut(BaseModel):
    status: str
    uptime_seconds: float
    backend: str
    facets_loaded: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    d = row.to_dict()
    for col in ("positive_indicators", "negative_indicators",
                "synonyms", "related_facets", "examples", "keywords"):
        if col in d and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                d[col] = []
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthOut, tags=["System"])
def health():
    """Liveness check — always fast, no model load."""
    try:
        df = _get_facets_df()
        n = len(df)
    except Exception:
        n = 0

    try:
        from src.models.llm_client import get_llm_client
        client = get_llm_client()
        backend = getattr(client, "model_name", "unknown")
    except Exception:
        backend = "unavailable"

    return HealthOut(
        status="ok",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        backend=backend,
        facets_loaded=n,
    )


@app.get("/facets", tags=["Facets"])
def list_facets(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """
    List all enriched facets with pagination.

    Returns facet_id, facet_name, category, description for each facet.
    """
    df = _get_facets_df()

    if category:
        df = df[df["category"].str.lower() == category.lower()]
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No facets found for category '{category}'"
            )

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
        "facets": [
            {
                "facet_id":   row["facet_id"],
                "facet_name": row["facet_name"],
                "category":   row.get("category", ""),
                "description": str(row.get("description", ""))[:200],
            }
            for _, row in page_df.iterrows()
        ],
    }


@app.get("/facets/{facet_id}", tags=["Facets"])
def get_facet(facet_id: str):
    """
    Full record for a single facet by its stable slug ID.
    """
    df = _get_facets_df()
    matches = df[df["facet_id"] == facet_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Facet '{facet_id}' not found")
    return _row_to_dict(matches.iloc[0])


@app.post("/retrieve", tags=["Retrieval"])
def retrieve(req: RetrieveRequest):
    """
    Return the top-K facets most relevant to a free-text query
    using hybrid dense + BM25 retrieval.
    """
    retriever = _get_retriever()
    t0 = time.perf_counter()
    results = retriever.retrieve(req.query, top_k=req.top_k)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query": req.query,
        "top_k": req.top_k,
        "latency_ms": latency_ms,
        "results": [
            {
                "rank":       r["rank"],
                "facet_id":   r["facet_id"],
                "facet_name": r["facet_name"],
                "category":   r.get("category", ""),
                "score":      round(r["score"], 6),
            }
            for r in results
        ],
    }


@app.post("/score/turn", tags=["Scoring"])
def score_turn(req: ScoreTurnRequest):
    """
    Score a single conversation turn against the top-K retrieved facets.
    Returns facet scores with confidence, evidence span, and rationale.
    """
    pipeline = _get_pipeline()
    t0 = time.perf_counter()
    result = pipeline.score_text(
        text=req.text,
        turn_id=req.turn_id,
        speaker=req.speaker,
        top_k=req.top_k,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "turn_id":    result.turn_id,
        "latency_ms": latency_ms,
        "facet_scores": [
            {
                "facet_id":     fs.facet_id,
                "facet_name":   fs.facet_name,
                "score":        fs.score,
                "confidence":   fs.confidence,
                "rationale":    fs.rationale,
                "evidence_span": fs.evidence.span if fs.evidence else None,
            }
            for fs in result.facet_scores
        ],
    }


@app.post("/score", tags=["Scoring"])
def score_conversation(conv: ConversationIn):
    """
    Score every turn in a full conversation.
    Returns per-turn facet scores plus an aggregate summary.
    """
    from src.utils.types import Conversation, ConversationTurn

    pipeline = _get_pipeline()
    t0 = time.perf_counter()

    conversation = Conversation(
        conversation_id=conv.conversation_id,
        turns=[
            ConversationTurn(turn_id=f"t{i}", speaker=t.speaker, text=t.text)
            for i, t in enumerate(conv.turns)
        ],
    )

    results = pipeline.score_conversation(conversation)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Build aggregate: mean score per facet across all turns
    facet_agg: Dict[str, List[float]] = {}
    turn_outputs = []
    for res in results:
        turn_fs = []
        for fs in res.facet_scores:
            facet_agg.setdefault(fs.facet_name, []).append(fs.score)
            turn_fs.append({
                "facet_id":     fs.facet_id,
                "facet_name":   fs.facet_name,
                "score":        fs.score,
                "confidence":   fs.confidence,
                "rationale":    fs.rationale,
                "evidence_span": fs.evidence.span if fs.evidence else None,
            })
        turn_outputs.append({
            "turn_id":      res.turn_id,
            "latency_ms":   res.latency_ms,
            "facet_scores": turn_fs,
        })

    summary = {
        fname: round(sum(scores) / len(scores), 3)
        for fname, scores in sorted(
            facet_agg.items(), key=lambda x: -sum(x[1]) / len(x[1])
        )
    }

    return {
        "conversation_id": conv.conversation_id,
        "total_latency_ms": latency_ms,
        "n_turns": len(results),
        "summary": summary,
        "turns": turn_outputs,
    }