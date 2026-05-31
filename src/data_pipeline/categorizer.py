"""
Phase 2 — Categorizer
======================
Assigns a category to a facet name using a two-stage hybrid strategy:

  STAGE 1 — Keyword fast-path (high precision, O(1) per facet)
      Each category carries a list of substrings.  If the lower-cased
      facet name contains any of them, that category is returned
      immediately — no embedding required.

  STAGE 2 — Embedding cosine-similarity fallback
      When no keyword fires, the facet name is embedded and compared
      against per-category centroid texts (description + seeds).  The
      category with the highest cosine similarity wins, unless it falls
      below SIM_FLOOR, in which case "other" is returned.

      sentence-transformers is used when available; a pure-Python
      character-trigram TF-IDF surrogate is used otherwise so the module
      remains importable in any environment.

Public surface
--------------
    categorize(facet_name, embedder=None, sim_floor=0.18) -> str
    build_category_embedder(model_name)     -> embedder instance
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Category profiles  (description + seeds + high-precision keywords)
# ---------------------------------------------------------------------------

CATEGORY_PROFILES: Dict[str, Dict] = {
    "safety": {
        "description": (
            "Risk, harm, toxicity, manipulation, deception, privacy violations, "
            "bias and fairness concerns present in a conversational turn."
        ),
        "seeds": [
            "toxicity", "harassment", "violence", "self-harm", "manipulation",
            "deception", "bias", "fairness", "privacy", "harmful content",
            "extremism", "misinformation", "hostility", "abuse",
        ],
        "keywords": [
            "toxic", "harm", "violen", "manipulat", "decep", "bias", "privacy",
            "abus", "hostil", "disrespect", "hate", "threat", "suicid", "coerciv",
        ],
    },
    "emotion": {
        "description": (
            "Affect and emotional content: valence, arousal, specific emotions, "
            "emotional regulation, empathy, compassion, warmth."
        ),
        "seeds": [
            "compassion", "empathy", "sympathy", "kindness", "warmth",
            "sadness", "joy", "happiness", "fear", "anger", "grief", "anxiety",
            "moroseness", "pessimism", "optimism", "hope", "despair",
            "compassion fatigue", "burnout", "loneliness", "trust",
        ],
        "keywords": [
            "compassion", "empath", "sympathy", "kindness", "warmth",
            "morose", "pessim", "optimis", "grief", "anxiet", "burnout",
            "lonelin", "discontent", "merriness", "peacefulness",
        ],
    },
    "cognitive": {
        "description": (
            "Thinking processes: reasoning, problem-solving, learning, memory, "
            "creativity, common sense, statistical and numerical reasoning."
        ),
        "seeds": [
            "statistical reasoning", "numerical reasoning", "common sense",
            "curiosity", "creativity", "analytical", "logic", "reasoning",
            "wisdom", "naivety", "gullibility", "critical thinking",
            "metacognition", "learning", "insight", "intuition",
        ],
        "keywords": [
            "reasoning", "logic", "creativ", "common sense", "naiv", "gullib",
            "statistic", "numer", "curiosity", "cogniti", "wisdom", "insight",
            "analytical", "problem solving",
        ],
    },
    "social": {
        "description": (
            "Interpersonal style: leadership, collaboration, cooperation, "
            "conflict, assertiveness, communication, influence."
        ),
        "seeds": [
            "assertiveness", "democratic leadership", "dominance",
            "collaboration", "cooperation", "teamwork", "communication",
            "social influence", "negotiation", "conflict", "authority",
            "submission", "conformity", "empowerment",
        ],
        "keywords": [
            "assertiv", "leadership", "democr", "collabor", "cooperat",
            "teamwork", "communicat", "negoti", "conflict", "dominance",
            "submiss", "conform", "affiliat",
        ],
    },
    "personality": {
        "description": (
            "Stable traits and dispositions: HEXACO/Big-Five facets, character "
            "strengths, virtues, risk-taking, integrity, authenticity."
        ),
        "seeds": [
            "risk taking", "adventure seeking", "honesty", "integrity",
            "authenticity", "courage", "openness", "conscientiousness",
            "agreeableness", "neuroticism", "perfectionism", "resilience",
            "persistence", "discipline", "impulsivity", "boldness",
            "confidence", "humility", "modesty",
        ],
        "keywords": [
            "risk tak", "risk-tak", "adventure", "honesty", "integrity",
            "authentic", "courage", "bravery", "bold", "daring", "resilience",
            "persistence", "humility", "modesty", "conscienti",
        ],
    },
    "clinical_health": {
        "description": (
            "Clinical, medical, or physiological signals: hormone levels, "
            "anatomical knowledge, symptoms, mental-health markers."
        ),
        "seeds": [
            "FSH level", "anatomy knowledge", "symptom", "diagnosis",
            "compulsive activities", "negative affect frequency",
            "spiritual pain", "acidity", "moroseness",
        ],
        "keywords": [
            "fsh", "hormone", "symptom", "diagnos", "clinic",
            "anatomy", "compuls", "acidity", "level",
        ],
    },
    "behavioral_lifestyle": {
        "description": (
            "Habitual behaviour patterns: diet, activities, hobbies, "
            "lifestyle frequencies and counts."
        ),
        "seeds": [
            "eco-tourism trips", "pilgrimage participation count",
            "training-cycle length", "processed-food frequency",
            "adventure-seeking behavior",
        ],
        "keywords": [
            "frequency", "trips", "count", "cycle", "lifestyle",
            "habit", "tourism", "food frequency", "participation",
        ],
    },
    "spirituality_culture": {
        "description": (
            "Spirituality, religion, culture, mindfulness practices, "
            "pilgrimage, sufi practice."
        ),
        "seeds": [
            "mindfulness", "spirituality", "spiritual pain", "pilgrimage",
            "sufi practice", "dhikr", "role of spirituality",
        ],
        "keywords": [
            "spiritual", "pilgrim", "sufi", "dhikr", "mindful",
            "religio", "faith", "meditation",
        ],
    },
    "other": {
        "description": "Catch-all for facets that do not fit any profile.",
        "seeds": ["miscellaneous", "uncategorized"],
        "keywords": [],
    },
}

VALID_CATEGORIES = set(CATEGORY_PROFILES.keys())
SIM_FLOOR_DEFAULT = 0.18


# ---------------------------------------------------------------------------
# Keyword fast-path
# ---------------------------------------------------------------------------

def _keyword_match(name: str) -> Optional[str]:
    """Return category string if any high-precision keyword fires, else None."""
    n = name.lower().strip()
    for cat, profile in CATEGORY_PROFILES.items():
        if cat == "other":
            continue
        for kw in profile.get("keywords", []):
            if kw in n:
                return cat
    return None


# ---------------------------------------------------------------------------
# Lightweight trigram TF-IDF fallback (no external dependencies)
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> List[str]:
    t = re.sub(r"[^a-z0-9 ]", "", text.lower())
    tokens = t.split()
    return [tok[i:i+3] for tok in tokens for i in range(len(tok) - 2)]


def _tfidf_vec(trigrams: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = defaultdict(float)
    for tg in trigrams:
        tf[tg] += 1.0
    n = max(sum(tf.values()), 1)
    return {k: (v / n) * idf.get(k, 1.0) for k, v in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _build_trigram_index() -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Pre-compute per-category trigram vectors and IDF weights."""
    cat_docs: Dict[str, str] = {}
    for cat, profile in CATEGORY_PROFILES.items():
        cat_docs[cat] = profile["description"] + " " + " ".join(profile["seeds"])

    # IDF over category corpus
    df_count: Dict[str, int] = defaultdict(int)
    all_trigrams_per_cat: Dict[str, List[str]] = {}
    for cat, doc in cat_docs.items():
        tg = _trigrams(doc)
        all_trigrams_per_cat[cat] = tg
        for t in set(tg):
            df_count[t] += 1

    N = len(cat_docs)
    idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in df_count.items()}
    cat_vecs = {cat: _tfidf_vec(tg, idf) for cat, tg in all_trigrams_per_cat.items()}
    return cat_vecs, idf


_CAT_VECS, _IDF = _build_trigram_index()


def _trigram_categorize(name: str, sim_floor: float) -> str:
    facet_vec = _tfidf_vec(_trigrams(name), _IDF)
    best_cat, best_sim = "other", 0.0
    for cat, cvec in _CAT_VECS.items():
        if cat == "other":
            continue
        sim = _cosine(facet_vec, cvec)
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    return best_cat if best_sim >= sim_floor else "other"


# ---------------------------------------------------------------------------
# Sentence-transformer path (optional, higher quality)
# ---------------------------------------------------------------------------

def build_category_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Return a (embedder, cat_vecs_np, cat_keys) triple for batch use.
    Raises ImportError if sentence-transformers is not installed.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)
    cat_keys = [c for c in CATEGORY_PROFILES if c != "other"]
    cat_texts = [
        CATEGORY_PROFILES[c]["description"] + " " + " ".join(CATEGORY_PROFILES[c]["seeds"])
        for c in cat_keys
    ]
    cat_vecs = embedder.encode(cat_texts, normalize_embeddings=True)
    return embedder, np.array(cat_vecs), cat_keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize(
    facet_name: str,
    embedder=None,
    cat_vecs=None,
    cat_keys: Optional[List[str]] = None,
    sim_floor: float = SIM_FLOOR_DEFAULT,
) -> str:
    """
    Return the category string for *facet_name*.

    Parameters
    ----------
    facet_name : str
        Human-readable facet name, e.g. "Risk Taking".
    embedder : optional SentenceTransformer (or compatible) instance.
        When provided together with *cat_vecs* and *cat_keys*, embedding
        similarity is used as the fallback instead of trigram TF-IDF.
    cat_vecs : np.ndarray, shape (n_cats, dim)  — pre-computed L2-normalised
        category vectors.  Required when *embedder* is set.
    cat_keys : list[str] matching rows in *cat_vecs*.
    sim_floor : float
        Minimum similarity for an embedding match to be trusted.

    Returns
    -------
    str : one of the keys in CATEGORY_PROFILES.
    """
    # Stage 1 — fast keyword match
    cat = _keyword_match(facet_name)
    if cat is not None:
        return cat

    # Stage 2 — embedding or trigram fallback
    if embedder is not None and cat_vecs is not None and cat_keys is not None:
        import numpy as np

        vec = embedder.encode([facet_name], normalize_embeddings=True)
        sims = (vec @ cat_vecs.T)[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        return cat_keys[best_idx] if best_sim >= sim_floor else "other"

    return _trigram_categorize(facet_name, sim_floor)