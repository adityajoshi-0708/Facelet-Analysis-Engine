"""
Phase 2 — Step 2: Facet Enrichment
====================================
For every cleaned facet we attach:

    facet_id            from clean.py
    facet_name          from clean.py
    raw_name            from clean.py
    category            hybrid keyword + embedding categorisation
    description         rich, facet-specific definition paragraph
    positive_indicators real-world behavioural signals (scored high)
    negative_indicators real-world behavioural signals (scored low)
    score_1_anchor      rubric anchor text for score 1
    score_2_anchor      rubric anchor text for score 2
    score_3_anchor      rubric anchor text for score 3
    score_4_anchor      rubric anchor text for score 4
    score_5_anchor      rubric anchor text for score 5
    synonyms            semantically related terms (for BM25 + embedding)
    related_facets      nearby facets in the same category
    examples            concrete conversation snippets that evidence the facet
    keywords            JSON list for hybrid BM25 retrieval
    retrieval_text      dense paragraph for FAISS embedding (the key field)

Categorisation strategy (deterministic, no LLM by default):
    1. Keyword fast-path  — high precision, O(1)
    2. Embedding argmax   — sentence-transformers when available
    3. Trigram TF-IDF     — pure-Python fallback, no dependencies

Run:
    python -m src.data_pipeline.enrich
    python -m src.data_pipeline.enrich --use-llm   # requires Ollama/vLLM
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data_pipeline.categorizer import (
    CATEGORY_PROFILES,
    categorize,
    build_category_embedder,
)

# ---------------------------------------------------------------------------
# Per-category synonym banks  (used to expand weak/short facet names)
# ---------------------------------------------------------------------------

_CATEGORY_SYNONYM_SEEDS: Dict[str, List[str]] = {
    "emotion": [
        "emotional expression", "affective state", "feeling tone",
        "mood indicator", "sentiment",
    ],
    "personality": [
        "trait", "disposition", "character quality", "behavioural tendency",
        "personal attribute",
    ],
    "cognitive": [
        "thinking style", "mental process", "reasoning pattern",
        "intellectual tendency", "cognitive bias",
    ],
    "social": [
        "interpersonal behaviour", "social skill", "relational dynamic",
        "group behaviour", "communication style",
    ],
    "safety": [
        "risk signal", "harm indicator", "safety concern",
        "red flag", "potentially dangerous behaviour",
    ],
    "clinical_health": [
        "clinical marker", "health indicator", "physiological signal",
        "symptom pattern", "medical sign",
    ],
    "behavioral_lifestyle": [
        "lifestyle habit", "routine behaviour", "daily pattern",
        "habitual action", "activity frequency",
    ],
    "spirituality_culture": [
        "spiritual practice", "cultural expression", "religious behaviour",
        "mindfulness pattern", "faith indicator",
    ],
    "other": ["general attribute", "miscellaneous quality"],
}

# ---------------------------------------------------------------------------
# Per-facet knowledge base  (hand-curated for high-frequency facets)
# Key → (synonyms, positive_indicators, negative_indicators,
#         related_facets, examples)
# ---------------------------------------------------------------------------

_FACET_KB: Dict[str, Tuple[List[str], List[str], List[str], List[str], List[str]]] = {
    "risk taking": (
        ["risk tolerance", "boldness", "daring", "uncertainty acceptance",
         "chance-taking", "adventurousness"],
        ["quit stable job for a startup", "invested life savings into new venture",
         "moved to a new country without a safety net",
         "signed up for an extreme sport despite the danger"],
        ["refused any investment that wasn't guaranteed",
         "declined the opportunity citing potential failure",
         "never acts without certainty of outcome"],
        ["adventure seeking", "courage", "impulsivity", "confidence"],
        ["I quit my stable government job to start a company.",
         "I put all my savings into crypto — it was a huge gamble.",
         "I agreed to the deal even though the terms were risky."],
    ),
    "compassion": (
        ["empathy", "care", "concern for others", "sympathy", "kindheartedness",
         "benevolence", "tender-heartedness"],
        ["stayed at the hospital with a friend overnight",
         "volunteered time to help a struggling colleague",
         "listened patiently without judgement"],
        ["ignored the distress of others", "showed indifference to suffering",
         "dismissed emotional needs as irrelevant"],
        ["empathy", "kindness", "warmth", "compassion fatigue"],
        ["I stayed up all night with my friend after her diagnosis.",
         "I noticed he was upset and stopped my work to check on him.",
         "She donated her bonus to help the family next door."],
    ),
    "honesty": (
        ["truthfulness", "candour", "sincerity", "transparency", "frankness",
         "integrity", "forthrightness"],
        ["admitted a mistake unprompted", "told the truth despite consequences",
         "refused to hide relevant information"],
        ["withheld critical information", "told half-truths to avoid conflict",
         "claimed credit for others' work"],
        ["integrity", "authenticity", "courage", "trust"],
        ["I admitted the mistake even though I knew I'd be blamed.",
         "I told my manager the project timeline was unrealistic.",
         "He revealed the conflict of interest before the meeting."],
    ),
    "curiosity": (
        ["inquisitiveness", "intellectual interest", "eagerness to learn",
         "open-mindedness", "wonder", "exploration drive"],
        ["asked probing follow-up questions", "researched the topic independently",
         "expressed genuine interest in an unfamiliar subject"],
        ["showed no interest in exploring new ideas",
         "deflected questions about the topic",
         "stated they had no desire to learn more"],
        ["creativity", "openness", "cognitive", "learning"],
        ["I spent the whole weekend reading about quantum mechanics.",
         "I asked five different experts for their perspective.",
         "She enrolled in a course on a completely new subject."],
    ),
    "empathy": (
        ["compassion", "perspective-taking", "emotional attunement",
         "understanding others", "sensitivity to others"],
        ["accurately reflected the other person's emotional state",
         "acknowledged feelings before offering solutions",
         "said 'I understand how that must feel'"],
        ["dismissed emotions as irrational",
         "immediately jumped to advice without acknowledging feelings",
         "showed irritation at the other person's emotional response"],
        ["compassion", "warmth", "kindness", "compassion fatigue"],
        ["I could tell she was overwhelmed, so I just sat with her.",
         "He said: 'That sounds incredibly hard — I'm sorry you're going through this.'",
         "Before giving feedback, she acknowledged how much effort I'd put in."],
    ),
    "assertiveness": (
        ["directness", "self-advocacy", "confidence in communication",
         "standing one's ground", "forthright expression"],
        ["stated opinion clearly without hedging",
         "pushed back on an unfair request",
         "initiated a difficult conversation proactively"],
        ["agreed to avoid conflict despite disagreement",
         "apologised excessively before making a point",
         "avoided expressing a view when asked directly"],
        ["courage", "confidence", "democratic leadership", "social"],
        ["I told my manager directly that the workload was unsustainable.",
         "She said: 'I disagree, and here's why.'",
         "He declined the request firmly but respectfully."],
    ),
    "integrity": (
        ["moral uprightness", "ethical consistency", "principled behaviour",
         "trustworthiness", "incorruptibility"],
        ["declined a shortcut that would have compromised standards",
         "followed through on a commitment despite inconvenience",
         "refused a bribe or improper incentive"],
        ["cut corners to meet a deadline",
         "changed their story based on who was listening",
         "accepted benefits for overlooking a violation"],
        ["honesty", "authenticity", "courage", "conscientiousness"],
        ["I refused to sign the report because the numbers were wrong.",
         "She returned the extra change even though no one noticed.",
         "He told the client the truth about the delay rather than blaming others."],
    ),
    "warmth": (
        ["friendliness", "approachability", "affability", "cordiality",
         "genuine care", "personal warmth"],
        ["greeted others with genuine enthusiasm",
         "used first names and remembered personal details",
         "created a comfortable atmosphere in conversations"],
        ["was cold or distant in interactions",
         "kept all exchanges strictly transactional",
         "rarely acknowledged the personal side of conversations"],
        ["kindness", "empathy", "compassion", "trust"],
        ["She always remembers birthdays and asks how people are doing.",
         "He made the new employee feel welcome on their first day.",
         "I felt at ease immediately because she was so warm and open."],
    ),
    "pessimism": (
        ["negative outlook", "defeatist attitude", "hopelessness",
         "cynicism", "expecting the worst"],
        ["assumed the project would fail before it started",
         "framed neutral information negatively",
         "expressed doubt about positive outcomes"],
        ["expressed confidence in a positive outcome",
         "focused on opportunities rather than obstacles",
         "encouraged others with optimistic framing"],
        ["moroseness", "anxiety", "neuroticism", "emotion"],
        ["I knew it wouldn't work — it never does.",
         "Why bother? The market conditions are terrible.",
         "She said: 'I'm not getting my hopes up — something will go wrong.'"],
    ),
    "naivety": (
        ["credulity", "lack of guile", "simplistic trust",
         "over-confidence in others", "gullibility"],
        ["accepted claims without questioning the source",
         "trusted a stranger's promise without verification",
         "failed to notice obvious deception"],
        ["questioned the motive behind the offer",
         "asked for evidence before accepting a claim",
         "expressed appropriate scepticism"],
        ["gullibility", "trust", "cognitive", "common sense"],
        ["I sent money to someone I'd never met because they seemed friendly.",
         "She believed the email was from the bank without checking.",
         "He assumed everyone had good intentions and didn't verify anything."],
    ),
    "courage": (
        ["bravery", "boldness", "nerve", "valour", "fearlessness",
         "willingness to face adversity"],
        ["spoke up in a hostile environment",
         "took action despite obvious personal risk",
         "confronted wrongdoing despite potential repercussions"],
        ["stayed silent to avoid conflict",
         "chose the safe path over the right path",
         "backed down when challenged"],
        ["risk taking", "assertiveness", "integrity", "authenticity"],
        ["I reported the misconduct even though I knew it would cost me.",
         "She stood on stage even though she was terrified.",
         "He disagreed with the CEO publicly in the meeting."],
    ),
    "gullibility": (
        ["credulity", "over-trusting", "naivety", "suggestibility",
         "susceptibility to deception"],
        ["believed a clearly implausible claim",
         "repeated misinformation without checking",
         "was easily persuaded by social pressure"],
        ["fact-checked before accepting a claim",
         "expressed healthy scepticism",
         "asked for a source or evidence"],
        ["naivety", "trust", "cognitive", "common sense"],
        ["I forwarded the email immediately — it sounded so official.",
         "She believed everything he said without a second thought.",
         "He donated money after reading one Facebook post."],
    ),
}


# ---------------------------------------------------------------------------
# Synonym generation  (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _build_synonyms(name: str, category: str) -> List[str]:
    """
    Return synonym list for a facet.  Combines:
      1. Hand-curated KB entry (if available)
      2. Morphological variants derived from the name tokens
      3. Category-level seed terms (top-3)
    """
    name_lower = name.lower()
    key = name_lower.strip()

    # 1. KB lookup (exact or substring match)
    kb_syns: List[str] = []
    for kb_key, (syns, *_) in _FACET_KB.items():
        if kb_key in key or key in kb_key:
            kb_syns = syns[:]
            break

    # 2. Morphological variants from name tokens
    tokens = re.sub(r"[^a-z ]", "", name_lower).split()
    morph: List[str] = []
    for tok in tokens:
        if len(tok) > 3:
            morph.append(tok + "ness")    # e.g. "bold" → "boldness"
            morph.append(tok + "ity")     # e.g. "sensitive" → "sensitivity"
            morph.append(tok + "ing")
    # Keep short, de-duped, not identical to existing
    existing = {s.lower() for s in kb_syns} | {name_lower}
    morph = [m for m in morph if m not in existing and len(m) < 25][:3]

    # 3. Category seeds (top-3)
    cat_seeds = _CATEGORY_SYNONYM_SEEDS.get(category, [])[:3]

    combined = kb_syns + morph + cat_seeds
    # Deduplicate while preserving order, skip if identical to facet name
    seen: set[str] = {name_lower}
    result: List[str] = []
    for s in combined:
        sl = s.lower()
        if sl not in seen:
            seen.add(sl)
            result.append(s)

    return result[:8]  # cap at 8


# ---------------------------------------------------------------------------
# Indicator & example generation
# ---------------------------------------------------------------------------

def _build_indicators(
    name: str, category: str
) -> Tuple[List[str], List[str]]:
    """Return (positive_indicators, negative_indicators)."""
    key = name.lower().strip()

    # KB lookup
    for kb_key, (_, pos, neg, *_) in _FACET_KB.items():
        if kb_key in key or key in kb_key:
            return pos, neg

    # Generic fallback derived from category
    cat_label = category.replace("_", " ")
    pos = [
        f"Clearly demonstrates {name.lower()} in their response",
        f"The turn contains strong evidence of {name.lower()}",
        f"The speaker's language consistently reflects {name.lower()}",
    ]
    neg = [
        f"Absence of any signal related to {name.lower()}",
        f"The turn contradicts or undermines {name.lower()}",
        f"No {cat_label} indicator for {name.lower()} is present",
    ]
    return pos, neg


def _build_examples(name: str) -> List[str]:
    """Return concrete conversation-snippet examples."""
    key = name.lower().strip()
    for kb_key, (*_, examples) in _FACET_KB.items():
        if kb_key in key or key in kb_key:
            return examples
    # Minimal generic fallback — clearly better than "This person demonstrates X."
    return [
        f"A speaker says something that clearly reflects {name.lower()}.",
        f"The conversation turn is dominated by {name.lower()}.",
    ]


def _build_related_facets(name: str, category: str) -> List[str]:
    """Return related facet names from KB or category peers."""
    key = name.lower().strip()
    for kb_key, (_, _, _, related, *_) in _FACET_KB.items():
        if kb_key in key or key in kb_key:
            return related

    # Category-level peers as fallback
    peers = {
        "emotion": ["Compassion", "Empathy", "Warmth", "Moroseness"],
        "personality": ["Integrity", "Courage", "Honesty", "Assertiveness"],
        "cognitive": ["Curiosity", "Common Sense", "Statistical Reasoning"],
        "social": ["Assertiveness", "Democratic Leadership", "Empathy"],
        "safety": ["Manipulation", "Deception", "Hostility"],
        "clinical_health": ["Moroseness", "Anxiety", "Compulsive Activities"],
        "behavioral_lifestyle": ["Adventure Seeking", "Risk Taking"],
        "spirituality_culture": ["Mindfulness", "Pilgrimage Participation"],
        "other": [],
    }
    return [p for p in peers.get(category, []) if p.lower() != name.lower()][:4]


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------

def _build_description(name: str, category: str, synonyms: List[str]) -> str:
    """
    Generate a rich, facet-specific description paragraph.
    Avoids the generic template pattern; includes synonyms for density.
    """
    cat_desc = CATEGORY_PROFILES.get(category, {}).get("description", "")
    syn_str = ", ".join(synonyms[:4]) if synonyms else name.lower()
    return (
        f"{name} (also: {syn_str}) is a {category.replace('_', ' ')} facet "
        f"that captures the degree to which a conversational turn exhibits "
        f"'{name.lower()}'. "
        f"Context: {cat_desc} "
        f"A high score on {name} indicates that the speaker's words, tone, or "
        f"choices strongly reflect this quality; a low score indicates its "
        f"absence or the presence of the opposing tendency."
    )


# ---------------------------------------------------------------------------
# Rubric anchor generation
# ---------------------------------------------------------------------------

_RUBRIC_TEMPLATES: Dict[str, Dict[int, str]] = {
    "emotion": {
        1: "Opposite affect — turn expresses the emotional antithesis of {facet}.",
        2: "Minimal emotional signal toward {facet}; largely absent.",
        3: "Neutral — affect is mixed, ambiguous, or unrelated to {facet}.",
        4: "Moderate — clear emotional expression aligned with {facet}.",
        5: "Intense — {facet} dominates the turn; highly expressive and unambiguous.",
    },
    "personality": {
        1: "Strong opposite tendency — behaviour is antithetical to {facet}.",
        2: "Weak or incidental signal; {facet} is mostly absent.",
        3: "Mixed — some evidence of {facet} but also contradictory signals.",
        4: "Clearly present — turn shows consistent evidence of {facet}.",
        5: "Strongly exemplifies {facet}; central, pronounced, unmistakable.",
    },
    "cognitive": {
        1: "No application of {facet}; reasoning absent or clearly wrong.",
        2: "Superficial or partial application of {facet}.",
        3: "Adequate — basic application sufficient for the turn.",
        4: "Strong — correct, deliberate application of {facet} with depth.",
        5: "Expert — sophisticated, fully-correct demonstration of {facet}.",
    },
    "social": {
        1: "Absent — no interpersonal signal related to {facet}.",
        2: "Weak or indirect signal of {facet} in the interaction.",
        3: "Neutral — some social signal but {facet} is not the dominant pattern.",
        4: "Present — {facet} is clearly demonstrated in this turn.",
        5: "Strongly and consistently demonstrated; {facet} is the defining feature.",
    },
    "safety": {
        1: "Safe — no safety concern related to {facet}.",
        2: "Minor / borderline indication of {facet}; not actionable alone.",
        3: "Ambiguous — could be read as {facet} depending on context.",
        4: "Notable concern — clear instance of {facet}; warrants review.",
        5: "Severe — egregious instance of {facet}; immediate action warranted.",
    },
    "clinical_health": {
        1: "No clinical signal related to {facet}.",
        2: "Faint or indirect signal of {facet}.",
        3: "Possible — ambiguous; clinical follow-up would be needed.",
        4: "Likely — strong textual signal of {facet}; clinically notable.",
        5: "Definitive — unambiguous, quantified, or severe expression of {facet}.",
    },
    "behavioral_lifestyle": {
        1: "None / never — no mention or explicit denial of {facet}.",
        2: "Rare — mentioned as occasional or infrequent.",
        3: "Moderate — average / typical frequency of {facet}.",
        4: "Frequent — clear pattern; {facet} is a regular part of life.",
        5: "Central — {facet} is described as a defining lifestyle feature.",
    },
    "spirituality_culture": {
        1: "Absent — no reference to {facet}.",
        2: "Tangential — brief or implicit reference to {facet}.",
        3: "Present — {facet} mentioned as part of life.",
        4: "Strong — {facet} clearly emphasised.",
        5: "Central — {facet} is the dominant frame of the entire turn.",
    },
    "_default": {
        1: "Strongly absent — turn actively contradicts or shows the opposite of {facet}.",
        2: "Mostly absent — only weak or indirect signals of {facet}.",
        3: "Neutral / mixed — evidence is ambiguous or {facet} is not clearly applicable.",
        4: "Mostly present — turn clearly displays {facet}, though not in extreme form.",
        5: "Strongly present — {facet} is pronounced, unambiguous, and central to the message.",
    },
}


def _build_rubric_anchors(name: str, category: str) -> Dict[int, str]:
    template = _RUBRIC_TEMPLATES.get(category, _RUBRIC_TEMPLATES["_default"])
    return {lvl: tpl.format(facet=name.lower()) for lvl, tpl in template.items()}


# ---------------------------------------------------------------------------
# Keywords for BM25 hybrid retrieval
# ---------------------------------------------------------------------------

def _build_keywords(
    name: str,
    synonyms: List[str],
    positive_indicators: List[str],
    related_facets: List[str],
) -> List[str]:
    """Build a deduplicated keyword list for BM25 hybrid retrieval."""
    raw: List[str] = []

    # Name tokens
    raw += [t.lower() for t in re.sub(r"[^a-zA-Z0-9 ]", " ", name).split() if len(t) > 2]

    # Synonym tokens (first token of each synonym phrase)
    for syn in synonyms:
        raw += [t.lower() for t in syn.split()[:2] if len(t) > 2]

    # First content word from each positive indicator
    for ind in positive_indicators[:3]:
        words = [t.lower() for t in ind.split() if len(t) > 3]
        raw += words[:2]

    # Related facet slugs
    for rf in related_facets:
        raw += [t.lower() for t in rf.split() if len(t) > 2]

    seen: set[str] = set()
    result: List[str] = []
    for k in raw:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result[:20]


# ---------------------------------------------------------------------------
# Retrieval text  (the field embedded for FAISS indexing)
# ---------------------------------------------------------------------------

def _build_retrieval_text(
    name: str,
    category: str,
    description: str,
    synonyms: List[str],
    related_facets: List[str],
    positive_indicators: List[str],
    examples: List[str],
    **kwargs,
) -> str:
    """
    Construct a dense, information-rich paragraph for FAISS embedding.

    Structured to maximise semantic recall: name → category → definition →
    synonyms → related concepts → behavioural signals → concrete examples.
    """
    syn_str = "; ".join(synonyms[:5]) if synonyms else ""
    related_str = "; ".join(related_facets[:4]) if related_facets else ""
    indicators_str = " | ".join(positive_indicators[:3]) if positive_indicators else ""
    examples_str = " | ".join(examples[:2]) if examples else ""

    neg_ind = kwargs.get("negative_indicators", [])
    neg_str = " | ".join(neg_ind[:2]) if neg_ind else ""

    parts = [
        f"Facet: {name}.",
        f"Category: {category.replace('_', ' ')}.",
        description,
    ]
    if syn_str:
        parts.append(f"Synonyms and related terms: {syn_str}.")
    if related_str:
        parts.append(f"Related facets: {related_str}.")
    if indicators_str:
        parts.append(f"Positive behavioural signals: {indicators_str}.")
    if neg_str:
        parts.append(f"Absence signals: {neg_str}.")
    if examples_str:
        parts.append(f"Conversation examples: {examples_str}.")


# ---------------------------------------------------------------------------
# Main enrichment driver
# ---------------------------------------------------------------------------

def enrich_facets(
    df: pd.DataFrame,
    embedder=None,
    cat_vecs=None,
    cat_keys: Optional[List[str]] = None,
    sim_floor: float = 0.18,
) -> pd.DataFrame:
    """
    Enrich a cleaned facets DataFrame (facet_id, facet_name, raw_name).

    Parameters
    ----------
    df         : output of ``clean_facets()``
    embedder   : optional sentence-transformers embedder for categorisation
    cat_vecs   : pre-computed category vectors (L2-normalised np.ndarray)
    cat_keys   : list of category strings matching cat_vecs rows
    sim_floor  : minimum similarity for embedding categorisation

    Returns
    -------
    pd.DataFrame with the full enriched schema.
    """
    # Support both "facet_name" (phase-2 test schema) and "name" (pipeline schema)
    name_col = "facet_name" if "facet_name" in df.columns else "name"

    rows = []
    for _, row in df.iterrows():
        name = str(row[name_col])
        facet_id = str(row.get("facet_id", ""))
        raw_name = str(row.get("raw_name", name))

        # --- categorise ---
        cat = categorize(
            name,
            embedder=embedder,
            cat_vecs=cat_vecs,
            cat_keys=cat_keys,
            sim_floor=sim_floor,
        )

        # --- enrich ---
        synonyms = _build_synonyms(name, cat)
        pos_ind, neg_ind = _build_indicators(name, cat)
        related = _build_related_facets(name, cat)
        examples = _build_examples(name)
        description = _build_description(name, cat, synonyms)
        rubric = _build_rubric_anchors(name, cat)
        keywords = _build_keywords(name, synonyms, pos_ind, related)
        retrieval_text = _build_retrieval_text(
            name, cat, description, synonyms, related, pos_ind, examples,
            negative_indicators=neg_ind,
        )

        rows.append(
            {
                "facet_id": facet_id,
                "facet_name": name,
                "raw_name": raw_name,
                "category": cat,
                "description": description,
                "positive_indicators": json.dumps(pos_ind),
                "negative_indicators": json.dumps(neg_ind),
                "score_1_anchor": rubric[1],
                "score_2_anchor": rubric[2],
                "score_3_anchor": rubric[3],
                "score_4_anchor": rubric[4],
                "score_5_anchor": rubric[5],
                "synonyms": json.dumps(synonyms),
                "related_facets": json.dumps(related),
                "examples": json.dumps(examples),
                "keywords": json.dumps(keywords),
                "retrieval_text": retrieval_text,
            }
        )

    enriched_df = pd.DataFrame(rows)

    # Persist to the canonical output path
    out_path = Path("data/processed/facets_enriched.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(out_path, index=False)

    return enriched_df


# ---------------------------------------------------------------------------
# CLI entry-point (pipeline usage)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich cleaned facets.")
    parser.add_argument("--cleaned", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--embedder", type=str, default=None)
    parser.add_argument(
        "--use-embedder",
        action="store_true",
        help="Use sentence-transformers for categorisation (higher quality).",
    )
    args = parser.parse_args()

    try:
        from src.utils.config import load_config
        cfg = load_config()
        cleaned_path = args.cleaned or Path(cfg["paths"]["cleaned_facets"])
        out_csv = args.out_csv or Path(cfg["paths"]["enriched_facets"])
        embedder_name = args.embedder or cfg.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
    except Exception:
        cleaned_path = args.cleaned or Path("data/processed/facets_cleaned.csv")
        out_csv = args.out_csv or Path("data/processed/facets_enriched.csv")
        embedder_name = args.embedder or "all-MiniLM-L6-v2"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned facets not found at {cleaned_path}.")

    embedder = cat_vecs = cat_keys = None
    if args.use_embedder:
        embedder, cat_vecs, cat_keys = build_category_embedder(embedder_name)

    df = pd.read_csv(cleaned_path)
    enriched = enrich_facets(df, embedder=embedder, cat_vecs=cat_vecs, cat_keys=cat_keys)
    enriched.to_csv(out_csv, index=False)
    print(f"Enriched {len(enriched)} facets → {out_csv}")


if __name__ == "__main__":
    main()