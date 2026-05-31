"""
scripts/analyze_facets.py — Step 1 of Phase 17A.

Reads Facets_Assignment.csv, infers category for every facet,
classifies each as conversation-inferable / weakly-inferable / non-inferable,
and writes evaluation_results/facet_analysis.json.

Run from project root:
    python scripts/analyze_facets.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _find_csv_path() -> Path:
    base_dir = ROOT / "data" / "index"
    candidate = base_dir / "Facets_Assignment.csv"
    if candidate.exists():
        return candidate

    # Accept common filename variants for the facets assignment file.
    allowed = [
        "facets_assignment.csv",
        "facets assignment.csv",
        "facets-assignment.csv",
        "facetsassignment.csv",
    ]
    for path in base_dir.glob("*.csv"):
        if path.name.lower() in allowed:
            return path
        normalized = path.name.lower().replace(" ", "_").replace("-", "_")
        if normalized in allowed:
            return path

    raise FileNotFoundError(
        f"Could not find Facets_Assignment.csv in {base_dir}. "
        f"Existing CSV files: {[p.name for p in base_dir.glob('*.csv')]}"
    )

CSV_PATH    = _find_csv_path()
OUT_PATH    = ROOT / "evaluation_results" / "facet_analysis.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Category keyword rules  (same logic as src/data_pipeline/categorizer.py
# but standalone so the script has no pipeline dependency)
# ---------------------------------------------------------------------------
_CATEGORY_RULES: dict[str, list[str]] = {
    "clinical": [
        "fsh", "hormone", "cortisol", "testosterone", "estrogen", "parathyroid",
        "chromatin", "biomarker", "hba1c", "bmi", "blood", "basophil",
        "metabolic", "serotonin", "polygenic", "immune", "caffeine intake",
        "macronutrient", "sleep apnea", "chronic pain", "drug-use",
        "inversion comfort", "vision-check", "caffeine sensitivity",
    ],
    "spirituality": [
        "spiritual", "astrology", "zodiac", "faith", "prayer", "meditation",
        "mindfulness", "transcend", "divine", "pilgrimage", "sufi", "sikh",
        "kabbalah", "islamic", "jewish", "buddhist", "hindu", "gnostic",
        "i ching", "satya", "scripture", "sacred text", "quran", "vrata",
        "bhagavad", "yoga discipline", "holiness", "aura", "reiki",
        "ego dissolution", "new-age", "channeling", "bahá'í",
    ],
    "clinical_lifestyle": [
        "eco-tourism", "travel-companion", "passport", "digital-nomad",
        "commute", "home-security", "cloud-backup", "subscription count",
        "blog-subscriber", "peer-to-peer lending", "gamified-finance",
        "open-source", "robotics", "graffiti", "dance-cardio", "dance rehearsal",
        "dance-style", "choir", "music-lesson", "museum visit", "time outdoors",
        "wake-time", "snacking", "breakfast-skip", "processed-food",
        "eating habits", "dietary", "local-food", "sustainable-transport",
        "public-transport", "training-cycle", "pet-enrichment", "sleep-environ",
    ],
    "cognition": [
        "reasoning", "logic", "analytical", "critical thinking", "statistical",
        "problem solving", "creativity", "intelligence", "memory", "learning",
        "common-sense", "common sense", "naive", "naivety", "gullible", "gullibility",
        "spatial", "auditory", "sequential memory", "numerical", "alphanumeric",
        "alphabetical", "information retention", "synthesis", "analogies",
        "working memory", "mental arithmetic", "rapid cognitive", "sentence structure",
        "spelling", "language use", "comprehension", "estimating", "logical sequence",
        "data analysis", "network basics", "material properties", "anatomy",
        "troubleshooting", "computer skills", "economic reasoning",
        "understanding math", "understanding mechanic", "use of math",
        "psychomotor", "precision of movement", "non-verbal communication",
        "listening skills",
    ],
    "leadership": [
        "leadership", "democratic", "authoritarian", "visionary", "coaching",
        "strategic", "executive", "management", "delegation", "transactional",
        "encouraging participation", "contribution to group", "desire to influence",
    ],
    "mental_health": [
        "depression", "anxiety", "burnout", "trauma", "ptsd", "psychosis",
        "bipolar", "ocd", "adhd", "dissociation", "suicid", "self.harm",
        "disorder", "phobia", "hypomania", "hysteria", "psychoticism",
        "negative affect", "stress recovery", "physical-violence",
        "sleep-disorder",
    ],
    "emotion": [
        "compassion", "empathy", "kindness", "warmth", "love", "grief",
        "sadness", "joy", "anger", "fear", "guilt", "shame", "pride",
        "morose", "pessim", "fatigue", "emotionalism", "merriness",
        "joyfulness", "blissful", "desperation", "discontentment",
        "high-spirited", "contentment", "happiness", "affection",
        "ardency", "vivacity", "depth of emotional",
    ],
    "relational": [
        "assertive", "passive", "aggressive", "trust", "loyal", "jealous",
        "attachment", "intimacy", "boundaries", "social", "relationship",
        "chivalrous", "affiliation", "cooperation", "collaboration",
        "talkativeness", "social boldness", "seeking approval", "submissive",
        "servil", "abasement", "martyrdom", "overprotective",
    ],
    "behavioral": [
        "adventure", "impulsive", "procrastinat", "habit", "addiction",
        "compulsive", "routine", "discipline", "slothful", "hardworking",
        "self-improvement", "boredom", "goal-directed", "initiative",
        "volunteering", "meeting deadline", "feedback-giving", "self-efficacy",
        "perseverance", "doggedness", "persistence",
    ],
    "personality": [
        "risk", "honest", "integrity", "authentic", "courage", "curiosity",
        "openness", "conscientiousness", "extraversion", "agreeableness",
        "neuroticism", "determinedness", "frankness", "genuine", "outspoken",
        "individuali", "originality", "quirkiness", "conventionalism",
        "conservatism", "liberalism", "patriotism", "ethnocentrism",
        "rebellious", "immaturity", "suspicion", "hostility", "disrespect",
        "dishonesty", "brazenness", "cunning", "acidity", "cantankerous",
        "coarseness", "dignity", "decency", "civility", "impartial",
        "equitable", "justice", "wisdom",
    ],
}

_DEFAULT = "personality"

# ---------------------------------------------------------------------------
# Inferability rules
# ---------------------------------------------------------------------------
_NON_INFERABLE_KEYWORDS = [
    # bio / lab
    "fsh", "basophil", "chromatin", "serotonin transporter", "polygenic",
    "immune-response age", "metabolic rate", "caffeine sensitivity gene",
    "macronutrient ratio", "parathyroid", "hba1c",
    # astrology / esoteric
    "i ching", "aura-color", "kabbalah sephira", "ego dissolution",
    "gnostic", "sufi retreat", "sufi practice", "sikh spiritual",
    "jewish spiritual", "buddhist practice", "hindu spiritual",
    "bahá'í", "new-age", "channeling", "sacred text engagement",
    "quran surahs", "zohar", "bhagavad", "vrata vows", "yoga discipline hours",
    "energy-healing", "reiki",
    # biometric / device
    "sleep apnea", "sleep-disorder diagnosis", "chronic pain presence",
    "vision-check frequency", "wake-time consistency",
    "sleep-environment temperature", "inversion comfort",
    # lifestyle tracking
    "commute time", "home-security", "cloud-backup", "subscription count",
    "blog-subscriber", "gamified-finance", "peer-to-peer lending",
    "digital-nomad months", "passport-stamps", "eco-tourism trips",
    "training-cycle length", "pet-enrichment", "dance-cardio",
    "dance rehearsal hours", "dance-style mastery", "choir participation",
    "music-lessons years", "museum visits", "time outdoors",
    "public-transport km", "sustainable-transport", "local-food sourcing",
    "processed-food frequency", "snacking behavior", "breakfast-skipping",
    "eating habits", "dietary habits", "caffeine intake",
    "drug-use history", "physical-violence exposure",
    # purely technical
    "alphanumeric", "alphabetical filing", "numeric filing",
    "network basics", "computer skills", "robotics-interaction",
    "open-source contributions", "skill-endorsements count",
    "graffiti appreciation",
    # headers / category labels (not real facets)
    "subcomponents:", "styles:", "themes:", "parameters:", "facets:",
    "motivation:", "tendencies:", "behaviors:", "components:", "styles and behaviors:",
    "end points:", "potential:", "types:", "leadership styles:",
]

_WEAK_KEYWORDS = [
    "burnout", "depression symptoms", "self esteem", "self-esteem", "selfesteem",
    "anxiety", "stress recovery", "sense-of-coherence", "attachment style",
    "attachment avoidance", "psychological construct", "hexaco", "big five",
    "enneagram", "social desirability", "acculturative stress",
    "consummatory pleasure", "excuse-making", "executive-function",
    "operant-learning", "perfectionist", "psychological safety",
    "resilience-trait", "self-compassion", "social conformity",
    "cultural intelligence", "identity diffusion", "hope scale",
    "eye-contact", "faux pas", "mentalizing network",
    "social-cognition", "defense-mechanism", "kink-interest",
    "data-sharing consent", "preferred epistemology",
    "binding foundations",
]


def _clean_name(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^[0-9]+\.\s*", "", raw)  # strip leading number
    return raw.strip()


def _categorize(name: str) -> str:
    lower = name.lower()
    for cat, keywords in _CATEGORY_RULES.items():
        for kw in keywords:
            if kw in lower:
                return cat
    return _DEFAULT


def _inferability(name: str) -> str:
    lower = name.lower()
    for kw in _NON_INFERABLE_KEYWORDS:
        if kw in lower:
            return "non_inferable"
    for kw in _WEAK_KEYWORDS:
        if kw in lower:
            return "weakly_inferable"
    return "inferable"


def main():
    # ── Read CSV ──────────────────────────────────────────────────────────────
    raw_lines = CSV_PATH.read_text(encoding="utf-8").splitlines()
    facets = []
    for line in raw_lines:
        name = _clean_name(line)
        if not name or name.lower() == "facets":
            continue
        facets.append(name)

    total = len(facets)

    # ── Classify each facet ───────────────────────────────────────────────────
    category_dist: dict[str, list[str]]      = defaultdict(list)
    inferability_dist: dict[str, list[str]]  = defaultdict(list)
    records = []

    for name in facets:
        cat   = _categorize(name)
        infer = _inferability(name)
        category_dist[cat].append(name)
        inferability_dist[infer].append(name)
        records.append({"facet": name, "category": cat, "inferability": infer})

    # ── Build output ──────────────────────────────────────────────────────────
    result = {
        "total_facets": total,
        "category_distribution": {
            cat: {"count": len(names), "facets": names}
            for cat, names in sorted(category_dist.items())
        },
        "inferability_summary": {
            "inferable":         len(inferability_dist["inferable"]),
            "weakly_inferable":  len(inferability_dist["weakly_inferable"]),
            "non_inferable":     len(inferability_dist["non_inferable"]),
        },
        "inferable_facets":        inferability_dist["inferable"],
        "weakly_inferable_facets": inferability_dist["weakly_inferable"],
        "non_inferable_facets":    inferability_dist["non_inferable"],
        "all_records":             records,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✓ Wrote {OUT_PATH}")
    print(f"  Total facets      : {total}")
    print(f"  Inferable         : {result['inferability_summary']['inferable']}")
    print(f"  Weakly inferable  : {result['inferability_summary']['weakly_inferable']}")
    print(f"  Non-inferable     : {result['inferability_summary']['non_inferable']}")
    print(f"  Categories found  : {list(result['category_distribution'].keys())}")


if __name__ == "__main__":
    main()