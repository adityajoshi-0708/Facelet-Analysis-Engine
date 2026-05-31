"""Phase 2 tests — data pipeline: clean, enrich, categorize."""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.clean import clean_facets
from src.data_pipeline.enrich import enrich_facets
from src.data_pipeline.categorizer import categorize

SEED_FACETS = [
    "Risk Taking", "Compassion", "Honesty", "Naivety", "Adventure Seeking",
    "Assertiveness", "Empathy", "Statistical Reasoning", "Compassion Fatigue",
    "Democratic Leadership", "Moroseness", "Common Sense", "Kindness", "Warmth",
    "Courage", "Authenticity", "Integrity", "Pessimism", "Curiosity", "Gullibility",
]


def test_clean_facets():
    df = clean_facets(SEED_FACETS)
    assert len(df) > 0, "Cleaned DataFrame must not be empty"
    print("✓ clean_facets passed")


def test_clean_facets_from_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "facets.csv"
        path.write_text("facet_name\nTrust\n", encoding="utf-8")
        df = clean_facets(Path(temp_dir))
        assert len(df) == 1
        assert df.iloc[0]["facet_name"] == "Trust"
    print("✓ clean_facets_from_directory passed")


def test_generate_facet_id():
    df = clean_facets(SEED_FACETS)
    assert "facet_id" in df.columns
    assert "facet_name" in df.columns
    print("✓ generate_facet_id passed")


def test_no_spaces_in_ids():
    df = clean_facets(SEED_FACETS)
    assert df["facet_id"].str.contains(" ").sum() == 0, "facet_id must have no spaces"
    print("✓ no_spaces_in_ids passed")


def test_enriched_columns():
    clean_df = clean_facets(SEED_FACETS)
    enriched_df = enrich_facets(clean_df)
    required_cols = [
        "facet_id", "facet_name", "category", "description",
        "positive_indicators", "negative_indicators",
        "score_1_anchor", "score_2_anchor", "score_3_anchor",
        "score_4_anchor", "score_5_anchor",
        "synonyms", "related_facets",
    ]
    for col in required_cols:
        assert col in enriched_df.columns, f"Missing column: {col}"
    assert "category" in enriched_df.columns
    print("✓ enriched_columns passed")


def test_retrieval_text_present():
    """retrieval_text must exist and be non-empty for every row."""
    clean_df = clean_facets(SEED_FACETS)
    enriched_df = enrich_facets(clean_df)
    assert "retrieval_text" in enriched_df.columns, "Missing retrieval_text column"
    assert enriched_df["retrieval_text"].str.len().min() > 50, (
        "retrieval_text entries are too short to be useful for FAISS embedding"
    )
    print("✓ retrieval_text_present passed")


def test_keywords_are_json_list():
    """keywords column must contain valid JSON lists."""
    import json
    clean_df = clean_facets(SEED_FACETS)
    enriched_df = enrich_facets(clean_df)
    assert "keywords" in enriched_df.columns, "Missing keywords column"
    for val in enriched_df["keywords"]:
        parsed = json.loads(val)
        assert isinstance(parsed, list), f"keywords is not a JSON list: {val}"
    print("✓ keywords_are_json_list passed")


def test_categorizer_compassion():
    assert categorize("Compassion") == "emotion", (
        f"Expected 'emotion', got '{categorize('Compassion')}'"
    )
    print("✓ categorizer_compassion passed")


def test_categorizer_risk_taking():
    assert categorize("Risk Taking") == "personality", (
        f"Expected 'personality', got '{categorize('Risk Taking')}'"
    )
    print("✓ categorizer_risk_taking passed")


def test_categorizer_statistical_reasoning():
    result = categorize("Statistical Reasoning")
    assert result == "cognitive", (
        f"Expected 'cognitive', got '{result}'"
    )
    print("✓ categorizer_statistical_reasoning passed")


def test_categorizer_democratic_leadership():
    result = categorize("Democratic Leadership")
    assert result == "social", (
        f"Expected 'social', got '{result}'"
    )
    print("✓ categorizer_democratic_leadership passed")


def test_csv_saved():
    clean_df = clean_facets(SEED_FACETS)
    enrich_facets(clean_df)
    assert Path("data/processed/facets_enriched.csv").exists(), (
        "data/processed/facets_enriched.csv not found"
    )
    print("✓ csv_saved passed")


def test_examples_are_concrete():
    """Examples should not contain the generic boilerplate pattern."""
    import json
    clean_df = clean_facets(SEED_FACETS)
    enriched_df = enrich_facets(clean_df)
    generic_pattern = "this person demonstrates"
    for _, row in enriched_df.iterrows():
        examples = json.loads(row["examples"])
        for ex in examples:
            assert generic_pattern not in ex.lower(), (
                f"Generic example detected for '{row['facet_name']}': {ex}"
            )
    print("✓ examples_are_concrete passed")


if __name__ == "__main__":
    test_clean_facets()
    test_clean_facets_from_directory()
    test_generate_facet_id()
    test_no_spaces_in_ids()
    test_enriched_columns()
    test_retrieval_text_present()
    test_keywords_are_json_list()
    test_categorizer_compassion()
    test_categorizer_risk_taking()
    test_categorizer_statistical_reasoning()
    test_categorizer_democratic_leadership()
    test_csv_saved()
    test_examples_are_concrete()
    print("✅ All Phase 2 tests passed")