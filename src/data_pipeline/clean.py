"""
Phase 2 — Step 1: Data Cleaning
================================
Reads a raw list of facet names (or CSV/directory) and produces a clean
DataFrame with normalized names and stable snake_case facet_ids.

Cleaning pipeline (in order):
  1.  BOM + non-breaking-space removal
  2.  Leading number/punctuation stripping  ("793. Risk Taking" → "Risk Taking")
  3.  CamelCase splitting                   ("HonestyHumility" → "Honesty Humility")
  4.  Trailing punctuation removal          ("Democratic Leadership:" → "Democratic Leadership")
  5.  Internal whitespace collapsing        ("Risk     Taking" → "Risk Taking")
  6.  Title-case normalisation
  7.  Case-insensitive deduplication (first occurrence wins)
  8.  Stable slug / facet_id generation    ("Risk Taking" → "risk_taking")

Output schema:  facet_id | facet_name | raw_name
(All downstream enrichment columns are added by enrich.py — this module
 is intentionally LLM-free so it runs in any environment.)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Candidate CSV filenames when searching a directory
_CSV_CANDIDATES: tuple[str, ...] = (
    "Facets_Assignment.csv",
    "Facets Assignment.csv",
    "facets_assignment.csv",
    "facets assignment.csv",
    "facets.csv",
)


def _normalize_name(raw: str) -> str:
    """Full normalization pipeline for a single raw facet string."""
    s = raw

    # 1. BOM + non-breaking space
    s = s.replace("\ufeff", "").replace("\u00a0", " ")

    # 2. Leading number / bullet  e.g. "793. ", "12) ", "3- "
    s = re.sub(r"^\d+[\.\):\-\s]+", "", s.strip())

    # 3. CamelCase splitting  e.g. "HonestyHumility" → "Honesty Humility"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    # Also handle acronym boundaries  e.g. "FSHLevel" → "FSH Level"
    s = re.sub(r"(?<=[A-Z]{2})(?=[A-Z][a-z])", " ", s)

    # 4. Trailing punctuation
    s = s.rstrip(":;.,!?-")

    # 5. Collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 6. Title-case
    return s.title()


def _to_slug(name: str) -> str:
    """Convert 'Risk Taking' → 'risk_taking' (stable, URL-safe slug)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _find_csv_in_dir(directory: Path) -> Path:
    """Return the first matching candidate CSV inside *directory*."""
    if not directory.is_dir():
        return directory  # caller passed a file path directly

    for candidate in _CSV_CANDIDATES:
        p = directory / candidate
        if p.exists():
            return p

    csv_files = sorted(directory.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        f"Cannot identify a facets CSV in {directory}. "
        f"Expected one of: {', '.join(_CSV_CANDIDATES)} or exactly one .csv file."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_facets(
    source: Optional[Union[List[str], Path, str]] = None,
) -> pd.DataFrame:
    """
    Normalise raw facet names into a clean DataFrame.

    Parameters
    ----------
    source:
        • ``list[str]``  — explicit list of raw facet name strings
        • ``Path | str`` — path to a CSV/TXT file *or* a directory that
          contains a facets CSV (auto-detected by filename candidates)
        • ``None``       — auto-detect via ``src.utils.config.load_config()``

    Returns
    -------
    pd.DataFrame with columns: ``facet_id``, ``facet_name``, ``raw_name``
    """
    names: List[str] = []

    # --- resolve source -------------------------------------------------------
    if source is None:
        try:
            from src.utils.config import load_config
            cfg = load_config()
            raw_path = Path(cfg["data"].get("raw_dir") or cfg["paths"]["raw_facets"])
        except Exception as exc:
            raise FileNotFoundError(
                "No source provided and config lookup failed. "
                "Pass an explicit list, file path, or directory."
            ) from exc
        source = raw_path

    if isinstance(source, (str, Path)):
        source = Path(source)
        if source.is_dir():
            source = _find_csv_in_dir(source)
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")

        if source.suffix.lower() == ".csv":
            raw_df = pd.read_csv(source, encoding="utf-8-sig")
            # Accept "facet_name" or "name" column; fall back to first column
            col = next(
                (c for c in raw_df.columns if c.lower() in {"facet_name", "name", "facets"}),
                raw_df.columns[0],
            )
            names = raw_df[col].dropna().astype(str).tolist()
        else:
            # Plain-text: one facet per line
            names = [
                ln.strip()
                for ln in source.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
    else:
        # Assumed to be list[str]
        names = [str(n) for n in source]

    # --- normalise + deduplicate ----------------------------------------------
    pairs: list[tuple[str, str]] = []  # (raw, normalised)
    seen: set[str] = set()

    for raw in names:
        if not raw.strip():
            continue
        clean = _normalize_name(raw)
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((raw.strip(), clean))

    # --- build DataFrame ------------------------------------------------------
    df = pd.DataFrame(
        {
            "facet_id": [_to_slug(clean) for _, clean in pairs],
            "facet_name": [clean for _, clean in pairs],
            "raw_name": [raw for raw, _ in pairs],
        }
    )
    return df