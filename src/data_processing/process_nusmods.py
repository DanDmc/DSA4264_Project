"""
process_nusmods.py
==================
Step 1 (and only step) of the courses pipeline.

Fetches NUS module data from the NUSMods API, filters placeholder
descriptions and graduate-level modules, and writes two CSVs:
  - 01_modules_raw.csv      (all modules, unfiltered)
  - 02_modules_cleaned.csv  (undergraduate, valid descriptions,
                              with boilerplate-stripped description_clean)

Usage (from repo root):
    python -m src.data_processing.process_nusmods

The original description column is always preserved so you can
compare before/after at any point.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import requests

# Import shared paths from project config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    NUSMODS_API_URL,
    MODULES_RAW,
    MODULES_CLEANED,
    EXCLUDED_MODULE_LEVELS,
    PLACEHOLDER_DESCRIPTIONS,
    EXCLUDED_DESCRIPTION_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------
# These patterns match pedagogical framing language that wraps module
# descriptions. Removing them lets SkillNer focus on content words.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(this|the) (course|module) "
        r"(introduces?|covers?|provides?|aims?|explores?|examines?|focuses?|"
        r"seeks?|equips?|prepares?|enables?|emphasises?|emphasizes?|discusses?|"
        r"addresses?|builds?|develops?|considers?|investigates?|surveys?|"
        r"reviews?|offers?|presents?|teaches?|trains?|imparts?|furnishes?|"
        r"is about|is designed to|will|consists of|serves as)\s+",
        r"(this|the) (course|module)\s+",
        r"students (will|are (expected to|required to|encouraged to))\s+\w+\s+",
        r"(upon completion|by the end of (this )?(course|module)),?\s+"
        r"students (will|should)\s+",
        r"(learners?|participants?) (will|are)\s+",
        r"will be (covered|discussed|explored|examined|introduced|addressed|"
        r"presented|taught|treated|studied|considered|analysed|analyzed)\s*",
        r"with (an? )?(emphasis|focus|attention|aim|objective|goal)\s+on\s+",
        r"(the )?(aim|goal|objective|purpose) "
        r"(of this (course|module) )?(is|are)\s+(to\s+)?",
        r"in this (course|module),?\s+",
    ]
]


def strip_boilerplate(text: str) -> str:
    """Remove pedagogical boilerplate phrases from a description."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_module_credit(value: object) -> float | None:
    """Parse module credit to float; return None when not parseable."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# API fetch & field extraction
# ---------------------------------------------------------------------------

def fetch_module_data(api_url: str) -> list[dict]:
    """Fetch raw module list from the NUSMods v2 API."""
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    return response.json()


def extract_fields(data: list[dict]) -> list[dict]:
    """Select the fields we need and add a preprocessed description column."""
    records = []
    for item in data:
        description = item.get("description") or ""
        module_credit = parse_module_credit(item.get("moduleCredit"))
        records.append({
            "module code":       item.get("moduleCode"),
            "title":             item.get("title"),
            "description":       description,
            "description_clean": strip_boilerplate(description),
            "department":        item.get("department"),
            "faculty":           item.get("faculty"),
            "module_credit":     module_credit,
        })
    return records


# ---------------------------------------------------------------------------
# Description validity filter
# ---------------------------------------------------------------------------

def _normalize_description(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _is_valid_description(description: object) -> bool:
    """Return True for descriptions worth extracting skills from.

    Excludes placeholder descriptions (e.g. 'not available', 'nil') and
    modules containing excluded keywords (e.g. 'internship'). Both lists
    are configurable in src/config.py.
    """
    normalized = _normalize_description(description)
    if not normalized:
        return False
    if any(kw in normalized for kw in EXCLUDED_DESCRIPTION_KEYWORDS):
        return False
    if normalized in PLACEHOLDER_DESCRIPTIONS:
        return False
    if re.fullmatch(r"(faculty )?exchange course( - yus \(1 unit\))?", normalized):
        return False
    return True


def _is_undergraduate_level(module_code: object) -> bool:
    """Return True if module level is not in EXCLUDED_MODULE_LEVELS.

    NUS module codes encode level in the first digit of the numeric part.
    Excluded levels are configurable in src/config.py (default: 5, 6).
    """
    if module_code is None or (isinstance(module_code, float) and pd.isna(module_code)):
        return False
    m = re.search(r"(\d)", str(module_code))
    if not m:
        return False
    return int(m.group(1)) not in EXCLUDED_MODULE_LEVELS


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_to_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def save_cleaned_csv(records: list[dict], path: Path) -> tuple[int, int]:
    """Write cleaned modules filtered by description validity and level.

    Returns (kept, removed).
    """
    df = pd.DataFrame(records)
    cleaned = df[
        df["description"].apply(_is_valid_description)
        & df["module code"].apply(_is_undergraduate_level)
    ].copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False)
    return len(cleaned), len(df) - len(cleaned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch NUSMods data and write raw + cleaned CSVs."""
    print(f"Fetching modules from {NUSMODS_API_URL}")
    raw_data = fetch_module_data(NUSMODS_API_URL)
    extracted = extract_fields(raw_data)

    save_to_csv(extracted, MODULES_RAW)
    cleaned_count, removed_count = save_cleaned_csv(extracted, MODULES_CLEANED)

    print(f"  Fetched       {len(raw_data):>6,} modules from NUSMods API")
    print(f"  Saved raw     {len(extracted):>6,} records  → {MODULES_RAW}")
    print(
        f"  Saved cleaned {cleaned_count:>6,} records  → {MODULES_CLEANED}"
        f"  (removed {removed_count} filtered modules)"
    )


if __name__ == "__main__":
    main()
