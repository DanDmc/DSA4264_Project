"""
process_nusmods.py  (v2)
========================
Fetches NUS module data from the NUSMods API, filters placeholder descriptions,
and writes two CSVs: a raw file and a cleaned file.
 
Improvements over v1
--------------------
* Adds a `description_clean` column that strips pedagogical boilerplate
  ("This course introduces...", "Students will learn...", etc.) before
  the text reaches SkillNer.  SkillNer sees cleaner, skills-dense text
  and produces less noise as a result.
* The original `description` column is always preserved so you can
  compare before/after at any point.
* Minor: type hints added, docstrings improved.
 
Output
------
  modules.csv         -- raw extract, all modules
  cleaned_modules.csv -- filtered (no internships / placeholders) + description_clean
"""
 
import re
import requests
import pandas as pd
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
URL = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"
OUTPUT_RAW_PATH = "modules.csv"
OUTPUT_CLEANED_PATH = "cleaned_modules.csv"
 
 
# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------
# These patterns match the pedagogical framing language that wraps module
# descriptions.  They add noise to any NLP extractor because they contain
# verbs and nouns that look like skill tokens but are just course-structure
# language.  Removing them lets SkillNer focus on the content words.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # "This/The course/module <verb> ..."
        r"(this|the) (course|module) "
        r"(introduces?|covers?|provides?|aims?|explores?|examines?|focuses?|"
        r"seeks?|equips?|prepares?|enables?|emphasises?|emphasizes?|discusses?|"
        r"addresses?|builds?|develops?|considers?|investigates?|surveys?|"
        r"reviews?|offers?|presents?|teaches?|trains?|imparts?|furnishes?|"
        r"is about|is designed to|will|consists of|serves as)\s+",
        # Leftover bare "this/the course/module" after above
        r"(this|the) (course|module)\s+",
        # "Students will/are expected to <verb> ..."
        r"students (will|are (expected to|required to|encouraged to))\s+\w+\s+",
        # "Upon completion, students will ..."
        r"(upon completion|by the end of (this )?(course|module)),?\s+"
        r"students (will|should)\s+",
        r"(learners?|participants?) (will|are)\s+",
        # Passive end-of-clause: "will be discussed / examined / ..."
        r"will be (covered|discussed|explored|examined|introduced|addressed|"
        r"presented|taught|treated|studied|considered|analysed|analyzed)\s*",
        # "with (an) emphasis/focus on ..."
        r"with (an? )?(emphasis|focus|attention|aim|objective|goal)\s+on\s+",
        # "The aim of this course is to ..."
        r"(the )?(aim|goal|objective|purpose) "
        r"(of this (course|module) )?(is|are)\s+(to\s+)?",
        # "In this course, ..."
        r"in this (course|module),?\s+",
    ]
]
 
 
def strip_boilerplate(text: str) -> str:
    """Remove pedagogical boilerplate phrases from a description.
 
    Applies patterns iteratively (order matters: broader patterns first)
    then collapses any resulting double-spaces.
    """
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
    """Select the fields we need and add a preprocessed description column.
 
    Fields captured
    ---------------
    module code        : e.g. CS3230
    title              : module title
    description        : original description (always preserved)
    description_clean  : boilerplate-stripped description for SkillNer
    department         : owning department (e.g. Computer Science)
    faculty            : owning faculty (e.g. School of Computing)
    module_credit      : number of MCs — used for MC-weighted metrics
    """
    records = []
    for item in data:
        description = item.get("description") or ""
        module_credit = parse_module_credit(item.get("moduleCredit"))
        records.append({
            "module code":           item.get("moduleCode"),
            "title":                 item.get("title"),
            "description":           description,
            "description_clean":     strip_boilerplate(description),
            "department":            item.get("department"),
            "faculty":               item.get("faculty"),
            "module_credit":         module_credit,
        })
    return records
 
 
# ---------------------------------------------------------------------------
# Description validity filter  (logic unchanged from v1)
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
    """Return True for descriptions that are worth extracting skills from."""
    normalized = _normalize_description(description)
    if not normalized:
        return False
    if "internship" in normalized:
        return False
    if normalized in {
        "not available",
        "not available.",
        "not applicable",
        "unrestricted elective",
        "nil",
        "department exchange course",
        "advance placement credit",
    }:
        return False
    if re.fullmatch(r"(faculty )?exchange course( - yus \(1 unit\))?", normalized):
        return False
    return True


def _is_undergraduate_level(module_code: object) -> bool:
    """Return True if module is not 5000 or 6000 level.

    NUS module codes encode level in the first digit of the numeric part.
    5000 and 6000 level modules are graduate courses — excluded from cleaned output.
    """
    if module_code is None or (isinstance(module_code, float) and pd.isna(module_code)):
        return False
    m = re.search(r"(\d)", str(module_code))
    if not m:
        return False
    return int(m.group(1)) not in {5, 6}
 
 
# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
 
def save_to_csv(records: list[dict], path: str) -> None:
    pd.DataFrame(records).to_csv(path, index=False)
 
 
def save_cleaned_csv(records: list[dict], path: str) -> tuple[int, int]:
    """Write cleaned modules filtered by description validity and undergraduate level.

    Keeps modules that:
    - Have a valid, non-placeholder description
    - Are undergraduate level (module code first digit 1-4, excludes 5000+)

    Returns (kept, removed).
    """
    df = pd.DataFrame(records)
    cleaned = df[
        df["description"].apply(_is_valid_description)
        & df["module code"].apply(_is_undergraduate_level)
    ].copy()
    cleaned.to_csv(path, index=False)
    return len(cleaned), len(df) - len(cleaned)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    raw_data = fetch_module_data(URL)
    extracted = extract_fields(raw_data)
 
    Path(OUTPUT_RAW_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(extracted, OUTPUT_RAW_PATH)
 
    cleaned_count, removed_count = save_cleaned_csv(extracted, OUTPUT_CLEANED_PATH)
 
    print(f"Fetched       {len(raw_data):>6,} modules from NUSMods API")
    print(f"Saved raw     {len(extracted):>6,} records  -> {OUTPUT_RAW_PATH}")
    print(
        f"Saved cleaned {cleaned_count:>6,} records  -> {OUTPUT_CLEANED_PATH}"
        f"  (removed {removed_count} filtered modules)"
    )
    print()
    print("Next step: run SkillNer on the 'description_clean' column of")
    print(f"  {OUTPUT_CLEANED_PATH}")
    print("  New fields available: department, faculty, module_credit,")
    print("  fulfill_requirements — pass cleaned_modules.csv to")
    print("  extract_course_skills.py, then hard_soft_skills.py.")
 
 
if __name__ == "__main__":
    main()
    
