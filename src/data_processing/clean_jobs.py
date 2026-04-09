"""
clean_jobs.py
=============
Step 1b of the jobs pipeline.

Applies text cleaning to the extracted jobs CSV:
  - HTML to plain text conversion
  - Non-Latin script removal (CJK, Arabic, Thai, etc.)
  - Emoji removal
  - Boilerplate phrase removal (TAFEP, equal opportunity, etc.)
  - Formula-injection character sanitisation
  - Constructs job_text (clean_title + cleaned description)
"""

from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import JOBS_EXTRACTED, JOBS_CLEANED


# -----------------------------
# Regex helpers
# -----------------------------

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

NON_LATIN_SCRIPT_RE = re.compile(
    r"["
    r"\u1100-\u11FF"
    r"\u2E80-\u2FFF"
    r"\u3000-\u303F"
    r"\u3040-\u309F"
    r"\u30A0-\u30FF"
    r"\u3100-\u312F"
    r"\u3130-\u318F"
    r"\u3200-\u32FF"
    r"\u3400-\u4DBF"
    r"\u4E00-\u9FFF"
    r"\uA000-\uA4CF"
    r"\uA960-\uA97F"
    r"\uAC00-\uD7FF"
    r"\uF900-\uFAFF"
    r"\uFE30-\uFE4F"
    r"\u0600-\u06FF"
    r"\u0590-\u05FF"
    r"\u0E00-\u0E7F"
    r"\u0900-\u097F"
    r"\u0980-\u09FF"
    r"\u0A00-\u0A7F"
    r"\u0A80-\u0AFF"
    r"\u0B00-\u0B7F"
    r"\u0B80-\u0BFF"
    r"\u0C00-\u0C7F"
    r"\u0C80-\u0CFF"
    r"\u0D00-\u0D7F"
    r"]+"
)

EMOJI_RE = re.compile(
    r"["
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002600-\U000026FF"
    r"\U00002700-\U000027BF"
    r"\U0000FE00-\U0000FE0F"
    r"\U0000200D"
    r"]+"
)

FORMULA_START_RE = re.compile(r"^[=+@\-]+")

BOILERPLATE_PHRASES = [
    "tafep",
    "employers pledge",
    "equal opportunity employer",
    "equal opportunity",
    "non-discrimination",
    "regardless of gender",
    "regardless of race",
    "regardless of ethnicity",
    "regardless of religion",
    "fair employment practices",
]


# -----------------------------
# Core Cleaning Functions
# -----------------------------

def normalize_text(text: str) -> str:
    """Basic whitespace cleanup."""
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def remove_emojis(text: str) -> str:
    if not text:
        return ""
    return EMOJI_RE.sub("", text)


def remove_non_latin_script(text: str) -> str:
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        stripped = NON_LATIN_SCRIPT_RE.sub(" ", line)
        stripped = WHITESPACE_RE.sub(" ", stripped).strip()
        if stripped:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def remove_boilerplate_sentences(text: str) -> str:
    if not text:
        return ""

    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    keep: List[str] = []

    for p in parts:
        p_lower = p.lower()
        if any(phrase in p_lower for phrase in BOILERPLATE_PHRASES):
            continue
        keep.append(p)

    return "\n".join(keep).strip()


def sanitise_formula_start(value: str) -> str:
    if not value:
        return ""
    return FORMULA_START_RE.sub("", value)


def excel_safe_wrap(value: str) -> str:
    if not value:
        return ""
    if value[0] in "-+=@":
        return "'" + value
    return value


# -----------------------------
# Title Cleaning (NEW)
# -----------------------------

def clean_title(title: str) -> str:
    """
    Clean job title:
    - Remove emojis
    - Remove non-Latin scripts
    - Remove boilerplate
    - Normalize whitespace
    - Excel safety
    """
    if not title:
        return ""

    text = remove_emojis(title)
    text = remove_non_latin_script(text)
    text = remove_boilerplate_sentences(text)
    text = normalize_text(text)
    text = sanitise_formula_start(text)
    text = excel_safe_wrap(text)

    return text


# -----------------------------
# Description Cleaning
# -----------------------------

def html_to_text(html: str) -> str:
    if not html:
        return ""

    html = re.sub(r"</(p|div|br|li)>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<(br)\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<li[^>]*>", "- ", html, flags=re.IGNORECASE)

    text = TAG_RE.sub(" ", html)
    text = unescape(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join([line for line in text.split("\n") if line.strip()])

    return text.strip()


def clean_description(raw_html: str) -> str:
    text = html_to_text(raw_html)
    text = remove_emojis(text)
    text = remove_non_latin_script(text)
    text = remove_boilerplate_sentences(text)
    text = sanitise_formula_start(text)
    text = excel_safe_wrap(text)
    return text


# -----------------------------
# Job Text Construction
# -----------------------------

def build_job_text(clean_title_val: str, clean_desc: str) -> str:
    pieces = []
    if clean_title_val:
        pieces.append(clean_title_val.lstrip("'").strip())
    if clean_desc:
        pieces.append(clean_desc.lstrip("'").strip())

    result = ". ".join(pieces).strip()
    return excel_safe_wrap(result)


# -----------------------------
# Main Processing Logic
# -----------------------------

def clean_jobs_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # NEW: clean title
    df["clean_title"] = df["title"].fillna("").apply(clean_title)

    # Existing: clean description
    df["clean_description"] = df["raw_description"].fillna("").apply(clean_description)

    # Build job_text using cleaned title
    df["job_text"] = df.apply(
        lambda row: build_job_text(row["clean_title"], row["clean_description"]),
        axis=1
    )

    # Reorder columns
    cols = list(df.columns)
    raw_idx = cols.index("raw_description")

    cols.remove("clean_title")
    cols.remove("clean_description")
    cols.remove("job_text")

    cols.insert(raw_idx + 1, "clean_title")
    cols.insert(raw_idx + 2, "clean_description")
    cols.insert(raw_idx + 3, "job_text")

    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean extracted job descriptions.")
    parser.add_argument(
        "--job-ids",
        nargs="+",
        metavar="ID",
        help="If provided, only process jobs with these UUIDs.",
    )
    args = parser.parse_args()

    filter_ids: Optional[Set[str]] = set(args.job_ids) if args.job_ids else None

    print(f"Loading extracted jobs from {JOBS_EXTRACTED}")
    df = pd.read_csv(JOBS_EXTRACTED, encoding="utf-8-sig")
    print(f"  → {len(df):,} jobs loaded")

    if filter_ids:
        df = df[df["job_id"].isin(filter_ids)]

    print("Applying cleaning transformations...")
    df_cleaned = clean_jobs_df(df)

    JOBS_CLEANED.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(JOBS_CLEANED, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df_cleaned):,} cleaned jobs to: {JOBS_CLEANED}")


if __name__ == "__main__":
    main()