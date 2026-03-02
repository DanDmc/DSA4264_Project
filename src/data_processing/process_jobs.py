"""
process_jobs.py

Purpose:
- Load a single raw MyCareersFuture job JSON file
- Convert it into a clean, standardized "processed job" record
- Print the processed record for manual inspection

Next steps:
- Add a function to process all JSON files in a folder into a CSV/Parquet.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Regex helpers
# -----------------------------

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Boilerplate phrases to remove (case-insensitive)
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
    "fair employment practices"
]


# -----------------------------
# Text Cleaning
# -----------------------------

def html_to_text(html: str) -> str:
    """
    Convert HTML job descriptions to readable plain text.
    - Strips tags
    - Decodes HTML entities
    - Normalizes whitespace
    """
    if not html:
        return ""

    # Preserve basic structure for readability
    html = re.sub(r"</(p|div|br|li)>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<(br)\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<li[^>]*>", "- ", html, flags=re.IGNORECASE)

    # Remove remaining tags
    text = TAG_RE.sub(" ", html)

    # Decode HTML entities (&nbsp;, &amp;, etc.)
    text = unescape(text)

    # Normalize whitespace but preserve line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join([line for line in text.split("\n") if line.strip()])

    return text.strip()


def remove_boilerplate_sentences(text: str) -> str:
    """
    Remove sentences/lines containing known HR/EEO boilerplate phrases.
    """
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


# -----------------------------
# Structured Field Extractors
# -----------------------------

def safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_salary(job: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    sal = job.get("salary") or {}
    smin = sal.get("minimum")
    smax = sal.get("maximum")

    try:
        smin_f = float(smin) if smin is not None else None
    except (TypeError, ValueError):
        smin_f = None

    try:
        smax_f = float(smax) if smax is not None else None
    except (TypeError, ValueError):
        smax_f = None

    if smin_f is not None and smax_f is not None:
        savg_f = (smin_f + smax_f) / 2.0
    else:
        savg_f = None

    return smin_f, smax_f, savg_f


def extract_skills(job: Dict[str, Any]) -> List[str]:
    skills = job.get("skills") or []
    out: List[str] = []

    for s in skills:
        if isinstance(s, dict):
            name = s.get("skill")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return deduped


def extract_category(job: Dict[str, Any]) -> Optional[str]:
    cats = job.get("categories") or []
    if cats and isinstance(cats, list) and isinstance(cats[0], dict):
        c = cats[0].get("category")
        return c.strip() if isinstance(c, str) and c.strip() else None
    return None


def parse_posting_date(job: Dict[str, Any]) -> Optional[str]:
    new_posting = safe_get(job, ["metadata", "newPostingDate"])
    if isinstance(new_posting, str) and new_posting.strip():
        return new_posting.strip()

    created_at = safe_get(job, ["metadata", "createdAt"])
    if isinstance(created_at, str) and created_at.strip():
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            return None

    return None


# -----------------------------
# Processed Job Dataclass
# -----------------------------

@dataclass
class ProcessedJob:
    # Text fields
    job_id: str
    title: str
    clean_description: str
    job_text: str
    skills_list: List[str]

    # Structured fields
    minimum_years_experience: Optional[int]
    ssoc_code: Optional[str]
    category: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    salary_avg: Optional[float]
    posting_date: Optional[str]


# -----------------------------
# Main Processing Logic
# -----------------------------

def process_one_job(job: Dict[str, Any]) -> ProcessedJob:
    job_id = (job.get("uuid") or "").strip()
    title = (job.get("title") or "").strip()

    raw_desc = job.get("description") or ""
    clean_desc = html_to_text(raw_desc)
    clean_desc = remove_boilerplate_sentences(clean_desc)

    # job_text = title + cleaned description
    pieces = []
    if title:
        pieces.append(title)
    if clean_desc:
        pieces.append(clean_desc)
    job_text = ". ".join(pieces).strip()

    skills_list = extract_skills(job)

    mye = job.get("minimumYearsExperience")
    try:
        mye_i = int(mye) if mye is not None else None
    except (TypeError, ValueError):
        mye_i = None

    ssoc_code = job.get("ssocCode")
    if isinstance(ssoc_code, str):
        ssoc_code = ssoc_code.strip() or None
    else:
        ssoc_code = None

    category = extract_category(job)

    salary_min, salary_max, salary_avg = parse_salary(job)
    posting_date = parse_posting_date(job)

    return ProcessedJob(
        job_id=job_id,
        title=title,
        clean_description=clean_desc,
        job_text=job_text,
        skills_list=skills_list,
        minimum_years_experience=mye_i,
        ssoc_code=ssoc_code,
        category=category,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_avg=salary_avg,
        posting_date=posting_date,
    )


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Process one MyCareersFuture job JSON file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single job JSON file (raw).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        job = json.load(f)

    processed = process_one_job(job)

    print(json.dumps(asdict(processed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()