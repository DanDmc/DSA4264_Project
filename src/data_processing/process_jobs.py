"""
process_jobs.py
===============
Step 1a of the jobs pipeline.

Reads raw MyCareersFuture job JSON files from batch subfolders under
JOBS_RAW_DIR (e.g. raw/jobs/20260125_20260131/), extracts fields into
a structured format, and writes a single consolidated CSV.

This script does MINIMAL transformation — just extraction and type
conversion. All text cleaning (HTML stripping, non-Latin removal,
boilerplate filtering, formula sanitisation) happens in clean_jobs.py.

Usage (from repo root):
    python -m src.data_processing.process_jobs

    # Test on specific job IDs only:
    python -m src.data_processing.process_jobs --job-ids ID1 ID2

Output:
    {JOBS_PROCESSED_DIR}/01a_jobs_extracted.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import shared paths from project config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import JOBS_RAW_DIR, JOBS_EXTRACTED


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
# Extracted Job Dataclass
# -----------------------------

@dataclass
class ExtractedJob:
    job_id: str
    title: str
    raw_description: str          # HTML as-is from JSON
    skills_list: List[str]
    minimum_years_experience: Optional[int]
    ssoc_code: Optional[str]
    category: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    salary_avg: Optional[float]
    posting_date: Optional[str]
    source_batch: str             # which batch folder this job came from


# -----------------------------
# Main Processing Logic
# -----------------------------

def extract_one_job(job: Dict[str, Any], source_batch: str) -> ExtractedJob:
    """Extract fields from a single raw job JSON dict."""
    job_id = (job.get("uuid") or "").strip()
    title = (job.get("title") or "").strip()

    # Keep raw HTML description — cleaning happens in clean_jobs.py
    raw_desc = job.get("description") or ""

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

    return ExtractedJob(
        job_id=job_id,
        title=title,
        raw_description=raw_desc,
        skills_list=skills_list,
        minimum_years_experience=mye_i,
        ssoc_code=ssoc_code,
        category=category,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_avg=salary_avg,
        posting_date=posting_date,
        source_batch=source_batch,
    )


def process_all_batches(
    jobs_raw_dir: Path,
    filter_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Scan all batch subfolders under jobs_raw_dir, extract every JSON file.

    Args:
        jobs_raw_dir: Root directory containing batch subfolders.
        filter_ids:   If provided, only jobs whose uuid is in this set are
                      returned. Useful for quick spot-checks.

    Returns:
        (rows, failed_files) where failed_files is a list of (path, error).
    """
    if not jobs_raw_dir.exists():
        raise FileNotFoundError(f"Jobs raw directory not found: {jobs_raw_dir}")

    batch_dirs = sorted([d for d in jobs_raw_dir.iterdir() if d.is_dir()])

    if not batch_dirs:
        raise FileNotFoundError(
            f"No batch subfolders found in {jobs_raw_dir}. "
            "Expected structure: raw/jobs/YYYYMMDD_YYYYMMDD/*.json"
        )

    rows: List[Dict[str, Any]] = []
    failed_files: List[Tuple[str, str]] = []

    for batch_dir in batch_dirs:
        batch_name = batch_dir.name
        json_files = sorted(batch_dir.glob("*.json"))
        print(f"Processing batch '{batch_name}': {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    job = json.load(f)

                # Skip if we're filtering to specific IDs
                if filter_ids is not None:
                    job_id = (job.get("uuid") or "").strip()
                    if job_id not in filter_ids:
                        continue

                extracted = extract_one_job(job, source_batch=batch_name)
                row = asdict(extracted)

                # Convert skills list to comma-separated string for CSV
                if isinstance(row.get("skills_list"), list):
                    row["skills_list"] = ", ".join(row["skills_list"])

                rows.append(row)

            except Exception as e:
                failed_files.append((f"{batch_name}/{json_file.name}", str(e)))

    return rows, failed_files


def save_rows_to_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save extracted job rows to CSV.

    Uses QUOTE_ALL so every field is wrapped in double-quotes — this prevents
    Excel from misinterpreting values that start with '-', '+', '=', or '@'
    as formulas (the #NAME? / #VALUE! problem).
    """
    if not rows:
        print("No rows to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} extracted jobs to: {output_path}")


# -----------------------------
# Entry point
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw job JSONs into a CSV (no cleaning).")
    parser.add_argument(
        "--job-ids",
        nargs="+",
        metavar="ID",
        help="If provided, only process jobs with these UUIDs (useful for spot-checks).",
    )
    args = parser.parse_args()

    filter_ids: Optional[Set[str]] = set(args.job_ids) if args.job_ids else None
    if filter_ids:
        print(f"Filtering to {len(filter_ids)} job ID(s): {filter_ids}")

    print(f"Scanning batch folders in: {JOBS_RAW_DIR}")
    rows, failed_files = process_all_batches(JOBS_RAW_DIR, filter_ids=filter_ids)

    save_rows_to_csv(rows, JOBS_EXTRACTED)

    print(f"\nExtracted: {len(rows)} jobs")
    if failed_files:
        print(f"Failed: {len(failed_files)} files")
        for filepath, error in failed_files[:10]:
            print(f"  - {filepath}: {error}")


if __name__ == "__main__":
    main()
