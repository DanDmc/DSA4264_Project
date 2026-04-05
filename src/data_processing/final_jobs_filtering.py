"""
final_jobs_filtering.py
=======================
Step 3 of the jobs pipeline.

Filters SSOC-mapped jobs to entry-level roles based on
MAX_YEARS_EXPERIENCE (default: 2). This threshold is configurable
in src/config.py so MOE officers can adjust it without editing
pipeline code.

Usage (from repo root):
    python -m src.data_processing.final_jobs_filtering

Reads:
    - JOBS_SSOC_MAPPED (processed/jobs/02_jobs_ssoc_mapped.csv)

Writes:
    - JOBS_FILTERED    (processed/jobs/03_jobs_filtered.csv)
"""

import sys
from pathlib import Path

import pandas as pd

# Import shared paths and parameters from project config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import JOBS_SSOC_MAPPED, JOBS_FILTERED, MAX_YEARS_EXPERIENCE


def main():
    print(f"Loading jobs from {JOBS_SSOC_MAPPED}")
    df = pd.read_csv(JOBS_SSOC_MAPPED)
    print(f"  → {len(df):,} jobs before filtering")

    # filter to entry-level roles
    filtered_df = df[df["minimum_years_experience"] <= MAX_YEARS_EXPERIENCE]
    print(f"  → {len(filtered_df):,} jobs after filtering (≤{MAX_YEARS_EXPERIENCE} years experience)")

    JOBS_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(JOBS_FILTERED, index=False)
    print(f"Saved to {JOBS_FILTERED}")


if __name__ == "__main__":
    main()
