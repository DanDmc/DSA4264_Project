"""
jobs_pipeline.py
================
End-to-end pipeline for processing raw job postings.

Each step can be run independently or as part of the full sequence.

  Step 1  parse        Parse raw JSON files from batch subfolders
  Step 2  ssoc         Enrich with SSOC 2024 hierarchy
  Step 3  filter       Filter to entry-level roles (≤ MAX_YEARS_EXPERIENCE)

Usage (from repo root)
-----
Run full pipeline (steps 1 → 2 → 3):
    python -m src.jobs_pipeline

Run specific steps only:
    python -m src.jobs_pipeline --steps 2 3

Data flow
---------
  raw/jobs/YYYYMMDD_YYYYMMDD/*.json
      │  step 1
      ▼
  processed/jobs/01_jobs_parsed.csv
      │  step 2
      ▼
  processed/jobs/02_jobs_ssoc_mapped.csv
      │  step 3
      ▼
  processed/jobs/03_jobs_filtered.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────
# Step functions — each wraps the corresponding script's main()
# ──────────────────────────────────────────────

def step_1_parse():
    """Parse raw job JSONs from batch subfolders into a single CSV."""
    from src.data_processing.process_jobs import main as run
    run()


def step_2_ssoc():
    """Enrich parsed jobs with SSOC 2024 hierarchy codes and titles."""
    from src.data_processing.ssoc_mapping import main as run
    run()


def step_3_filter():
    """Filter to entry-level roles based on years of experience."""
    from src.data_processing.final_jobs_filtering import main as run
    run()


# step registry — order matters
STEPS = {
    1: ("parse", step_1_parse),
    2: ("ssoc", step_2_ssoc),
    3: ("filter", step_3_filter),
}


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_pipeline(steps_to_run: list[int]) -> None:
    total_start = time.time()

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"Unknown step {step_num}. Valid steps: {list(STEPS.keys())}")
            sys.exit(1)

        label, func = STEPS[step_num]
        print(f"\n{'='*60}")
        print(f"  Step {step_num}: {label}")
        print(f"{'='*60}\n")

        step_start = time.time()
        func()
        elapsed = time.time() - step_start
        print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {total_elapsed:.1f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the jobs processing pipeline."
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(STEPS.keys()),
        help="Which steps to run (default: all). E.g. --steps 2 3",
    )
    args = parser.parse_args()
    run_pipeline(args.steps)


if __name__ == "__main__":
    main()
