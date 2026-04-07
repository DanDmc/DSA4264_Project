"""
01_jobs_processing.py
=====================
End-to-end pipeline for processing raw job postings.

Each step can be run independently or as part of the full sequence.

  Step 1a  extract     Extract raw JSON files from batch subfolders (no cleaning)
  Step 1b  clean       Apply text cleaning (HTML→text, non-Latin removal, etc.)
  Step 2   ssoc        Enrich with SSOC 2024 hierarchy
  Step 3   filter      Filter to entry-level roles (≤ MAX_YEARS_EXPERIENCE)

Usage (from repo root)
-----
Run full pipeline (steps 1a → 1b → 2 → 3):
    python -m src.pipelines.01_jobs_processing

Run specific steps only:
    python -m src.pipelines.01_jobs_processing --steps 1b 2 3

Data flow
---------
  raw/jobs/YYYYMMDD_YYYYMMDD/*.json
      │  step 1a (extract)
      ▼
  processed/jobs/01a_jobs_extracted.csv
      │  step 1b (clean)
      ▼
  processed/jobs/01b_jobs_cleaned.csv
      │  step 2 (ssoc)
      ▼
  processed/jobs/02_jobs_ssoc_mapped.csv
      │  step 3 (filter)
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

def step_1a_extract():
    """Extract raw job JSONs from batch subfolders into a single CSV."""
    from src.data_processing.process_jobs import main as run
    run()


def step_1b_clean():
    """Apply text cleaning to extracted jobs."""
    from src.data_processing.clean_jobs import main as run
    run()


def step_2_ssoc():
    """Enrich cleaned jobs with SSOC 2024 hierarchy codes and titles."""
    from src.data_processing.ssoc_mapping import main as run
    run()


def step_3_filter():
    """Filter to entry-level roles based on years of experience."""
    from src.data_processing.final_jobs_filtering import main as run
    run()


# step registry — order matters
# Using string keys to support "1a", "1b" notation
STEPS = {
    "1a": ("extract", step_1a_extract),
    "1b": ("clean", step_1b_clean),
    "2": ("ssoc", step_2_ssoc),
    "3": ("filter", step_3_filter),
}

# Default order when running all steps
DEFAULT_STEP_ORDER = ["1a", "1b", "2", "3"]


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_pipeline(steps_to_run: list[str]) -> None:
    total_start = time.time()

    for step_key in steps_to_run:
        if step_key not in STEPS:
            print(f"Unknown step '{step_key}'. Valid steps: {list(STEPS.keys())}")
            sys.exit(1)

        label, func = STEPS[step_key]
        print(f"\n{'='*60}")
        print(f"  Step {step_key}: {label}")
        print(f"{'='*60}\n")

        step_start = time.time()
        func()
        elapsed = time.time() - step_start
        print(f"\n  ✓ Step {step_key} completed in {elapsed:.1f}s")

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
        type=str,
        nargs="+",
        default=DEFAULT_STEP_ORDER,
        help="Which steps to run (default: all). E.g. --steps 1b 2 3",
    )
    args = parser.parse_args()
    run_pipeline(args.steps)


if __name__ == "__main__":
    main()
