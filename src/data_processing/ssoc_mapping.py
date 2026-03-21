"""
ssoc_mapping.py
───────────────
Enrich jobs_processed.csv with the full SSOC 2024 hierarchy:
  - Major group       (1-digit) → code + title
  - Sub-major group   (2-digit) → code + title
  - Minor group       (3-digit) → code + title
  - Unit group        (4-digit) → code + title
  - Occupation        (5-digit) → code + title + detailed definition

Writes to a NEW file so we don't touch the original.

Usage:
    python src/data_processing/ssoc_mapping.py

Reads:
    - data/raw/ssoc2024-detailed-definitions.xlsx
    - data/processed/jobs_processed.csv

Writes:
    - data/processed/jobs_processed_ssoc_mapped.csv
"""

import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# paths relative to project root
RAW_SSOC = Path("data/raw/ssoc2024-detailed-definitions.xlsx")
JOBS_INPUT = Path("data/processed/jobs_processed.csv")
JOBS_OUTPUT = Path("data/processed/jobs_processed_ssoc_mapped.csv")

# the five hierarchy levels — digit length, column prefix
LEVELS = [
    (1, "ssoc_major"),
    (2, "ssoc_submajor"),
    (3, "ssoc_minor"),
    (4, "ssoc_unit"),
    (5, "ssoc_occupation"),
]


def load_ssoc_reference(path: Path) -> pd.DataFrame:
    """
    Parse the SSOC 2024 spreadsheet. Header is on row 5 (0-indexed row 4),
    so we skip the first 4 rows of metadata.
    """
    df = pd.read_excel(path, header=None, skiprows=4)
    df.columns = [
        "ssoc_code", "ssoc_title", "groups_classified",
        "detailed_definitions", "tasks", "notes",
        "examples_classified", "examples_elsewhere"
    ]

    df = df[["ssoc_code", "ssoc_title", "detailed_definitions"]].copy()
    df = df.dropna(subset=["ssoc_code"])

    # normalise codes to clean strings
    df["ssoc_code"] = (df["ssoc_code"].astype(str).str.strip()
                       .str.replace(r"\.0$", "", regex=True))
    df["code_len"] = df["ssoc_code"].str.len()

    return df


def build_level_lookups(ssoc_df: pd.DataFrame) -> dict:
    """
    Build a separate lookup dict for each hierarchy level.
    Returns {digit_length: {code: (title, definition)}}

    Definitions only exist at the 4- and 5-digit level in the spreadsheet,
    so they'll be NaN for 1-3 digit codes (which is fine — we only use
    the definition from the 5-digit occupation level).
    """
    lookups = {}
    for n_digits, _ in LEVELS:
        level_df = ssoc_df[ssoc_df["code_len"] == n_digits]
        lookups[n_digits] = {
            row["ssoc_code"]: (row["ssoc_title"], row["detailed_definitions"])
            for _, row in level_df.iterrows()
        }
    return lookups


def enrich_jobs(jobs: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    """
    For each job, slice its 5-digit ssoc_code into prefixes and
    look up the title (and definition for level 5) at each level.
    """
    for n_digits, col_prefix in LEVELS:
        lookup = lookups[n_digits]

        # slice the code to the right length
        prefixes = jobs["ssoc_code"].str[:n_digits]

        # vectorised lookup via map — much faster than row-wise apply
        mapped = prefixes.map(lookup)

        jobs[f"{col_prefix}_code"] = prefixes
        jobs[f"{col_prefix}_title"] = mapped.apply(
            lambda x: x[0] if isinstance(x, tuple) else pd.NA
        )

    # detailed definition only makes sense at the occupation (5-digit) level
    occ_lookup = lookups[5]
    jobs["ssoc_occupation_description"] = jobs["ssoc_code"].map(occ_lookup).apply(
        lambda x: x[1] if isinstance(x, tuple) else pd.NA
    )

    return jobs


def main():
    print(f"Loading SSOC definitions from {RAW_SSOC}")
    ssoc_df = load_ssoc_reference(RAW_SSOC)
    print(f"  → {len(ssoc_df)} entries across all levels")

    for n_digits, label in LEVELS:
        count = (ssoc_df["code_len"] == n_digits).sum()
        print(f"    {label:20s} ({n_digits}-digit): {count:>5} entries")

    lookups = build_level_lookups(ssoc_df)

    print(f"\nLoading jobs from {JOBS_INPUT}")
    jobs = pd.read_csv(JOBS_INPUT)
    print(f"  → {len(jobs):,} jobs")

    # normalise ssoc_code in jobs
    jobs["ssoc_code"] = (jobs["ssoc_code"].astype(str).str.strip()
                         .str.replace(r"\.0$", "", regex=True))

    jobs = enrich_jobs(jobs, lookups)

    # ── match report ──────────────────────────────────────────────────────
    print("\nMatch results by level:")
    for _, col_prefix in LEVELS:
        matched = jobs[f"{col_prefix}_title"].notna().sum()
        print(f"  {col_prefix:20s}: {matched:,}/{len(jobs):,} "
              f"({matched/len(jobs)*100:.1f}%)")

    unmatched = jobs.loc[jobs["ssoc_occupation_title"].isna(), "ssoc_code"].unique()
    if len(unmatched) > 0:
        print(f"\nUnmatched 5-digit codes ({len(unmatched)}): {unmatched[:20]}")

    # save — never overwrite the original
    jobs.to_csv(JOBS_OUTPUT, index=False)
    print(f"\nSaved to {JOBS_OUTPUT}")

    # show what columns got added
    new_cols = [c for c in jobs.columns if c.startswith("ssoc_") and c != "ssoc_code"]
    print(f"New columns: {new_cols}")


if __name__ == "__main__":
    main()