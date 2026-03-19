"""
ssoc_mapping.py
───────────────
Enrich jobs_processed.csv with SSOC title and definition columns
from the official SSOC 2024 detailed definitions spreadsheet.

Writes to a NEW file (jobs_processed_ssoc_mapped.csv) so we don't
touch the original.

Usage:
    python src/data_processing/ssoc_mapping.py

Reads:
    - data/raw/ssoc2024-detailed-definitions.xlsx
    - data/processed/jobs_processed.csv

Writes:
    - data/processed/jobs_processed_ssoc_mapped.csv
"""

import pandas as pd
from pathlib import Path

# paths relative to project root — run this from there
RAW_SSOC = Path("data/raw/ssoc2024-detailed-definitions.xlsx")
JOBS_INPUT = Path("data/processed/jobs_processed.csv")
JOBS_OUTPUT = Path("data/processed/jobs_processed_ssoc_mapped.csv")


def load_ssoc_lookup(path: Path) -> pd.DataFrame:
    """
    Parse the SSOC 2024 spreadsheet into a clean lookup table.
    The xlsx has 4 junk rows before the real header, and codes at
    multiple granularity levels (1-5 digits). We keep all of them
    so we can do fallback matching.
    """
    df = pd.read_excel(path, header=None, skiprows=4)
    df.columns = [
        "ssoc_code", "ssoc_title", "groups_classified",
        "detailed_definitions", "tasks", "notes",
        "examples_classified", "examples_elsewhere"
    ]

    # only keep what we need
    df = df[["ssoc_code", "ssoc_title", "detailed_definitions"]].copy()
    df = df.dropna(subset=["ssoc_code"])

    # normalize codes to strings — some come in as ints, some as floats
    df["ssoc_code"] = df["ssoc_code"].astype(str).str.strip()
    df["ssoc_code"] = df["ssoc_code"].str.replace(r"\.0$", "", regex=True)

    return df


def build_lookup_dict(ssoc_df: pd.DataFrame) -> dict:
    """
    Build a dict keyed by ssoc_code → (title, definition).
    Multiple granularity levels coexist — 5-digit is most specific,
    4-digit is the unit group, etc.
    """
    lookup = {}
    for _, row in ssoc_df.iterrows():
        code = row["ssoc_code"]
        lookup[code] = (row["ssoc_title"], row["detailed_definitions"])
    return lookup


def map_ssoc(code, lookup: dict) -> tuple:
    """
    Try to match a job's ssoc_code against the lookup.
    Strategy:
      1. Exact match on 5-digit code (most jobs should hit here)
      2. Fall back to 4-digit (unit group level)
      3. Fall back to 3-digit, 2-digit, 1-digit
      4. Give up → (NaN, NaN)

    This handles cases where the job has a code that's slightly
    off or uses a different granularity than the reference.
    """
    code_str = str(code).strip()

    # try exact, then progressively shorter prefixes
    for length in [len(code_str), 4, 3, 2, 1]:
        prefix = code_str[:length]
        if prefix in lookup:
            return lookup[prefix]

    return (pd.NA, pd.NA)


def main():
    print(f"Loading SSOC definitions from {RAW_SSOC}")
    ssoc_df = load_ssoc_lookup(RAW_SSOC)
    print(f"  → {len(ssoc_df)} SSOC entries loaded")

    lookup = build_lookup_dict(ssoc_df)

    print(f"Loading jobs from {JOBS_INPUT}")
    jobs = pd.read_csv(JOBS_INPUT)
    print(f"  → {len(jobs):,} jobs")

    # normalize the ssoc_code column in jobs to string
    jobs["ssoc_code"] = jobs["ssoc_code"].astype(str).str.strip()
    jobs["ssoc_code"] = jobs["ssoc_code"].str.replace(r"\.0$", "", regex=True)

    # do the mapping
    mapped = jobs["ssoc_code"].apply(lambda c: map_ssoc(c, lookup))
    jobs["ssoc_title"] = mapped.apply(lambda x: x[0])
    jobs["ssoc_description"] = mapped.apply(lambda x: x[1])

    # quick report on match quality
    matched = jobs["ssoc_title"].notna().sum()
    total = len(jobs)
    print(f"\nMatch results: {matched:,}/{total:,} ({matched/total*100:.1f}%)")

    # show what didn't match, if any
    unmatched_codes = jobs.loc[jobs["ssoc_title"].isna(), "ssoc_code"].unique()
    if len(unmatched_codes) > 0:
        print(f"Unmatched codes ({len(unmatched_codes)}): {unmatched_codes[:20]}")

    # save to a NEW file — never overwrite the original
    jobs.to_csv(JOBS_OUTPUT, index=False)
    print(f"\nSaved to {JOBS_OUTPUT} (original untouched)")
    print(f"New columns: ssoc_title, ssoc_description")


if __name__ == "__main__":
    main()
