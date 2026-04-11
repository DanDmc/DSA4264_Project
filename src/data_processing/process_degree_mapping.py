"""
Process degree mapping: explode raw Excel into a flat lookup table.

Reads the degree mapping Excel file (one row per major, comma-separated
module codes) and produces a normalised CSV with one row per
(university, faculty, major, module_code, module_type) combination.

Usage:
    python -m src.data_processing.process_degree_mapping

Output:
    processed/courses/degree_module_mapping.csv
"""

import sys
import pandas as pd

from src.config import (
    DEGREE_MAPPING_RAW,
    DEGREE_MODULE_MAPPING,
    MODULES_CLEANED,
)


def load_degree_mapping(path):
    """Read the raw degree mapping Excel file."""
    df = pd.read_excel(path, engine="openpyxl")
    expected = {"University", "Faculty", "Major", "Core Modules", "Electives"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        print(f"ERROR: Missing columns in degree mapping: {missing}")
        sys.exit(1)
    # Drop fully blank rows
    df = df.dropna(subset=["University", "Faculty", "Major"], how="any")
    return df


def explode_modules(df):
    """Explode comma-separated module codes into individual rows."""
    records = []
    for _, row in df.iterrows():
        uni = row["University"]
        fac = row["Faculty"]
        major = row["Major"]

        for module_type, col in [("core", "Core Modules"), ("elective", "Electives")]:
            codes_str = row[col]
            if pd.isna(codes_str) or not str(codes_str).strip():
                continue
            for code in str(codes_str).split(","):
                code = code.strip()
                if code:
                    records.append({
                        "university": uni,
                        "faculty": fac,
                        "major": major,
                        "module_code": code,
                        "module_type": module_type,
                    })

    result = pd.DataFrame(records)
    # Deduplicate (same module listed twice for same major)
    result = result.drop_duplicates()
    return result


def validate_against_modules(mapping_df, modules_path):
    """Check how many mapped module codes exist in modules_cleaned.csv."""
    try:
        modules = pd.read_csv(modules_path)
    except FileNotFoundError:
        print(f"  WARNING: {modules_path} not found — skipping validation.")
        return

    module_codes_in_catalogue = set(modules["module code"].str.strip())
    mapped_codes = set(mapping_df["module_code"])

    matched = mapped_codes & module_codes_in_catalogue
    unmatched = mapped_codes - module_codes_in_catalogue

    print(f"\n  Validation against modules_cleaned.csv:")
    print(f"    Module codes in degree mapping:  {len(mapped_codes)}")
    print(f"    Matched in modules catalogue:    {len(matched)}")
    print(f"    Not found in catalogue:          {len(unmatched)}")

    if unmatched:
        # Show a sample — these could be grad modules filtered out, typos, etc.
        sample = sorted(unmatched)[:20]
        print(f"    Sample unmatched: {sample}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")


def main():
    print(f"Reading degree mapping from: {DEGREE_MAPPING_RAW}")
    if not DEGREE_MAPPING_RAW.exists():
        print(f"ERROR: File not found: {DEGREE_MAPPING_RAW}")
        print("Place the degree mapping Excel file in raw/ and update config.py if needed.")
        sys.exit(1)

    raw = load_degree_mapping(DEGREE_MAPPING_RAW)
    print(f"  Loaded {len(raw)} majors")

    mapping = explode_modules(raw)
    print(f"  Exploded to {len(mapping)} (major, module, type) rows")
    print(f"  Unique module codes: {mapping['module_code'].nunique()}")
    print(f"  Unique majors:       {mapping['major'].nunique()}")
    print(f"  Universities:        {mapping['university'].unique().tolist()}")

    # Summary by major
    summary = (
        mapping
        .groupby(["university", "faculty", "major", "module_type"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"\n  Modules per major (core / elective):")
    for (uni, fac, major), counts in summary.iterrows():
        core_n = counts.get("core", 0)
        elec_n = counts.get("elective", 0)
        print(f"    {major:<45s}  core={core_n:<4d}  elective={elec_n}")

    # Validate against module catalogue
    validate_against_modules(mapping, MODULES_CLEANED)

    # Save
    DEGREE_MODULE_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(DEGREE_MODULE_MAPPING, index=False)
    print(f"\n  Saved to: {DEGREE_MODULE_MAPPING}")


if __name__ == "__main__":
    main()
