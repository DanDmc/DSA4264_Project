"""
Process degree mapping: explode raw Excel into a flat lookup table.

Reads the degree mapping Excel file (one row per major, comma-separated
module codes) and produces a normalised CSV with one row per
(university, faculty, major, module_code, module_type) combination.

Enriches each row with the module's home department and faculty from
modules_cleaned.csv, so downstream analysis can slice by either the
degree's faculty or the module's home department without extra joins.

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
                        "degree_faculty": fac,
                        "major": major,
                        "module_code": code,
                        "module_type": module_type,
                    })

    result = pd.DataFrame(records)
    # Deduplicate (same module listed twice for same major)
    result = result.drop_duplicates()
    return result


def enrich_with_module_metadata(mapping_df, modules_path):
    """Left-join module home department and faculty from modules_cleaned.csv."""
    try:
        modules = pd.read_csv(modules_path)
    except FileNotFoundError:
        print(f"  WARNING: {modules_path} not found — skipping enrichment.")
        return mapping_df

    # Keep only the columns we need for the join
    module_meta = (
        modules[["module code", "department", "faculty"]]
        .rename(columns={
            "module code": "module_code",
            "department": "module_department",
            "faculty": "module_faculty",
        })
        .drop_duplicates(subset=["module_code"])
    )

    enriched = mapping_df.merge(module_meta, on="module_code", how="left")

    matched = enriched["module_department"].notna().sum()
    unmatched = enriched["module_department"].isna().sum()
    print(f"\n  Enrichment from modules_cleaned.csv:")
    print(f"    Rows with module metadata:    {matched}")
    print(f"    Rows without (not in catalogue): {unmatched}")

    if unmatched > 0:
        missing_codes = (
            enriched.loc[enriched["module_department"].isna(), "module_code"]
            .unique()
        )
        sample = sorted(missing_codes)[:20]
        print(f"    Sample unmatched codes: {sample}")
        if len(missing_codes) > 20:
            print(f"    ... and {len(missing_codes) - 20} more")

    return enriched


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

    # Enrich with module home department/faculty
    mapping = enrich_with_module_metadata(mapping, MODULES_CLEANED)

    # Summary by major
    summary = (
        mapping
        .groupby(["university", "degree_faculty", "major", "module_type"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"\n  Modules per major (core / elective):")
    for (uni, fac, major), counts in summary.iterrows():
        core_n = counts.get("core", 0)
        elec_n = counts.get("elective", 0)
        print(f"    {major:<45s}  core={core_n:<4d}  elective={elec_n}")

    # Save
    DEGREE_MODULE_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(DEGREE_MODULE_MAPPING, index=False)
    print(f"\n  Saved to: {DEGREE_MODULE_MAPPING}")
    print(f"  Columns: {list(mapping.columns)}")


if __name__ == "__main__":
    main()
