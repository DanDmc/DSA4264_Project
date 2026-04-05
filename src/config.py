"""
Shared project configuration.

All data paths are derived from DATA_ROOT, which each teammate sets
in their local .env file (see .env.example in repo root).

Usage in any script:
    from src.config import JOBS_RAW_DIR, JOBS_PARSED, JOBS_SSOC_MAPPED, ...
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load .env from repo root
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DATA_ROOT = os.getenv("DATA_ROOT")
if not DATA_ROOT:
    print("ERROR: DATA_ROOT is not set.")
    print("Copy .env.example to .env and fill in your OneDrive path.")
    sys.exit(1)

DATA_ROOT = Path(DATA_ROOT)

# ──────────────────────────────────────────────
# Raw data paths
# ──────────────────────────────────────────────
RAW_DIR = DATA_ROOT / "raw"

JOBS_RAW_DIR = RAW_DIR / "jobs"                         # contains batch subfolders e.g. 20250125_20250131/
SSOC_DEFINITIONS = RAW_DIR / "ssoc2024-detailed-definitions.xlsx"

# ──────────────────────────────────────────────
# Processed data paths — jobs pipeline
# ──────────────────────────────────────────────
JOBS_PROCESSED_DIR = DATA_ROOT / "processed" / "jobs"

JOBS_PARSED = JOBS_PROCESSED_DIR / "01_jobs_parsed.csv"
JOBS_SSOC_MAPPED = JOBS_PROCESSED_DIR / "02_jobs_ssoc_mapped.csv"
JOBS_FILTERED = JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"

# ──────────────────────────────────────────────
# Pipeline parameters — jobs
# ──────────────────────────────────────────────
MAX_YEARS_EXPERIENCE = 2  # default filter for fresh grad roles

# ──────────────────────────────────────────────
# Pipeline parameters — courses
# ──────────────────────────────────────────────

# Module levels to exclude (NUS module codes encode level in first digit
# of numeric part, e.g. CS5228 → level 5). Levels 5-6 are graduate courses.
EXCLUDED_MODULE_LEVELS = {5, 6}

# Modules whose normalised description matches any of these exactly are
# treated as placeholders and excluded from the cleaned output.
PLACEHOLDER_DESCRIPTIONS = {
    "not available",
    "not available.",
    "not applicable",
    "unrestricted elective",
    "nil",
    "department exchange course",
    "advance placement credit",
}

# Modules whose description contains any of these keywords (case-insensitive)
# are excluded — e.g. internship modules don't have extractable skills.
EXCLUDED_DESCRIPTION_KEYWORDS = ["internship"]

# ──────────────────────────────────────────────
# Raw data paths — courses
# ──────────────────────────────────────────────
COURSES_RAW_DIR = RAW_DIR / "courses"                   # NUSMods API output lands here

# NUSMods API
NUSMODS_API_URL = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"

# ──────────────────────────────────────────────
# Processed data paths — courses pipeline
# ──────────────────────────────────────────────
COURSES_PROCESSED_DIR = DATA_ROOT / "processed" / "courses"

MODULES_RAW = COURSES_RAW_DIR / "modules_raw.csv"
MODULES_CLEANED = COURSES_PROCESSED_DIR / "modules_cleaned.csv"
