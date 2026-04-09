"""
Shared project configuration.

All data paths are derived from DATA_ROOT, which each teammate sets
in their local .env file (see .env.example in repo root).

Usage in any script:
    from src.config import JOBS_RAW_DIR, JOBS_EXTRACTED, JOBS_CLEANED, ...
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

# Step 1a: extract raw JSONs → CSV (no cleaning)
JOBS_EXTRACTED = JOBS_PROCESSED_DIR / "01a_jobs_extracted.csv"

# Step 1b: apply text cleaning
JOBS_CLEANED = JOBS_PROCESSED_DIR / "01b_jobs_cleaned.csv"

# Step 2: SSOC mapping
JOBS_SSOC_MAPPED = JOBS_PROCESSED_DIR / "02_jobs_ssoc_mapped.csv"

# Step 3: filter to entry-level
JOBS_FILTERED = JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"

# Legacy alias (points to cleaned output for backward compatibility)
JOBS_PARSED = JOBS_CLEANED

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
MODULE_SKILL_PAIRS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"

# ──────────────────────────────────────────────
# Embedding paths
# ──────────────────────────────────────────────
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"

# ──────────────────────────────────────────────
# Embedding parameters
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 64  # reduce to 32 or 16 if you hit GPU OOM

# BGE instruction prefixes (asymmetric retrieval — course is the query, job is the document)
MODULE_PREFIX = "Represent this university course for matching to relevant job positions: "
JOB_PREFIX = "Represent this job posting for matching to relevant university courses: "

# Text fields to concatenate for embedding (order matters)
MODULE_TEXT_FIELDS = ["title", "description_clean"]
JOB_TEXT_FIELDS = ["title", "clean_description"]

# Metadata columns saved alongside embeddings (maps array row → identity)
MODULE_INDEX_COLS = [
    "module code", "title", "department", "faculty", "module_credit",
]
JOB_INDEX_COLS = [
    "job_id", "title", "category", "salary_avg",
    "ssoc_code", "ssoc_major_title", "ssoc_submajor_title",
    "ssoc_minor_title", "ssoc_unit_title",
]
