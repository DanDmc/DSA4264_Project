"""
Configuration for the course–job embedding pipeline.

All constants (model, paths, text fields, prefixes) live here so that
changing the model or input format means editing one file, not hunting
through code.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths (relative to repo root)
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # src/embedding/config.py → repo root

DATA_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = REPO_ROOT / "data" / "embeddings"

MODULES_FILE = DATA_DIR / "cleaned_modules.csv"
JOBS_FILE = DATA_DIR / "final_jobs_processed_filtered.csv"

# ──────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
BATCH_SIZE = 64  # reduce to 32 or 16 if you hit GPU OOM

# ──────────────────────────────────────────────
# Text preparation
# ──────────────────────────────────────────────

# Columns to concatenate for each dataset
MODULE_TEXT_FIELDS = ["title", "description_clean"]
JOB_TEXT_FIELDS = ["title", "clean_description", "skills_list"]

# BGE instruction prefixes (asymmetric retrieval)
MODULE_PREFIX = "Represent this university course for matching to relevant job positions: "
JOB_PREFIX = "Represent this job posting for matching to relevant university courses: "

# ──────────────────────────────────────────────
# Index metadata columns (saved alongside embeddings)
# ──────────────────────────────────────────────
MODULE_INDEX_COLS = [
    "module code", "title", "department", "faculty",
    "module_credit", "Undergraduate/Graduate",
]

JOB_INDEX_COLS = [
    "job_id", "title", "category", "salary_avg",
    "ssoc_code", "ssoc_major_title", "ssoc_submajor_title",
    "ssoc_minor_title", "ssoc_unit_title",
]
