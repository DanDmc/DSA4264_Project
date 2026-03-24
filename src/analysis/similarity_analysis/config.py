"""
Configuration for the similarity analysis pipeline.

All tuneable parameters live here — top-k, thresholds, SSOC grouping level,
and paths. Change these without touching analysis logic.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths (relative to repo root)
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/analysis/similarity_analysis/config.py → repo root

EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
OUTPUT_DIR = REPO_ROOT / "outputs" / "similarity_analysis_outputs"

# Embedding files (must match what embed.py produced)
MODEL_TAG = "bge-large-en-v1.5"
MODULE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / f"module_embeddings_{MODEL_TAG}.npy"
JOB_EMBEDDINGS_FILE = EMBEDDINGS_DIR / f"job_embeddings_{MODEL_TAG}.npy"
MODULE_INDEX_FILE = EMBEDDINGS_DIR / "module_index.csv"
JOB_INDEX_FILE = EMBEDDINGS_DIR / "job_index.csv"

# ──────────────────────────────────────────────
# Top-k matching
# ──────────────────────────────────────────────
TOP_K = 20  # number of matches to store per module/job

# ──────────────────────────────────────────────
# SSOC aggregation level
# ──────────────────────────────────────────────
# Options: "ssoc_major_title", "ssoc_submajor_title", "ssoc_minor_title", "ssoc_unit_title"
SSOC_GROUP_COL = "ssoc_minor_title"

# ──────────────────────────────────────────────
# Coverage score
# ──────────────────────────────────────────────
# A faculty "covers" an SSOC group if at least one of its modules
# has similarity >= this threshold to at least one job in that group.
COVERAGE_THRESHOLD = 0.5
