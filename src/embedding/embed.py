"""
Embedding pipeline for NUS modules and job postings.

Usage (from repo root):
    python src/embedding/embed.py

Works on CPU (slower) or GPU (fast). Automatically detects CUDA.
Reads from data/processed/, writes to data/embeddings/.
"""

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Import config from sibling module
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (
    BATCH_SIZE,
    DATA_DIR,
    EMBEDDING_DIM,
    JOB_INDEX_COLS,
    JOB_PREFIX,
    JOB_TEXT_FIELDS,
    JOBS_FILE,
    MODEL_NAME,
    MODULE_INDEX_COLS,
    MODULE_PREFIX,
    MODULE_TEXT_FIELDS,
    MODULES_FILE,
    OUTPUT_DIR,
)


# ──────────────────────────────────────────────
# Text preparation
# ──────────────────────────────────────────────

def prepare_text(row, fields, prefix):
    """
    Concatenate specified fields into a single string with a BGE prefix.

    Handles NaN/missing values gracefully. Fields are joined with ". "
    so the model sees natural sentence boundaries.
    """
    parts = []
    for field in fields:
        val = str(row.get(field, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)

    text = ". ".join(parts) if parts else ""
    return prefix + text


def prepare_module_texts(df):
    """Prepare all module texts for embedding."""
    return df.apply(lambda row: prepare_text(row, MODULE_TEXT_FIELDS, MODULE_PREFIX), axis=1).tolist()


def prepare_job_texts(df):
    """Prepare all job texts for embedding. Skills are prefixed with 'Skills: ' for clarity."""
    def _prepare(row):
        parts = []
        for field in JOB_TEXT_FIELDS:
            val = str(row.get(field, "")).strip()
            if val and val.lower() != "nan":
                if field == "skills_list":
                    parts.append(f"Skills: {val}")
                else:
                    parts.append(val)
        text = ". ".join(parts) if parts else ""
        return JOB_PREFIX + text

    return df.apply(_prepare, axis=1).tolist()


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

def load_model():
    """Load the sentence transformer model onto the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (this will be slower than GPU)")

    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def embed_texts(model, texts, label=""):
    """
    Embed a list of texts. Returns L2-normalized numpy array.

    L2 normalization means cosine similarity = dot product downstream.
    """
    print(f"\nEmbedding {len(texts):,} {label} texts (batch_size={BATCH_SIZE})...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    rate = len(texts) / elapsed

    print(f"  Shape: {embeddings.shape}")
    print(f"  Time:  {elapsed:.1f}s ({rate:.0f} texts/sec)")
    print(f"  L2 norm check: {np.linalg.norm(embeddings[0]):.4f} (should be ~1.0)")

    return embeddings


# ──────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────

def save_outputs(module_embeddings, job_embeddings, modules_df, jobs_df):
    """Save embedding arrays, index CSVs, and config metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = MODEL_NAME.split("/")[-1]

    # Embedding matrices
    mod_path = OUTPUT_DIR / f"module_embeddings_{model_tag}.npy"
    job_path = OUTPUT_DIR / f"job_embeddings_{model_tag}.npy"
    np.save(mod_path, module_embeddings)
    np.save(job_path, job_embeddings)

    # Index CSVs (map array row index → metadata)
    module_index = modules_df[MODULE_INDEX_COLS].copy()
    module_index.index.name = "embed_idx"
    module_index.to_csv(OUTPUT_DIR / "module_index.csv")

    job_index = jobs_df[JOB_INDEX_COLS].copy()
    job_index.index.name = "embed_idx"
    job_index.to_csv(OUTPUT_DIR / "job_index.csv")

    # Config record so future users know what generated these files
    config_record = {
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "normalized": True,
        "module_text_fields": MODULE_TEXT_FIELDS,
        "module_instruction_prefix": MODULE_PREFIX,
        "job_text_fields": JOB_TEXT_FIELDS,
        "job_instruction_prefix": JOB_PREFIX,
        "n_modules": len(modules_df),
        "n_jobs": len(jobs_df),
        "created_at": datetime.now().isoformat(),
        "source_files": {
            "modules": str(MODULES_FILE.relative_to(OUTPUT_DIR.parents[1])),
            "jobs": str(JOBS_FILE.relative_to(OUTPUT_DIR.parents[1])),
        },
    }
    with open(OUTPUT_DIR / "embedding_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # Print summary
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("-" * 60)
    for fpath in sorted(OUTPUT_DIR.iterdir()):
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fpath.name:50s} {size_mb:>8.2f} MB")


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

def sanity_check(module_embeddings, job_embeddings, modules_df, jobs_df, n_samples=3, top_k=5):
    """Spot-check top-k job matches for a few modules."""
    print(f"\n{'='*70}")
    print("SANITY CHECK: Top-{} job matches for {} sample modules".format(top_k, n_samples))
    print(f"{'='*70}")

    # Pick evenly spaced modules
    indices = np.linspace(0, len(modules_df) - 1, n_samples, dtype=int)

    for idx in indices:
        sims = module_embeddings[idx] @ job_embeddings.T
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        mod = modules_df.iloc[idx]
        print(f"\nMODULE: [{mod['module code']}] {mod['title']}")
        print(f"  Faculty: {mod['faculty']}")

        for rank, j_idx in enumerate(top_k_idx, 1):
            job = jobs_df.iloc[j_idx]
            print(f"  #{rank}  (sim={sims[j_idx]:.4f})  {job['title']}")
            print(f"       Category: {job['category']} | SSOC: {job.get('ssoc_minor_title', 'N/A')}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("COURSE–JOB EMBEDDING PIPELINE")
    print("=" * 70)

    # 1. Load data
    print(f"\nLoading modules from: {MODULES_FILE}")
    modules_df = pd.read_csv(MODULES_FILE)
    print(f"  → {len(modules_df):,} modules")

    print(f"Loading jobs from: {JOBS_FILE}")
    jobs_df = pd.read_csv(JOBS_FILE)
    print(f"  → {len(jobs_df):,} jobs")

    # 2. Prepare text
    module_texts = prepare_module_texts(modules_df)
    job_texts = prepare_job_texts(jobs_df)

    print(f"\nSample module text (truncated):")
    print(f"  {module_texts[0][:200]}...")
    print(f"\nSample job text (truncated):")
    print(f"  {job_texts[0][:200]}...")

    # 3. Embed
    model = load_model()
    module_embeddings = embed_texts(model, module_texts, label="module")
    job_embeddings = embed_texts(model, job_texts, label="job")

    # 4. Save
    save_outputs(module_embeddings, job_embeddings, modules_df, jobs_df)

    # 5. Sanity check
    sanity_check(module_embeddings, job_embeddings, modules_df, jobs_df)

    print(f"\n{'='*70}")
    print("DONE. Next steps:")
    print("  1. Review the sanity check results above")
    print("  2. If results look good, commit data/embeddings/ to the repo")
    print("  3. Use Git LFS for .npy files if >50MB: git lfs track '*.npy'")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
