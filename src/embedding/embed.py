"""
embed.py
========
Embedding pipeline for NUS modules and job postings.

Supports two modes:
  - whole_text:  one embedding per document (baseline)
  - sentence:    one embedding per sentence, with mapping back to parent doc
                 (for job-side averaged best-match similarity)

Usage (from repo root):
    python -m src.embedding.embed                     # both modes
    python -m src.embedding.embed --mode whole_text    # baseline only
    python -m src.embedding.embed --mode sentence      # sentence-level only

Outputs saved to EMBEDDINGS_DIR (OneDrive).
Works on CPU (slower) or GPU (fast). Automatically detects CUDA.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Import from shared project config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
    JOB_INDEX_COLS,
    JOB_PREFIX,
    JOB_TEXT_FIELDS,
    JOBS_FILTERED,
    MODULE_INDEX_COLS,
    MODULE_PREFIX,
    MODULE_TEXT_FIELDS,
    MODULES_CLEANED,
)

# Regex for sentence splitting — splits on period/question/exclamation
# followed by whitespace and an uppercase letter or digit
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# ──────────────────────────────────────────────
# Text preparation
# ──────────────────────────────────────────────

def _join_fields(row, fields):
    """Concatenate specified fields into a single string, handling NaN."""
    parts = []
    for field in fields:
        val = str(row.get(field, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return ". ".join(parts) if parts else ""


def prepare_whole_texts(df, fields, prefix):
    """Prepare one text per document with BGE instruction prefix."""
    return [prefix + _join_fields(row, fields) for _, row in df.iterrows()]


def split_into_sentences(text):
    """Split text into sentences. Returns list of non-empty sentences."""
    if not text or not text.strip():
        return []
    # normalize whitespace and newlines
    text = text.replace("\n", ". ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text)]
    # filter out very short fragments (< 10 chars) that aren't real sentences
    return [s for s in parts if len(s) >= 10]


def prepare_sentence_data(df, fields, prefix, id_col):
    """Prepare sentence-level data with mapping back to parent document.

    Returns:
        texts: list of prefixed sentence strings ready for embedding
        sentence_index: DataFrame with columns [doc_id, sentence_idx, sentence_text]
    """
    texts = []
    index_rows = []

    for _, row in df.iterrows():
        doc_id = row[id_col]
        full_text = _join_fields(row, fields)
        sentences = split_into_sentences(full_text)

        if not sentences:
            # fallback: use whole text as a single "sentence"
            sentences = [full_text] if full_text.strip() else []

        for sent_idx, sentence in enumerate(sentences):
            texts.append(prefix + sentence)
            index_rows.append({
                "doc_id": doc_id,
                "sentence_idx": sent_idx,
                "sentence_text": sentence,
            })

    sentence_index = pd.DataFrame(index_rows)
    return texts, sentence_index


# ──────────────────────────────────────────────
# Model loading and embedding
# ──────────────────────────────────────────────

def load_model():
    """Load the sentence transformer model onto the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {EMBEDDING_MODEL}")
    print(f"Device: {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (this will be slower than GPU)")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model


def embed_texts(model, texts, label=""):
    """Embed a list of texts. Returns L2-normalized numpy array."""
    print(f"\nEmbedding {len(texts):,} {label} texts (batch_size={EMBEDDING_BATCH_SIZE})...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time:  {elapsed:.1f}s ({rate:.0f} texts/sec)")

    return embeddings


# ──────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────

def save_whole_text(module_emb, job_emb, modules_df, jobs_df):
    """Save whole-text embeddings and index files."""
    out_dir = EMBEDDINGS_DIR / "whole_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = EMBEDDING_MODEL.split("/")[-1]

    np.save(out_dir / f"module_embeddings_{model_tag}.npy", module_emb)
    np.save(out_dir / f"job_embeddings_{model_tag}.npy", job_emb)

    module_index = modules_df[MODULE_INDEX_COLS].copy()
    module_index.index.name = "embed_idx"
    module_index.to_csv(out_dir / "module_index.csv")

    job_index = jobs_df[JOB_INDEX_COLS].copy()
    job_index.index.name = "embed_idx"
    job_index.to_csv(out_dir / "job_index.csv")

    _save_config_record(out_dir, "whole_text", len(modules_df), len(jobs_df))
    _print_dir_summary(out_dir)


def save_sentence_level(module_emb, job_emb, module_sent_idx, job_sent_idx):
    """Save sentence-level embeddings and index files."""
    out_dir = EMBEDDINGS_DIR / "sentence_level"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = EMBEDDING_MODEL.split("/")[-1]

    np.save(out_dir / f"module_sentence_embeddings_{model_tag}.npy", module_emb)
    np.save(out_dir / f"job_sentence_embeddings_{model_tag}.npy", job_emb)

    module_sent_idx.to_csv(out_dir / "module_sentence_index.csv", index=False)
    job_sent_idx.to_csv(out_dir / "job_sentence_index.csv", index=False)

    n_modules = module_sent_idx["doc_id"].nunique()
    n_jobs = job_sent_idx["doc_id"].nunique()
    _save_config_record(out_dir, "sentence_level", n_modules, n_jobs,
                        extra={
                            "module_sentences": len(module_sent_idx),
                            "job_sentences": len(job_sent_idx),
                            "avg_sentences_per_module": round(len(module_sent_idx) / max(n_modules, 1), 1),
                            "avg_sentences_per_job": round(len(job_sent_idx) / max(n_jobs, 1), 1),
                        })
    _print_dir_summary(out_dir)


def _save_config_record(out_dir, mode, n_modules, n_jobs, extra=None):
    """Save a JSON record of what produced these embeddings."""
    record = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "mode": mode,
        "normalized": True,
        "module_text_fields": MODULE_TEXT_FIELDS,
        "module_prefix": MODULE_PREFIX,
        "job_text_fields": JOB_TEXT_FIELDS,
        "job_prefix": JOB_PREFIX,
        "n_modules": n_modules,
        "n_jobs": n_jobs,
        "created_at": datetime.now().isoformat(),
    }
    if extra:
        record.update(extra)
    with open(out_dir / "embedding_config.json", "w") as f:
        json.dump(record, f, indent=2)


def _print_dir_summary(out_dir):
    """Print saved files and sizes."""
    print(f"\nSaved to: {out_dir}")
    for fpath in sorted(out_dir.iterdir()):
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fpath.name:50s} {size_mb:>8.2f} MB")


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

def sanity_check(module_emb, job_emb, modules_df, jobs_df, n_samples=3, top_k=5):
    """Spot-check top-k job matches for a few modules (whole-text only)."""
    print(f"\n{'=' * 70}")
    print(f"SANITY CHECK: Top-{top_k} job matches for {n_samples} sample modules")
    print(f"{'=' * 70}")

    indices = np.linspace(0, len(modules_df) - 1, n_samples, dtype=int)

    for idx in indices:
        sims = module_emb[idx] @ job_emb.T
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        mod = modules_df.iloc[idx]
        print(f"\nMODULE: [{mod['module code']}] {mod['title']}")
        print(f"  Faculty: {mod['faculty']}")

        for rank, j_idx in enumerate(top_k_idx, 1):
            job = jobs_df.iloc[j_idx]
            print(f"  #{rank}  (sim={sims[j_idx]:.4f})  {job['title']}")
            print(f"       SSOC: {job.get('ssoc_minor_title', 'N/A')}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed NUS modules and job postings.")
    parser.add_argument(
        "--mode",
        choices=["whole_text", "sentence", "both"],
        default="both",
        help="Embedding mode (default: both)",
    )
    args = parser.parse_args()

    # load data
    print(f"Loading modules from: {MODULES_CLEANED}")
    modules_df = pd.read_csv(MODULES_CLEANED)
    print(f"  → {len(modules_df):,} modules")

    print(f"Loading jobs from: {JOBS_FILTERED}")
    jobs_df = pd.read_csv(JOBS_FILTERED)
    print(f"  → {len(jobs_df):,} jobs")

    # load model once
    model = load_model()

    # whole-text mode
    if args.mode in ("whole_text", "both"):
        print(f"\n{'=' * 70}")
        print("MODE: WHOLE TEXT")
        print(f"{'=' * 70}")

        module_texts = prepare_whole_texts(modules_df, MODULE_TEXT_FIELDS, MODULE_PREFIX)
        job_texts = prepare_whole_texts(jobs_df, JOB_TEXT_FIELDS, JOB_PREFIX)

        module_emb = embed_texts(model, module_texts, label="module")
        job_emb = embed_texts(model, job_texts, label="job")

        save_whole_text(module_emb, job_emb, modules_df, jobs_df)
        sanity_check(module_emb, job_emb, modules_df, jobs_df)

    # sentence-level mode
    if args.mode in ("sentence", "both"):
        print(f"\n{'=' * 70}")
        print("MODE: SENTENCE LEVEL")
        print(f"{'=' * 70}")

        module_sent_texts, module_sent_idx = prepare_sentence_data(
            modules_df, MODULE_TEXT_FIELDS, MODULE_PREFIX, id_col="module code"
        )
        job_sent_texts, job_sent_idx = prepare_sentence_data(
            jobs_df, JOB_TEXT_FIELDS, JOB_PREFIX, id_col="job_id"
        )

        print(f"\nModules: {modules_df['module code'].nunique()} docs → {len(module_sent_texts):,} sentences")
        print(f"Jobs:    {jobs_df['job_id'].nunique()} docs → {len(job_sent_texts):,} sentences")

        module_sent_emb = embed_texts(model, module_sent_texts, label="module sentence")
        job_sent_emb = embed_texts(model, job_sent_texts, label="job sentence")

        save_sentence_level(module_sent_emb, job_sent_emb, module_sent_idx, job_sent_idx)

    print(f"\n{'=' * 70}")
    print("EMBEDDING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
