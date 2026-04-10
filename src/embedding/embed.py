"""
embed.py — Embedding pipeline for NUS modules and job postings
===============================================================

Embeds module descriptions and job postings using BGE-large, producing
numpy arrays + index CSVs that downstream analysis scripts consume.

Default mode is whole_text (one embedding per document), validated as
the best-performing approach via MRR and category separation tests.

Usage (from repo root):
    python -m src.embedding.embed                       # whole_text (default)
    python -m src.embedding.embed --mode whole_text      # explicit
    python -m src.embedding.embed --mode sentence        # sentence-level (experimental)
    python -m src.embedding.embed --mode both            # both modes

On Colab, pass --data-root to bypass .env:
    python -m src.embedding.embed --data-root /content/drive/MyDrive/DSA4264_Project_Data

Outputs:
    EMBEDDINGS_DIR/whole_text/
        module_embeddings_{model_tag}.npy   — (n_modules, 1024) float32
        job_embeddings_{model_tag}.npy      — (n_jobs, 1024) float32
        module_index.csv                    — maps embedding row → module metadata
        job_index.csv                       — maps embedding row → job metadata
        embedding_config.json               — reproducibility record

Works on CPU (slower) or GPU (fast). Automatically detects CUDA.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

def _load_config(data_root_override=None):
    """
    Import project config. If data_root_override is provided (e.g.
    --data-root on Colab), set DATA_ROOT env var before importing
    so config.py picks up the right paths without needing a .env file.
    """
    if data_root_override:
        os.environ["DATA_ROOT"] = str(data_root_override)

    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

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

    return {
        "EMBEDDING_BATCH_SIZE": EMBEDDING_BATCH_SIZE,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "EMBEDDINGS_DIR": EMBEDDINGS_DIR,
        "JOB_INDEX_COLS": JOB_INDEX_COLS,
        "JOB_PREFIX": JOB_PREFIX,
        "JOB_TEXT_FIELDS": JOB_TEXT_FIELDS,
        "JOBS_FILTERED": JOBS_FILTERED,
        "MODULE_INDEX_COLS": MODULE_INDEX_COLS,
        "MODULE_PREFIX": MODULE_PREFIX,
        "MODULE_TEXT_FIELDS": MODULE_TEXT_FIELDS,
        "MODULES_CLEANED": MODULES_CLEANED,
    }


# ──────────────────────────────────────────────
# Text preparation
# ──────────────────────────────────────────────

# sentence boundary regex — handles abbreviations better than naive period split
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def join_fields(row, fields):
    """
    Concatenate specified columns into a single string, skipping NaN.

    Public function — also used by the inference module to prepare
    a single document's text for embedding at query time.
    """
    parts = []
    for field in fields:
        val = str(row.get(field, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return ". ".join(parts) if parts else ""


def prepare_whole_texts(df, fields, prefix):
    """One text per document with BGE instruction prefix prepended."""
    return [prefix + join_fields(row, fields) for _, row in df.iterrows()]


def split_into_sentences(text):
    """
    Split text into sentence-like chunks. Handles job posting
    formatting (bullets, headers, abbreviations like 'e.g.') by
    splitting on newlines first, then on sentence-ending punctuation
    followed by uppercase.
    """
    if not text or not text.strip():
        return []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sentences = []
    for line in lines:
        if len(line) < 120:
            sentences.append(line)
        else:
            parts = SENTENCE_SPLIT_RE.split(line)
            sentences.extend(parts)
    return [s for s in sentences if len(s) >= 10]


def prepare_sentence_data(df, fields, prefix, id_col):
    """
    Prepare sentence-level data with mapping back to parent document.

    NOTE: Sentence-level embedding is experimental and not used in
    the main pipeline. Whole-text mode was validated as the better
    approach. This function is retained for future experimentation.
    """
    texts = []
    index_rows = []

    for _, row in df.iterrows():
        doc_id = row[id_col]
        full_text = join_fields(row, fields)
        sentences = split_into_sentences(full_text)

        if not sentences:
            sentences = [full_text] if full_text.strip() else []

        for sent_idx, sentence in enumerate(sentences):
            texts.append(prefix + sentence)
            index_rows.append({
                "doc_id": doc_id,
                "sentence_idx": sent_idx,
                "sentence_text": sentence,
            })

    return texts, pd.DataFrame(index_rows)


# ──────────────────────────────────────────────
# Model loading and embedding
# ──────────────────────────────────────────────

def load_model(model_name):
    """Load sentence transformer onto best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Device: {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (will be slower than GPU)")

    return SentenceTransformer(model_name, device=device)


def embed_texts(model, texts, batch_size, label=""):
    """Embed a list of texts. Returns L2-normalized numpy array."""
    print(f"\nEmbedding {len(texts):,} {label} texts (batch_size={batch_size})...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time:  {elapsed:.1f}s ({rate:.0f} texts/sec)")

    return embeddings


def embed_single_text(model, text, prefix):
    """
    Embed a single text string. Returns L2-normalized 1D numpy array.

    Designed for inference/query time — embed one job posting to
    compare against precomputed module embeddings. Kept here so all
    embedding logic (model, prefix, normalization) stays in one place.
    """
    return model.encode(
        prefix + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ──────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────

def _model_tag(model_name):
    return model_name.split("/")[-1]


def save_whole_text(module_emb, job_emb, modules_df, jobs_df, cfg):
    """Save whole-text embeddings + index CSVs + config record."""
    out_dir = cfg["EMBEDDINGS_DIR"] / "whole_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(cfg["EMBEDDING_MODEL"])

    np.save(out_dir / f"module_embeddings_{tag}.npy", module_emb)
    np.save(out_dir / f"job_embeddings_{tag}.npy", job_emb)

    module_index = modules_df[cfg["MODULE_INDEX_COLS"]].copy()
    module_index.index.name = "embed_idx"
    module_index.to_csv(out_dir / "module_index.csv")

    job_index = jobs_df[cfg["JOB_INDEX_COLS"]].copy()
    job_index.index.name = "embed_idx"
    job_index.to_csv(out_dir / "job_index.csv")

    _save_config_record(out_dir, "whole_text", len(modules_df), len(jobs_df), cfg)
    _print_dir_summary(out_dir)


def save_sentence_level(module_emb, job_emb, module_sent_idx, job_sent_idx, cfg):
    """
    Save sentence-level embeddings + sentence index CSVs.

    NOTE: Not used in the main pipeline — whole-text mode is the
    validated approach. Retained for future experimentation.
    """
    out_dir = cfg["EMBEDDINGS_DIR"] / "sentence_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(cfg["EMBEDDING_MODEL"])

    np.save(out_dir / f"module_sentence_embeddings_{tag}.npy", module_emb)
    np.save(out_dir / f"job_sentence_embeddings_{tag}.npy", job_emb)

    module_sent_idx.to_csv(out_dir / "module_sentence_index.csv", index=False)
    job_sent_idx.to_csv(out_dir / "job_sentence_index.csv", index=False)

    n_modules = module_sent_idx["doc_id"].nunique()
    n_jobs = job_sent_idx["doc_id"].nunique()
    _save_config_record(out_dir, "sentence_level", n_modules, n_jobs, cfg,
                        extra={
                            "module_sentences": len(module_sent_idx),
                            "job_sentences": len(job_sent_idx),
                            "avg_sentences_per_module": round(len(module_sent_idx) / max(n_modules, 1), 1),
                            "avg_sentences_per_job": round(len(job_sent_idx) / max(n_jobs, 1), 1),
                        })
    _print_dir_summary(out_dir)


def _save_config_record(out_dir, mode, n_modules, n_jobs, cfg, extra=None):
    """Write a JSON record of what produced these embeddings — for reproducibility."""
    record = {
        "model": cfg["EMBEDDING_MODEL"],
        "embedding_dim": cfg["EMBEDDING_DIM"],
        "mode": mode,
        "normalized": True,
        "module_text_fields": cfg["MODULE_TEXT_FIELDS"],
        "module_prefix": cfg["MODULE_PREFIX"],
        "job_text_fields": cfg["JOB_TEXT_FIELDS"],
        "job_prefix": cfg["JOB_PREFIX"],
        "n_modules": n_modules,
        "n_jobs": n_jobs,
        "created_at": datetime.now().isoformat(),
    }
    if extra:
        record.update(extra)
    with open(out_dir / "embedding_config.json", "w") as f:
        json.dump(record, f, indent=2)


def _print_dir_summary(out_dir):
    print(f"\nSaved to: {out_dir}")
    for fpath in sorted(out_dir.iterdir()):
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fpath.name:50s} {size_mb:>8.2f} MB")


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

def sanity_check(module_emb, job_emb, modules_df, jobs_df, cfg, n_samples=3, top_k=5):
    """
    Spot-check: for a few modules, print their top-k most similar jobs.
    Quick way to catch obvious issues before committing to full analysis.
    """
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
    parser = argparse.ArgumentParser(
        description="Embed NUS modules and job postings using BGE.",
        epilog="On Colab: pass --data-root /content/drive/MyDrive/DSA4264_Project_Data",
    )
    parser.add_argument(
        "--mode",
        choices=["whole_text", "sentence", "both"],
        default="whole_text",
        help="Embedding mode (default: whole_text — validated as best approach)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override DATA_ROOT (useful on Colab where .env doesn't exist)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override EMBEDDING_BATCH_SIZE (reduce to 32 or 16 if GPU OOM)",
    )
    args = parser.parse_args()

    cfg = _load_config(data_root_override=args.data_root)

    if args.batch_size:
        cfg["EMBEDDING_BATCH_SIZE"] = args.batch_size

    # load data
    print(f"Loading modules from: {cfg['MODULES_CLEANED']}")
    modules_df = pd.read_csv(cfg["MODULES_CLEANED"])
    print(f"  → {len(modules_df):,} modules")

    print(f"Loading jobs from: {cfg['JOBS_FILTERED']}")
    jobs_df = pd.read_csv(cfg["JOBS_FILTERED"])
    print(f"  → {len(jobs_df):,} jobs")

    model = load_model(cfg["EMBEDDING_MODEL"])

    # ── whole-text mode (default, validated as best approach) ──
    if args.mode in ("whole_text", "both"):
        print(f"\n{'=' * 70}")
        print("MODE: WHOLE TEXT (validated baseline)")
        print(f"{'=' * 70}")

        module_texts = prepare_whole_texts(modules_df, cfg["MODULE_TEXT_FIELDS"], cfg["MODULE_PREFIX"])
        job_texts = prepare_whole_texts(jobs_df, cfg["JOB_TEXT_FIELDS"], cfg["JOB_PREFIX"])

        module_emb = embed_texts(model, module_texts, cfg["EMBEDDING_BATCH_SIZE"], label="module")
        job_emb = embed_texts(model, job_texts, cfg["EMBEDDING_BATCH_SIZE"], label="job")

        save_whole_text(module_emb, job_emb, modules_df, jobs_df, cfg)
        sanity_check(module_emb, job_emb, modules_df, jobs_df, cfg)

    # ── sentence-level mode (experimental — not used in main pipeline) ──
    if args.mode in ("sentence", "both"):
        print(f"\n{'=' * 70}")
        print("MODE: SENTENCE LEVEL (experimental — not the validated approach)")
        print(f"{'=' * 70}")

        module_sent_texts, module_sent_idx = prepare_sentence_data(
            modules_df, cfg["MODULE_TEXT_FIELDS"], cfg["MODULE_PREFIX"], id_col="module code"
        )
        job_sent_texts, job_sent_idx = prepare_sentence_data(
            jobs_df, cfg["JOB_TEXT_FIELDS"], cfg["JOB_PREFIX"], id_col="job_id"
        )

        print(f"\nModules: {modules_df['module code'].nunique()} docs → {len(module_sent_texts):,} sentences")
        print(f"Jobs:    {jobs_df['job_id'].nunique()} docs → {len(job_sent_texts):,} sentences")

        module_sent_emb = embed_texts(model, module_sent_texts, cfg["EMBEDDING_BATCH_SIZE"], label="module sentence")
        job_sent_emb = embed_texts(model, job_sent_texts, cfg["EMBEDDING_BATCH_SIZE"], label="job sentence")

        save_sentence_level(module_sent_emb, job_sent_emb, module_sent_idx, job_sent_idx, cfg)

    print(f"\n{'=' * 70}")
    print("EMBEDDING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
