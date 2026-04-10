"""
Shared utilities for embedding validation experiments.
------------------------------------------------------
Contains: data loading, sentence splitting, metrics computation,
and result I/O. All four validation variants import from here
to keep logic DRY and comparable.
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(filepath):
    """Load validation dataset from Excel. Returns (jobs_dict, courses_dict)."""
    df_jobs = pd.read_excel(filepath, sheet_name="jobs")
    df_courses = pd.read_excel(filepath, sheet_name="courses")

    jobs = {
        row["id"]: {"text": row["text"], "category": row["category"]}
        for _, row in df_jobs.iterrows()
    }

    # courses sheet has a title column we preserve for v4-style experiments
    courses = {
        row["module_code"]: {
            "text": row["text"],
            "title": row.get("title", ""),
            "category": row["category"],
        }
        for _, row in df_courses.iterrows()
    }

    return jobs, courses


# ─────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────

def split_sentences(text, min_length=10):
    """
    Split text into sentence-like chunks.

    Job postings and course descriptions aren't clean prose — they have
    bullet points, headers, abbreviations like 'e.g.'. A simple period
    split butchers these. Instead we:
      1. Split on newlines (captures bullet/header structure)
      2. For longer lines, split on sentence-ending punctuation
         followed by whitespace + uppercase (avoids splitting 'e.g.')
      3. Drop very short fragments (headers, stray letters)

    Not perfect, but good enough for embedding — the model is robust
    to minor boundary errors.
    """
    if not text or not isinstance(text, str):
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    sentences = []
    for line in lines:
        if len(line) < 120:
            sentences.append(line)
        else:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
            sentences.extend(parts)

    return [s for s in sentences if len(s) >= min_length]


# ─────────────────────────────────────────────
# Cosine similarity helper
# ─────────────────────────────────────────────

def cosine(a, b):
    """Cosine similarity between two vectors."""
    return float(cosine_similarity([a], [b])[0][0])


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_retrieval_metrics(results, score_key):
    """
    Retrieval-style metrics treating each job as a query and courses as
    candidates. A "hit" is when the top-ranked course shares the job's
    category.

    Returns MRR, Top-1 accuracy, and Top-2 accuracy.
    - MRR (Mean Reciprocal Rank): average of 1/rank for the first
      same-category course. Higher = correct courses rank near the top.
    - Top-1/Top-2: fraction of jobs where a same-category course appears
      in the top 1 or 2 results.
    """
    job_map = {}
    for r in results:
        job_map.setdefault(r["job"], []).append(r)

    mrr_values = []
    top1_hits = 0
    top2_hits = 0

    for job_id, items in job_map.items():
        sorted_items = sorted(items, key=lambda x: x[score_key], reverse=True)

        # find rank of first same-category course
        rank = None
        for i, r in enumerate(sorted_items):
            if r["job_category"] == r["course_category"]:
                rank = i + 1
                break

        if rank:
            mrr_values.append(1 / rank)
            if rank == 1:
                top1_hits += 1
            if rank <= 2:
                top2_hits += 1

    n = len(job_map)
    return {
        "MRR": round(float(np.mean(mrr_values)) if mrr_values else 0.0, 4),
        "Top1": round(top1_hits / n, 4),
        "Top2": round(top2_hits / n, 4),
    }


def compute_category_separation(results, score_key):
    """
    How well does the embedding separate same-category pairs from
    cross-category pairs?

    - intra_mean: average similarity for same-category (job, course) pairs
    - inter_mean: average similarity for different-category pairs
    - separation_gap: difference (intra - inter). Bigger = better discrimination.
    - separation_ratio: intra / inter. >1 means same-category pairs score
      higher on average.
    - misalignment_rate: fraction of inter-category pairs that score above
      the intra-category median. This is the "false positive" rate — how
      often a wrong course would look like a good match. Directly relevant
      to MOE: if this is high, the metric can't reliably flag gaps.
    """
    intra_scores = []
    inter_scores = []

    for r in results:
        if r["job_category"] == r["course_category"]:
            intra_scores.append(r[score_key])
        else:
            inter_scores.append(r[score_key])

    intra_mean = float(np.mean(intra_scores))
    inter_mean = float(np.mean(inter_scores))
    intra_median = float(np.median(intra_scores))

    # what fraction of cross-category pairs score above the intra median?
    # i.e. how often would a misaligned course look "aligned"
    misaligned = sum(1 for s in inter_scores if s > intra_median)
    misalignment_rate = misaligned / len(inter_scores) if inter_scores else 0.0

    return {
        "intra_mean": round(intra_mean, 4),
        "inter_mean": round(inter_mean, 4),
        "separation_gap": round(intra_mean - inter_mean, 4),
        "separation_ratio": round(intra_mean / inter_mean, 4) if inter_mean else None,
        "misalignment_rate": round(misalignment_rate, 4),
    }


def compute_per_category_metrics(results, score_key):
    """
    Per-category MRR and separation stats. Useful for spotting categories
    where the embedding struggles — e.g. if 'digital_media' courses are
    too generic to distinguish from 'software_engineer'.
    """
    categories = sorted(set(r["job_category"] for r in results))
    per_cat = {}

    for cat in categories:
        cat_results = [r for r in results if r["job_category"] == cat]
        per_cat[cat] = {
            "retrieval": compute_retrieval_metrics(cat_results, score_key),
            "separation": compute_category_separation(cat_results, score_key),
        }

    return per_cat


# ─────────────────────────────────────────────
# Result I/O
# ─────────────────────────────────────────────

def save_results(output, filepath):
    """Save results dict to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


def build_variant_summary(results, score_key, variant_name, config):
    """
    Build the standard output dict for a single variant.
    Includes config, aggregate metrics, per-category breakdown,
    and raw pair-level results.
    """
    return {
        "variant": variant_name,
        "config": config,
        "aggregate": {
            "retrieval": compute_retrieval_metrics(results, score_key),
            "separation": compute_category_separation(results, score_key),
        },
        "per_category": compute_per_category_metrics(results, score_key),
        "raw_results": results,
    }
