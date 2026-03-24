"""
Similarity analysis pipeline for NUS modules ↔ job postings.

Loads precomputed embeddings, computes cosine similarity, and produces:
  1. Top-k jobs per module
  2. Top-k modules per job
  3. Faculty × SSOC group heatmap data
  4. Faculty coverage scores

Usage (from repo root):
    python src/analysis/similarity_analysis/similarity.py

Outputs saved to data/analysis/.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    COVERAGE_THRESHOLD,
    JOB_EMBEDDINGS_FILE,
    JOB_INDEX_FILE,
    MODULE_EMBEDDINGS_FILE,
    MODULE_INDEX_FILE,
    OUTPUT_DIR,
    SSOC_GROUP_COL,
    TOP_K,
)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_data():
    """Load embeddings and index files."""
    print("Loading embeddings and index files...")

    module_emb = np.load(MODULE_EMBEDDINGS_FILE).astype(np.float32)
    job_emb = np.load(JOB_EMBEDDINGS_FILE).astype(np.float32)
    module_idx = pd.read_csv(MODULE_INDEX_FILE, index_col="embed_idx")
    job_idx = pd.read_csv(JOB_INDEX_FILE, index_col="embed_idx")

    print(f"  Modules: {module_emb.shape[0]:,} embeddings, {len(module_idx):,} index rows")
    print(f"  Jobs:    {job_emb.shape[0]:,} embeddings, {len(job_idx):,} index rows")

    return module_emb, job_emb, module_idx, job_idx


def compute_similarity_matrix(module_emb, job_emb):
    """Compute full cosine similarity matrix (modules × jobs).

    Embeddings are L2-normalized, so dot product = cosine similarity.
    """
    print(f"\nComputing similarity matrix ({module_emb.shape[0]:,} × {job_emb.shape[0]:,})...")
    start = time.time()
    sim_matrix = module_emb @ job_emb.T
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Score range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    print(f"  Mean: {sim_matrix.mean():.4f}, Median: {np.median(sim_matrix):.4f}")
    return sim_matrix


# ──────────────────────────────────────────────
# Output 1: Top-k jobs per module
# ──────────────────────────────────────────────

def top_k_jobs_per_module(sim_matrix, module_idx, job_idx, top_k):
    """For each module, find the top-k most similar jobs."""
    print(f"\nComputing top-{top_k} jobs per module...")

    rows = []
    for i in range(sim_matrix.shape[0]):
        sims = sim_matrix[i]
        top_k_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]

        mod = module_idx.iloc[i]
        for rank, j in enumerate(top_k_indices, 1):
            job = job_idx.iloc[j]
            rows.append({
                "module_code": mod["module code"],
                "module_title": mod["title"],
                "faculty": mod["faculty"],
                "rank": rank,
                "similarity_score": round(float(sims[j]), 4),
                "job_id": job["job_id"],
                "job_title": job["title"],
                "job_category": job["category"],
                SSOC_GROUP_COL: job.get(SSOC_GROUP_COL, "N/A"),
                "salary_avg": job.get("salary_avg", None),
            })

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} rows ({len(module_idx):,} modules × {top_k} matches)")
    return df


# ──────────────────────────────────────────────
# Output 2: Top-k modules per job
# ──────────────────────────────────────────────

def top_k_modules_per_job(sim_matrix, module_idx, job_idx, top_k):
    """For each job, find the top-k most similar modules."""
    print(f"\nComputing top-{top_k} modules per job...")

    # Transpose: jobs × modules
    sim_T = sim_matrix.T

    rows = []
    for j in range(sim_T.shape[0]):
        sims = sim_T[j]
        top_k_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]

        job = job_idx.iloc[j]
        for rank, i in enumerate(top_k_indices, 1):
            mod = module_idx.iloc[i]
            rows.append({
                "job_id": job["job_id"],
                "job_title": job["title"],
                "job_category": job["category"],
                SSOC_GROUP_COL: job.get(SSOC_GROUP_COL, "N/A"),
                "rank": rank,
                "similarity_score": round(float(sims[i]), 4),
                "module_code": mod["module code"],
                "module_title": mod["title"],
                "faculty": mod["faculty"],
            })

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} rows ({len(job_idx):,} jobs × {top_k} matches)")
    return df


# ──────────────────────────────────────────────
# Output 3: Faculty × SSOC heatmap
# ──────────────────────────────────────────────

def faculty_ssoc_heatmap(sim_matrix, module_idx, job_idx):
    """Compute mean and max similarity for each faculty × SSOC group pair."""
    print(f"\nComputing faculty × SSOC heatmap (grouping by: {SSOC_GROUP_COL})...")

    faculties = module_idx["faculty"].values
    ssoc_groups = job_idx[SSOC_GROUP_COL].values

    unique_faculties = sorted(module_idx["faculty"].dropna().unique())
    unique_ssoc = sorted(job_idx[SSOC_GROUP_COL].dropna().unique())

    rows = []
    for fac in unique_faculties:
        fac_mask = faculties == fac
        fac_sims = sim_matrix[fac_mask]  # shape: (n_fac_modules, n_jobs)

        for ssoc in unique_ssoc:
            ssoc_mask = ssoc_groups == ssoc
            block = fac_sims[:, ssoc_mask]  # shape: (n_fac_modules, n_ssoc_jobs)

            if block.size == 0:
                continue

            rows.append({
                "faculty": fac,
                SSOC_GROUP_COL: ssoc,
                "mean_similarity": round(float(block.mean()), 4),
                "max_similarity": round(float(block.max()), 4),
                "n_modules": int(fac_mask.sum()),
                "n_jobs": int(ssoc_mask.sum()),
            })

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} faculty × SSOC pairs")
    return df


# ──────────────────────────────────────────────
# Output 4: Faculty coverage scores
# ──────────────────────────────────────────────

def faculty_coverage_scores(sim_matrix, module_idx, job_idx, threshold):
    """For each faculty, compute what % of SSOC groups are 'covered'.

    A faculty covers an SSOC group if at least one of its modules has
    similarity >= threshold to at least one job in that group.
    """
    print(f"\nComputing faculty coverage scores (threshold={threshold})...")

    faculties = module_idx["faculty"].values
    ssoc_groups = job_idx[SSOC_GROUP_COL].values

    unique_faculties = sorted(module_idx["faculty"].dropna().unique())
    unique_ssoc = sorted(job_idx[SSOC_GROUP_COL].dropna().unique())
    n_total_groups = len(unique_ssoc)

    rows = []
    for fac in unique_faculties:
        fac_mask = faculties == fac
        fac_sims = sim_matrix[fac_mask]

        n_covered = 0
        covered_groups = []

        for ssoc in unique_ssoc:
            ssoc_mask = ssoc_groups == ssoc
            block = fac_sims[:, ssoc_mask]

            if block.size > 0 and block.max() >= threshold:
                n_covered += 1
                covered_groups.append(ssoc)

        coverage_pct = round(n_covered / n_total_groups * 100, 2)

        rows.append({
            "faculty": fac,
            "n_modules": int(fac_mask.sum()),
            "n_ssoc_groups_total": n_total_groups,
            "n_ssoc_groups_covered": n_covered,
            "coverage_pct": coverage_pct,
        })

    df = pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
    print(f"  → {len(df)} faculties scored")
    return df


# ──────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────

def save_outputs(top_k_jobs_df, top_k_modules_df, heatmap_df, coverage_df):
    """Save all analysis outputs as CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "top_k_jobs_per_module.csv": top_k_jobs_df,
        "top_k_modules_per_job.csv": top_k_modules_df,
        "faculty_ssoc_heatmap.csv": heatmap_df,
        "faculty_coverage_scores.csv": coverage_df,
    }

    print(f"\nSaving outputs to: {OUTPUT_DIR}")
    print("-" * 60)
    for filename, df in files.items():
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {filename:45s} {size_mb:>8.2f} MB  ({len(df):,} rows)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SIMILARITY ANALYSIS PIPELINE")
    print(f"  TOP_K={TOP_K}  SSOC_GROUP={SSOC_GROUP_COL}  COVERAGE_THRESHOLD={COVERAGE_THRESHOLD}")
    print("=" * 70)

    # Load
    module_emb, job_emb, module_idx, job_idx = load_data()
    sim_matrix = compute_similarity_matrix(module_emb, job_emb)

    # Analyse
    top_k_jobs_df = top_k_jobs_per_module(sim_matrix, module_idx, job_idx, TOP_K)
    top_k_modules_df = top_k_modules_per_job(sim_matrix, module_idx, job_idx, TOP_K)
    heatmap_df = faculty_ssoc_heatmap(sim_matrix, module_idx, job_idx)
    coverage_df = faculty_coverage_scores(sim_matrix, module_idx, job_idx, COVERAGE_THRESHOLD)

    # Save
    save_outputs(top_k_jobs_df, top_k_modules_df, heatmap_df, coverage_df)

    # Quick preview
    print(f"\n{'='*70}")
    print("PREVIEW: Top 5 faculties by coverage")
    print("=" * 70)
    print(coverage_df.head().to_string(index=False))

    print(f"\n{'='*70}")
    print("DONE. Outputs saved to data/analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()
