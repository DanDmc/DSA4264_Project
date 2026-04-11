"""
similarity_analysis.py — Quick exploratory analysis of module-job alignment
---------------------------------------------------------------------------
Loads precomputed embeddings, computes the full cosine similarity matrix,
and produces summary stats + interpretable examples.

Usage (from repo root):
    python -m src.analysis.similarity_analysis

Outputs to EMBEDDINGS_DIR/../analysis_outputs/ (or prints to console).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBEDDINGS_DIR, EMBEDDING_MODEL

EMB_DIR = EMBEDDINGS_DIR / "whole_text"
MODEL_TAG = EMBEDDING_MODEL.split("/")[-1]
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "similarity_analysis_outputs"


def load():
    mod_emb = np.load(EMB_DIR / f"module_embeddings_{MODEL_TAG}.npy")
    job_emb = np.load(EMB_DIR / f"job_embeddings_{MODEL_TAG}.npy")
    mod_idx = pd.read_csv(EMB_DIR / "module_index.csv")
    job_idx = pd.read_csv(EMB_DIR / "job_index.csv")
    return mod_emb, job_emb, mod_idx, job_idx


def main():
    print("Loading embeddings...")
    mod_emb, job_emb, mod_idx, job_idx = load()
    print(f"  Modules: {mod_emb.shape[0]}, Jobs: {job_emb.shape[0]}")

    # ── full cosine similarity matrix (modules × jobs) ──
    # embeddings are L2-normalized, so dot product = cosine
    print("Computing similarity matrix...")
    sim_matrix = mod_emb @ job_emb.T  # (n_modules, n_jobs)

    # ── per-module stats ──
    mod_stats = mod_idx[["module code", "title", "faculty"]].copy()
    mod_stats["mean_sim"] = sim_matrix.mean(axis=1)
    mod_stats["max_sim"] = sim_matrix.max(axis=1)
    mod_stats["top5_mean_sim"] = np.sort(sim_matrix, axis=1)[:, -5:].mean(axis=1)
    mod_stats = mod_stats.sort_values("top5_mean_sim", ascending=False)

    # ── per-SSOC category stats ──
    # group jobs by ssoc_major_title, compute mean similarity per module-group pair
    ssoc_col = "ssoc_major_title"
    ssoc_groups = job_idx[ssoc_col].values
    unique_ssoc = sorted(job_idx[ssoc_col].dropna().unique())

    ssoc_rows = []
    for ssoc in unique_ssoc:
        mask = ssoc_groups == ssoc
        n_jobs = mask.sum()
        if n_jobs == 0:
            continue
        group_sims = sim_matrix[:, mask].mean(axis=1)  # mean sim of each module to this SSOC group
        ssoc_rows.append({
            "ssoc_major_title": ssoc,
            "n_jobs": n_jobs,
            "overall_mean_sim": float(group_sims.mean()),
            "top_module": mod_idx.iloc[group_sims.argmax()]["module code"],
            "top_module_title": mod_idx.iloc[group_sims.argmax()]["title"],
            "top_module_sim": float(group_sims.max()),
        })
    ssoc_stats = pd.DataFrame(ssoc_rows).sort_values("overall_mean_sim", ascending=False)

    # ── save outputs ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mod_stats.to_csv(OUT_DIR / "per_module_alignment.csv", index=False)
    ssoc_stats.to_csv(OUT_DIR / "per_ssoc_alignment.csv", index=False)
    np.save(OUT_DIR / "similarity_matrix.npy", sim_matrix.astype(np.float16))

    # ── print summaries ──
    print(f"\n{'=' * 70}")
    print("TOP 15 MODULES (by top-5 mean similarity to jobs)")
    print(f"{'=' * 70}")
    for _, r in mod_stats.head(15).iterrows():
        print(f"  {r['module code']:10s} top5={r['top5_mean_sim']:.4f}  max={r['max_sim']:.4f}  {r['title'][:50]}")

    print(f"\n{'=' * 70}")
    print("BOTTOM 10 MODULES (lowest alignment)")
    print(f"{'=' * 70}")
    for _, r in mod_stats.tail(10).iterrows():
        print(f"  {r['module code']:10s} top5={r['top5_mean_sim']:.4f}  max={r['max_sim']:.4f}  {r['title'][:50]}")

    print(f"\n{'=' * 70}")
    print("PER-SSOC MAJOR GROUP ALIGNMENT")
    print(f"{'=' * 70}")
    for _, r in ssoc_stats.iterrows():
        print(f"  [{r['n_jobs']:4d} jobs] mean={r['overall_mean_sim']:.4f}  "
              f"best={r['top_module']:8s} ({r['top_module_sim']:.4f})  {r['ssoc_major_title'][:45]}")

    # ── interpretability: show top-3 job matches for 5 diverse modules ──
    print(f"\n{'=' * 70}")
    print("EXAMPLE MATCHES (top-3 jobs for 5 sample modules)")
    print(f"{'=' * 70}")
    sample_indices = np.linspace(0, len(mod_idx) - 1, 5, dtype=int)
    for idx in sample_indices:
        mod = mod_idx.iloc[idx]
        top3 = np.argsort(sim_matrix[idx])[::-1][:3]
        print(f"\n  [{mod['module code']}] {mod['title']}")
        print(f"  Faculty: {mod['faculty']}")
        for rank, j_idx in enumerate(top3, 1):
            job = job_idx.iloc[j_idx]
            print(f"    #{rank} (sim={sim_matrix[idx, j_idx]:.4f}) {job['title'][:55]}")
            print(f"       SSOC: {job.get('ssoc_minor_title', 'N/A')}")

    print(f"\n{'=' * 70}")
    print(f"Saved to: {OUT_DIR}")
    print(f"  per_module_alignment.csv  ({len(mod_stats)} modules)")
    print(f"  per_ssoc_alignment.csv    ({len(ssoc_stats)} SSOC groups)")
    print(f"  similarity_matrix.npy     ({sim_matrix.shape}, float16)")


if __name__ == "__main__":
    main()
