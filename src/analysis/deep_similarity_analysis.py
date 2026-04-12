"""
deep_similarity_analysis.py — Granular module-job alignment analysis
---------------------------------------------------------------------
Builds on the initial similarity_analysis.py by operating at finer
SSOC levels, using top-K matching, and aggregating by degree/faculty/
department. Designed for MOE policy officer audience.

Usage (from repo root):
    python -m src.analysis.deep_similarity_analysis

Outputs:
    Large CSVs     → DATA_ROOT/results/similarity_analysis_results/
    Summaries/JSON → REPO_ROOT/outputs/deep_analysis_outputs/

All paths are sourced from config.py. Parameters are configurable at
the top of this file under ANALYSIS PARAMETERS.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── imports from project config ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    DEGREE_MODULE_MAPPING,
    RESULTS_DIR,
)

# ──────────────────────────────────────────────
# ANALYSIS PARAMETERS (adjust as needed)
# ──────────────────────────────────────────────
TOP_K = 10                      # top-K job matches per module / degree
THRESHOLD_PERCENTILES = [90, 95, 99]  # percentiles to evaluate as thresholds
BREADTH_THRESHOLD = None        # set manually, or auto-detected from distribution
BREADTH_SSOC_LEVEL = "ssoc_minor_title"  # level at which to count distinct groups

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
EMB_DIR = EMBEDDINGS_DIR / "whole_text"
MODEL_TAG = EMBEDDING_MODEL.split("/")[-1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Large CSVs → OneDrive
RESULTS_OUT = RESULTS_DIR / "similarity_analysis_results"
# Summaries → repo
SUMMARY_OUT = REPO_ROOT / "outputs" / "deep_analysis_outputs"

# Previously computed similarity matrix (from similarity_analysis.py)
SIM_MATRIX_PATH = REPO_ROOT / "outputs" / "similarity_analysis_outputs" / "similarity_matrix.npy"


# ══════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════

def load_data():
    """Load similarity matrix, index CSVs, and degree mapping."""
    print("Loading data...")

    # Prefer the precomputed float16 matrix if it exists
    if SIM_MATRIX_PATH.exists():
        print(f"  Loading precomputed similarity matrix from {SIM_MATRIX_PATH}")
        sim_matrix = np.load(SIM_MATRIX_PATH).astype(np.float32)
    else:
        print("  Precomputed matrix not found — computing from embeddings...")
        mod_emb = np.load(EMB_DIR / f"module_embeddings_{MODEL_TAG}.npy")
        job_emb = np.load(EMB_DIR / f"job_embeddings_{MODEL_TAG}.npy")
        sim_matrix = (mod_emb @ job_emb.T).astype(np.float32)
        print(f"  Computed similarity matrix: {sim_matrix.shape}")

    mod_idx = pd.read_csv(EMB_DIR / "module_index.csv")
    job_idx = pd.read_csv(EMB_DIR / "job_index.csv")

    print(f"  Modules: {len(mod_idx)}, Jobs: {len(job_idx)}, Matrix: {sim_matrix.shape}")

    # Degree mapping (optional — script still runs without it)
    deg_map = None
    if DEGREE_MODULE_MAPPING.exists():
        deg_map = pd.read_csv(DEGREE_MODULE_MAPPING)
        print(f"  Degree mapping: {len(deg_map)} rows, {deg_map['major'].nunique()} majors")
    else:
        print(f"  WARNING: Degree mapping not found at {DEGREE_MODULE_MAPPING}")
        print("           Degree/faculty aggregation will be skipped.")

    return sim_matrix, mod_idx, job_idx, deg_map


# ══════════════════════════════════════════════
# A. TOP-K MATCHING PER MODULE
# ══════════════════════════════════════════════

def top_k_per_module(sim_matrix, mod_idx, job_idx, k=TOP_K):
    """
    For each module, find its top-K most similar jobs.
    Returns a long-form DataFrame: one row per (module, rank) pair.
    """
    print(f"\n[A] Computing top-{k} job matches per module...")
    n_modules = sim_matrix.shape[0]

    # argsort descending — only need top-K, use argpartition for speed
    top_k_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]

    records = []
    for i in range(n_modules):
        tk_idx = top_k_indices[i]
        tk_sims = sim_matrix[i, tk_idx]
        # sort within the top-K by descending similarity
        order = np.argsort(tk_sims)[::-1]
        tk_idx = tk_idx[order]
        tk_sims = tk_sims[order]

        mod = mod_idx.iloc[i]
        for rank, (j_idx, score) in enumerate(zip(tk_idx, tk_sims), 1):
            job = job_idx.iloc[j_idx]
            records.append({
                "module_code": mod["module code"],
                "module_title": mod["title"],
                "module_faculty": mod["faculty"],
                "module_department": mod.get("department", ""),
                "rank": rank,
                "job_id": job["job_id"],
                "job_title": job["title"],
                "ssoc_code": job.get("ssoc_code", ""),
                "ssoc_major_title": job.get("ssoc_major_title", ""),
                "ssoc_submajor_title": job.get("ssoc_submajor_title", ""),
                "ssoc_minor_title": job.get("ssoc_minor_title", ""),
                "ssoc_unit_title": job.get("ssoc_unit_title", ""),
                "similarity": float(score),
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df)} rows ({n_modules} modules × {k} matches)")
    return df


# ══════════════════════════════════════════════
# B. SSOC MINOR / UNIT LEVEL BREAKDOWN
# ══════════════════════════════════════════════

def ssoc_granular_alignment(sim_matrix, mod_idx, job_idx,
                            ssoc_level="ssoc_minor_title", top_n=3):
    """
    For each SSOC group at the given level, compute alignment stats.
    Uses top-N mean (not mean-over-all) to avoid washout.

    Args:
        ssoc_level: column name — 'ssoc_minor_title' or 'ssoc_unit_title'
        top_n: number of top modules to average for each SSOC group
    """
    print(f"\n[B] Computing alignment at SSOC level: {ssoc_level} ...")
    ssoc_values = job_idx[ssoc_level].values
    unique_groups = sorted(job_idx[ssoc_level].dropna().unique())

    rows = []
    for group in unique_groups:
        mask = ssoc_values == group
        n_jobs = int(mask.sum())
        if n_jobs == 0:
            continue

        # For each module, compute its mean similarity to jobs in this group
        group_sims = sim_matrix[:, mask].mean(axis=1)  # (n_modules,)

        # Top-N modules for this SSOC group
        top_mod_indices = np.argsort(group_sims)[::-1][:top_n]
        top_mods = []
        for rank, m_idx in enumerate(top_mod_indices, 1):
            top_mods.append({
                f"top{rank}_module": mod_idx.iloc[m_idx]["module code"],
                f"top{rank}_title": mod_idx.iloc[m_idx]["title"],
                f"top{rank}_sim": float(group_sims[m_idx]),
            })

        # Also get the SSOC major for cross-referencing
        sample_job = job_idx[job_idx[ssoc_level] == group].iloc[0]

        row = {
            "ssoc_level": ssoc_level,
            "ssoc_group": group,
            "ssoc_major_title": sample_job.get("ssoc_major_title", ""),
            "n_jobs": n_jobs,
            "overall_mean_sim": float(group_sims.mean()),
            "top_module_mean_sim": float(group_sims[top_mod_indices].mean()),
        }
        for d in top_mods:
            row.update(d)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("top_module_mean_sim", ascending=False)
    print(f"  → {len(df)} SSOC groups at {ssoc_level} level")
    return df


# ══════════════════════════════════════════════
# C. DEGREE-LEVEL AGGREGATION
# ══════════════════════════════════════════════

def degree_level_analysis(sim_matrix, mod_idx, job_idx, deg_map, k=TOP_K):
    """
    For each degree (major), aggregate similarity scores across its modules.
    Computes: top-K job matches per degree, mean alignment, breadth.

    Uses degree_faculty (from degree mapping) as the faculty column.
    """
    if deg_map is None:
        print("\n[C] Skipping degree-level analysis (no degree mapping).")
        return None, None

    print(f"\n[C] Computing degree-level alignment...")

    # Build module_code → row index lookup
    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }

    # For each degree, find which modules are in the similarity matrix
    majors = deg_map["major"].unique()
    degree_rows = []
    degree_topk_rows = []

    for major in majors:
        deg_modules = deg_map[deg_map["major"] == major]
        deg_faculty = deg_modules["degree_faculty"].iloc[0]

        # Find which module codes are in the embedding index
        codes = deg_modules["module_code"].unique()
        matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]

        if len(matched_indices) == 0:
            continue

        # Sub-matrix: modules in this degree × all jobs
        sub_sim = sim_matrix[matched_indices, :]  # (n_deg_modules, n_jobs)

        # Per-job: take max similarity across the degree's modules
        # (a degree is "aligned" to a job if ANY of its modules are aligned)
        job_max_sims = sub_sim.max(axis=0)  # (n_jobs,)

        # Top-K jobs for this degree
        top_k_idx = np.argpartition(job_max_sims, -k)[-k:]
        top_k_sims = job_max_sims[top_k_idx]
        order = np.argsort(top_k_sims)[::-1]
        top_k_idx = top_k_idx[order]
        top_k_sims = top_k_sims[order]

        for rank, (j_idx, score) in enumerate(zip(top_k_idx, top_k_sims), 1):
            job = job_idx.iloc[j_idx]
            degree_topk_rows.append({
                "major": major,
                "degree_faculty": deg_faculty,
                "rank": rank,
                "job_id": job["job_id"],
                "job_title": job["title"],
                "ssoc_code": job.get("ssoc_code", ""),
                "ssoc_minor_title": job.get("ssoc_minor_title", ""),
                "ssoc_unit_title": job.get("ssoc_unit_title", ""),
                "similarity": float(score),
                # Track which module(s) drove this match
                "best_module": mod_idx.iloc[
                    matched_indices[sub_sim[:, j_idx].argmax()]
                ]["module code"],
            })

        # Summary stats for this degree
        degree_rows.append({
            "major": major,
            "degree_faculty": deg_faculty,
            "n_modules_total": len(codes),
            "n_modules_matched": len(matched_indices),
            "top_k_mean_sim": float(top_k_sims.mean()),
            "top_k_max_sim": float(top_k_sims.max()),
            "overall_mean_sim": float(job_max_sims.mean()),
        })

    degree_summary = pd.DataFrame(degree_rows).sort_values(
        "top_k_mean_sim", ascending=False
    )
    degree_topk = pd.DataFrame(degree_topk_rows)

    print(f"  → {len(degree_summary)} degrees analysed")
    print(f"  → {len(degree_topk)} degree-job match rows")
    return degree_summary, degree_topk


# ══════════════════════════════════════════════
# D. ALIGNMENT BREADTH
# ══════════════════════════════════════════════

def alignment_breadth(sim_matrix, mod_idx, job_idx, threshold,
                      ssoc_level=BREADTH_SSOC_LEVEL):
    """
    Per module: count distinct SSOC groups (at ssoc_level) where the module
    has at least one job above the threshold. High = generalist, low = specialist.
    """
    print(f"\n[D] Computing alignment breadth (threshold={threshold:.4f}, "
          f"level={ssoc_level})...")

    ssoc_labels = job_idx[ssoc_level].values
    unique_groups = sorted(job_idx[ssoc_level].dropna().unique())

    # Boolean mask: which jobs each module is "aligned" with
    aligned = sim_matrix >= threshold  # (n_modules, n_jobs)

    records = []
    for i in range(len(mod_idx)):
        aligned_jobs = aligned[i]  # boolean vector, length n_jobs
        if not aligned_jobs.any():
            n_groups = 0
            groups_list = []
        else:
            aligned_ssoc = ssoc_labels[aligned_jobs]
            groups_list = sorted(set(g for g in aligned_ssoc if pd.notna(g)))
            n_groups = len(groups_list)

        mod = mod_idx.iloc[i]
        records.append({
            "module_code": mod["module code"],
            "module_title": mod["title"],
            "module_faculty": mod["faculty"],
            "module_department": mod.get("department", ""),
            "n_aligned_jobs": int(aligned_jobs.sum()),
            "n_aligned_ssoc_groups": n_groups,
            "aligned_ssoc_groups": "; ".join(groups_list) if n_groups <= 20 else f"{n_groups} groups",
        })

    df = pd.DataFrame(records).sort_values("n_aligned_ssoc_groups", ascending=False)
    print(f"  → Modules with ≥1 aligned group: "
          f"{(df['n_aligned_ssoc_groups'] > 0).sum()} / {len(df)}")
    return df


def alignment_breadth_degrees(sim_matrix, mod_idx, job_idx, deg_map,
                              threshold, ssoc_level=BREADTH_SSOC_LEVEL):
    """Aggregate alignment breadth to degree level."""
    if deg_map is None:
        print("\n[D'] Skipping degree-level breadth (no degree mapping).")
        return None

    print(f"\n[D'] Computing degree-level alignment breadth...")

    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }
    ssoc_labels = job_idx[ssoc_level].values
    majors = deg_map["major"].unique()

    records = []
    for major in majors:
        deg_modules = deg_map[deg_map["major"] == major]
        deg_faculty = deg_modules["degree_faculty"].iloc[0]
        codes = deg_modules["module_code"].unique()
        matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]

        if len(matched_indices) == 0:
            continue

        sub_sim = sim_matrix[matched_indices, :]
        # Job is "aligned" to degree if ANY module exceeds threshold
        aligned = sub_sim.max(axis=0) >= threshold
        if not aligned.any():
            n_groups = 0
        else:
            groups = set(g for g in ssoc_labels[aligned] if pd.notna(g))
            n_groups = len(groups)

        records.append({
            "major": major,
            "degree_faculty": deg_faculty,
            "n_modules_matched": len(matched_indices),
            "n_aligned_jobs": int(aligned.sum()),
            "n_aligned_ssoc_groups": n_groups,
        })

    df = pd.DataFrame(records).sort_values("n_aligned_ssoc_groups", ascending=False)
    print(f"  → {len(df)} degrees analysed for breadth")
    return df


# ══════════════════════════════════════════════
# E. FACULTY & DEPARTMENT SUMMARY
# ══════════════════════════════════════════════

def faculty_department_summary(top_k_df, breadth_df, group_col, label):
    """
    Aggregate top-K and breadth results to faculty or department level.

    Args:
        top_k_df: output of top_k_per_module (long-form)
        breadth_df: output of alignment_breadth (per-module)
        group_col: column to group by (e.g. 'module_faculty', 'module_department')
        label: descriptive label for printing
    """
    print(f"\n[E] Building {label} summary (grouping by '{group_col}')...")

    # From top-K: mean/median of the top-1 similarity per module, grouped
    top1 = top_k_df[top_k_df["rank"] == 1].copy()
    sim_agg = (
        top1
        .groupby(group_col)["similarity"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "mean_top1_sim", "median": "median_top1_sim",
                         "count": "n_modules"})
    )

    # From breadth: mean breadth per group
    breadth_agg = (
        breadth_df
        .groupby(group_col)["n_aligned_ssoc_groups"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "mean_breadth", "median": "median_breadth"})
    )

    summary = sim_agg.join(breadth_agg, how="outer").reset_index()
    summary = summary.sort_values("mean_top1_sim", ascending=False)
    print(f"  → {len(summary)} {label} groups")
    return summary


def degree_faculty_summary(degree_summary):
    """Aggregate degree-level stats to faculty level."""
    if degree_summary is None:
        return None
    print(f"\n[E'] Building degree-faculty summary...")
    agg = (
        degree_summary
        .groupby("degree_faculty")
        .agg(
            n_degrees=("major", "count"),
            mean_top_k_sim=("top_k_mean_sim", "mean"),
            median_top_k_sim=("top_k_mean_sim", "median"),
            mean_overall_sim=("overall_mean_sim", "mean"),
        )
        .reset_index()
        .sort_values("mean_top_k_sim", ascending=False)
    )
    print(f"  → {len(agg)} faculty groups")
    return agg


# ══════════════════════════════════════════════
# F. THRESHOLD ANALYSIS
# ══════════════════════════════════════════════

def threshold_analysis(sim_matrix, percentiles=THRESHOLD_PERCENTILES):
    """
    Analyse the distribution of the full similarity matrix to identify
    a meaningful alignment threshold.
    """
    print(f"\n[F] Analysing similarity score distribution...")
    flat = sim_matrix.ravel()

    stats = {
        "matrix_shape": list(sim_matrix.shape),
        "n_values": int(flat.size),
        "global_mean": float(flat.mean()),
        "global_std": float(flat.std()),
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "percentiles": {},
    }

    for p in percentiles:
        val = float(np.percentile(flat, p))
        stats["percentiles"][f"p{p}"] = val
        print(f"  P{p}: {val:.4f}")

    # Suggest threshold as P95 (top 5% of all scores)
    suggested = stats["percentiles"].get("p95", stats["percentiles"].get("p90"))
    stats["suggested_threshold"] = suggested
    print(f"  Suggested threshold (P95): {suggested:.4f}")
    return stats


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    start = datetime.now()

    # ── Load ──
    sim_matrix, mod_idx, job_idx, deg_map = load_data()

    # ── Create output directories ──
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)

    # ── F. Threshold analysis (run first — breadth depends on it) ──
    thresh_stats = threshold_analysis(sim_matrix)
    threshold = BREADTH_THRESHOLD or thresh_stats["suggested_threshold"]
    print(f"\n  Using alignment threshold: {threshold:.4f}")

    # ── A. Top-K per module ──
    topk_df = top_k_per_module(sim_matrix, mod_idx, job_idx, k=TOP_K)
    topk_df.to_csv(RESULTS_OUT / "top_k_matches_per_module.csv", index=False)
    print(f"  Saved → {RESULTS_OUT / 'top_k_matches_per_module.csv'}")

    # ── B. SSOC granular alignment ──
    ssoc_minor = ssoc_granular_alignment(
        sim_matrix, mod_idx, job_idx, ssoc_level="ssoc_minor_title"
    )
    ssoc_minor.to_csv(RESULTS_OUT / "ssoc_minor_alignment.csv", index=False)

    ssoc_unit = ssoc_granular_alignment(
        sim_matrix, mod_idx, job_idx, ssoc_level="ssoc_unit_title"
    )
    ssoc_unit.to_csv(RESULTS_OUT / "ssoc_unit_alignment.csv", index=False)
    print(f"  Saved SSOC breakdowns → {RESULTS_OUT}")

    # ── C. Degree-level analysis ──
    degree_summary, degree_topk = degree_level_analysis(
        sim_matrix, mod_idx, job_idx, deg_map, k=TOP_K
    )
    if degree_topk is not None:
        degree_topk.to_csv(RESULTS_OUT / "top_k_matches_per_degree.csv", index=False)
    if degree_summary is not None:
        degree_summary.to_csv(RESULTS_OUT / "degree_alignment.csv", index=False)
        print(f"  Saved degree outputs → {RESULTS_OUT}")

    # ── D. Alignment breadth ──
    breadth_modules = alignment_breadth(
        sim_matrix, mod_idx, job_idx, threshold=threshold
    )
    breadth_modules.to_csv(RESULTS_OUT / "alignment_breadth_modules.csv", index=False)

    breadth_degrees = alignment_breadth_degrees(
        sim_matrix, mod_idx, job_idx, deg_map, threshold=threshold
    )
    if breadth_degrees is not None:
        breadth_degrees.to_csv(RESULTS_OUT / "alignment_breadth_degrees.csv", index=False)
    print(f"  Saved breadth outputs → {RESULTS_OUT}")

    # ── E. Faculty / Department / Degree-faculty summaries ──
    faculty_summary = faculty_department_summary(
        topk_df, breadth_modules, group_col="module_faculty", label="Faculty"
    )
    faculty_summary.to_csv(SUMMARY_OUT / "faculty_summary.csv", index=False)

    dept_summary = faculty_department_summary(
        topk_df, breadth_modules, group_col="module_department", label="Department"
    )
    dept_summary.to_csv(SUMMARY_OUT / "department_summary.csv", index=False)

    if degree_summary is not None:
        deg_fac_summary = degree_faculty_summary(degree_summary)
        deg_fac_summary.to_csv(SUMMARY_OUT / "degree_faculty_summary.csv", index=False)

        # Degree summary is small enough for repo
        degree_summary.to_csv(SUMMARY_OUT / "degree_summary.csv", index=False)

    # ── SSOC summaries (top/bottom 10 at each level) ──
    ssoc_minor.head(20).to_csv(
        SUMMARY_OUT / "ssoc_minor_top20.csv", index=False
    )
    ssoc_minor.tail(20).to_csv(
        SUMMARY_OUT / "ssoc_minor_bottom20.csv", index=False
    )

    # ── Threshold + metadata JSON ──
    metadata = {
        "run_timestamp": start.isoformat(),
        "parameters": {
            "top_k": TOP_K,
            "alignment_threshold": threshold,
            "breadth_ssoc_level": BREADTH_SSOC_LEVEL,
            "embedding_model": EMBEDDING_MODEL,
        },
        "data_shape": {
            "n_modules": int(sim_matrix.shape[0]),
            "n_jobs": int(sim_matrix.shape[1]),
            "n_degrees": int(deg_map["major"].nunique()) if deg_map is not None else 0,
        },
        "threshold_analysis": thresh_stats,
    }

    with open(SUMMARY_OUT / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved summaries → {SUMMARY_OUT}")

    # ── Console summary ──
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"DEEP ANALYSIS COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 70}")

    print(f"\n  Large CSVs (OneDrive): {RESULTS_OUT}")
    for f in sorted(RESULTS_OUT.glob("*.csv")):
        print(f"    {f.name}")

    print(f"\n  Summaries (repo):      {SUMMARY_OUT}")
    for f in sorted(SUMMARY_OUT.glob("*")):
        print(f"    {f.name}")

    print(f"\n  Threshold used: {threshold:.4f}")

    # Print top-5 degree headline if available
    if degree_summary is not None:
        print(f"\n  TOP 5 DEGREES BY ALIGNMENT:")
        for _, r in degree_summary.head(5).iterrows():
            print(f"    {r['major']:<40s}  top-K mean={r['top_k_mean_sim']:.4f}  "
                  f"[{r['degree_faculty']}]")

    # Print faculty summary
    print(f"\n  FACULTY SUMMARY (module home faculty):")
    for _, r in faculty_summary.iterrows():
        print(f"    {r['module_faculty']:<45s}  "
              f"top1_sim={r['mean_top1_sim']:.4f}  "
              f"breadth={r['mean_breadth']:.1f}  "
              f"(n={int(r['n_modules'])})")

    print(f"\n  DEPARTMENT SUMMARY (top 10):")
    for _, r in dept_summary.head(10).iterrows():
        print(f"    {r['module_department']:<45s}  "
              f"top1_sim={r['mean_top1_sim']:.4f}  "
              f"breadth={r['mean_breadth']:.1f}  "
              f"(n={int(r['n_modules'])})")


if __name__ == "__main__":
    main()
