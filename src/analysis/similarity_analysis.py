"""
similarity_analysis.py — Module-job alignment analysis pipeline
----------------------------------------------------------------
Consolidated analysis script that computes all similarity-based
metrics: top-K alignment, breadth, coverage, SSOC breakdowns,
degree-level aggregation, and targeted SSOC alignment.

Replaces the earlier similarity_analysis.py and deep_similarity_analysis.py.

Usage (from repo root):
    python -m src.analysis.similarity_analysis

Outputs:
    Similarity matrix (.npy) → DATA_ROOT/results/similarity_analysis_results/
    All CSVs + metadata JSON  → REPO_ROOT/outputs/similarity_analysis_outputs/

All paths and parameters are sourced from config.py.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── project config ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    DEGREE_MODULE_MAPPING,
    SIMILARITY_RESULTS_DIR,
    ANALYSIS_TOP_K,
    ANALYSIS_BREADTH_SSOC_LEVEL,
    ANALYSIS_DEGREE_AGG_TOP_N,
    ANALYSIS_COVERAGE_THRESHOLD,
    MAJOR_SSOC_MAPPING,
)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
EMB_DIR = EMBEDDINGS_DIR / "whole_text"
MODEL_TAG = EMBEDDING_MODEL.split("/")[-1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Large binary files (.npy) → OneDrive (too big for git)
NPY_OUT = SIMILARITY_RESULTS_DIR

# All CSVs + JSON → repo (version-controlled, shareable)
CSV_OUT = REPO_ROOT / "outputs" / "similarity_analysis_outputs"


# ──────────────────────────────────────────────
# HELPER: add dense rank columns
# ──────────────────────────────────────────────

def _add_rank(df, col, rank_col, ascending=False):
    """Add a dense rank column next to the source column (rank 1 = best)."""
    df[rank_col] = df[col].rank(method="dense", ascending=ascending).astype(int)
    # Reorder so rank column appears right after its source column
    cols = list(df.columns)
    src_pos = cols.index(col)
    cols.remove(rank_col)
    cols.insert(src_pos + 1, rank_col)
    return df[cols]


# ══════════════════════════════════════════════
# STAGE 1: LOAD DATA
# ══════════════════════════════════════════════

def load_data():
    """Load similarity matrix (or recompute), index CSVs, degree mapping, and major-SSOC mapping."""
    print("=" * 70)
    print("STAGE 1: Loading data")
    print("=" * 70)

    # Check for precomputed similarity matrix (OneDrive)
    sim_matrix_path = NPY_OUT / "similarity_matrix.npy"
    if sim_matrix_path.exists():
        print(f"  Loading precomputed similarity matrix from {sim_matrix_path}")
        sim_matrix = np.load(sim_matrix_path).astype(np.float32)
    else:
        # Fallback: check old repo location
        old_path = REPO_ROOT / "outputs" / "similarity_analysis_outputs" / "similarity_matrix.npy"
        if old_path.exists():
            print(f"  Loading precomputed similarity matrix from {old_path}")
            sim_matrix = np.load(old_path).astype(np.float32)
        else:
            print("  Precomputed matrix not found — computing from embeddings...")
            mod_emb = np.load(EMB_DIR / f"module_embeddings_{MODEL_TAG}.npy")
            job_emb = np.load(EMB_DIR / f"job_embeddings_{MODEL_TAG}.npy")
            sim_matrix = (mod_emb @ job_emb.T).astype(np.float32)
            print(f"  Computed: {sim_matrix.shape}")

    mod_idx = pd.read_csv(EMB_DIR / "module_index.csv")
    job_idx = pd.read_csv(EMB_DIR / "job_index.csv")
    print(f"  Modules: {len(mod_idx)}, Jobs: {len(job_idx)}, Matrix: {sim_matrix.shape}")

    # Degree mapping (optional)
    deg_map = None
    if DEGREE_MODULE_MAPPING.exists():
        deg_map = pd.read_csv(DEGREE_MODULE_MAPPING)
        print(f"  Degree mapping: {len(deg_map)} rows, {deg_map['major'].nunique()} majors")
    else:
        print(f"  WARNING: Degree mapping not found at {DEGREE_MODULE_MAPPING}")
        print("           Degree-level analysis will be skipped.")

    # Major-SSOC mapping (optional)
    major_ssoc = None
    if MAJOR_SSOC_MAPPING.exists():
        major_ssoc = pd.read_csv(MAJOR_SSOC_MAPPING)
        print(f"  Major-SSOC mapping: {len(major_ssoc)} rows, "
              f"{major_ssoc['major'].nunique()} majors → "
              f"{major_ssoc['ssoc_minor_title'].nunique()} SSOC minor groups")
    else:
        print(f"  WARNING: Major-SSOC mapping not found at {MAJOR_SSOC_MAPPING}")
        print("           Targeted SSOC alignment will be skipped.")

    return sim_matrix, mod_idx, job_idx, deg_map, major_ssoc


# ══════════════════════════════════════════════
# STAGE 2: DISTRIBUTION STATS & THRESHOLD
# ══════════════════════════════════════════════

def compute_distribution_stats(sim_matrix):
    """Compute global distribution statistics and the coverage threshold."""
    print("\n" + "=" * 70)
    print("STAGE 2: Similarity distribution stats & threshold")
    print("=" * 70)

    flat = sim_matrix.ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    percentiles = {p: float(np.percentile(flat, p)) for p in [25, 50, 75, 90, 95, 99]}

    # Coverage threshold
    if ANALYSIS_COVERAGE_THRESHOLD is not None:
        threshold = ANALYSIS_COVERAGE_THRESHOLD
        print(f"  Using manually set threshold: {threshold:.4f}")
    else:
        threshold = mean + std
        print(f"  Auto-computed threshold (mean + 1 SD): {threshold:.4f}")

    stats = {
        "matrix_shape": list(sim_matrix.shape),
        "n_values": int(flat.size),
        "global_mean": mean,
        "global_std": std,
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "percentiles": percentiles,
        "coverage_threshold": threshold,
        "threshold_method": "manual" if ANALYSIS_COVERAGE_THRESHOLD else "mean + 1 SD",
    }

    print(f"  Mean: {mean:.4f}, SD: {std:.4f}")
    print(f"  Min: {stats['global_min']:.4f}, Max: {stats['global_max']:.4f}")
    for p, v in percentiles.items():
        print(f"  P{p}: {v:.4f}")

    return stats, threshold


# ══════════════════════════════════════════════
# STAGE 3: MODULE-LEVEL TOP-K MATCHES
# ══════════════════════════════════════════════

def module_top_k(sim_matrix, mod_idx, job_idx, k):
    """
    For each module, find its top-K most similar jobs.
    Returns:
        top_k_df: long-form DataFrame (module × rank)
        mod_summary: one row per module with top-K mean, max
    """
    print("\n" + "=" * 70)
    print(f"STAGE 3: Module-level top-{k} matches")
    print("=" * 70)

    n_modules = sim_matrix.shape[0]
    capped_k = min(k, sim_matrix.shape[1])

    # Use argpartition for efficiency
    top_k_indices = np.argpartition(sim_matrix, -capped_k, axis=1)[:, -capped_k:]

    records = []
    summary_records = []

    for i in range(n_modules):
        tk_idx = top_k_indices[i]
        tk_sims = sim_matrix[i, tk_idx]
        # Sort within top-K descending
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
                "ssoc_minor_title": job.get("ssoc_minor_title", ""),
                "ssoc_unit_title": job.get("ssoc_unit_title", ""),
                "similarity": float(score),
            })

        summary_records.append({
            "module_code": mod["module code"],
            "module_title": mod["title"],
            "module_faculty": mod["faculty"],
            "module_department": mod.get("department", ""),
            "top_k_mean_sim": float(tk_sims.mean()),
            "top_k_max_sim": float(tk_sims.max()),
            "top_k_min_sim": float(tk_sims.min()),
        })

    top_k_df = pd.DataFrame(records)
    mod_summary = pd.DataFrame(summary_records)

    print(f"  → {len(top_k_df)} rows ({n_modules} modules × {capped_k} matches)")
    return top_k_df, mod_summary


# ══════════════════════════════════════════════
# STAGE 4: MODULE-LEVEL BREADTH & COVERAGE
# ══════════════════════════════════════════════

def module_breadth_and_coverage(sim_matrix, mod_idx, top_k_df, threshold,
                                 ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """
    Per module:
      - Breadth: distinct SSOC minor groups in its top-K matches
      - Coverage: % of all jobs above the similarity threshold
    Returns a DataFrame to merge into the module summary.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 4: Module-level breadth (SSOC level: {ssoc_level}) & coverage")
    print("=" * 70)

    n_jobs = sim_matrix.shape[1]

    # Breadth: from top-K matches
    breadth = (
        top_k_df
        .groupby("module_code")[ssoc_level]
        .apply(lambda s: s.dropna().nunique())
        .reset_index()
        .rename(columns={ssoc_level: "breadth_n_ssoc_groups"})
    )

    # Coverage: from full sim matrix
    coverage_records = []
    for i in range(len(mod_idx)):
        n_above = int((sim_matrix[i] >= threshold).sum())
        coverage_records.append({
            "module_code": mod_idx.iloc[i]["module code"],
            "coverage_n_jobs": n_above,
            "coverage_pct": round(100.0 * n_above / n_jobs, 2),
        })
    coverage = pd.DataFrame(coverage_records)

    result = breadth.merge(coverage, on="module_code", how="outer")

    print(f"  → Breadth: mean={result['breadth_n_ssoc_groups'].mean():.1f} "
          f"SSOC groups per module")
    print(f"  → Coverage: mean={result['coverage_pct'].mean():.1f}% of jobs "
          f"(threshold={threshold:.4f})")

    return result


# ══════════════════════════════════════════════
# STAGE 5: SSOC-GROUP ALIGNMENT
# ══════════════════════════════════════════════

def ssoc_group_alignment(sim_matrix, mod_idx, job_idx,
                         ssoc_level="ssoc_minor_title", top_n=3):
    """
    For each SSOC group: which modules are most aligned?
    Uses mean similarity of each module to all jobs in the group.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 5: SSOC-group alignment ({ssoc_level})")
    print("=" * 70)

    ssoc_values = job_idx[ssoc_level].values
    unique_groups = sorted(job_idx[ssoc_level].dropna().unique())

    rows = []
    for group in unique_groups:
        mask = ssoc_values == group
        n_jobs = int(mask.sum())
        if n_jobs == 0:
            continue

        group_sims = sim_matrix[:, mask].mean(axis=1)  # mean sim of each module to this group

        top_mod_indices = np.argsort(group_sims)[::-1][:top_n]

        # Also get the SSOC major for context
        sample_job = job_idx[job_idx[ssoc_level] == group].iloc[0]

        row = {
            "ssoc_group": group,
            "ssoc_major_title": sample_job.get("ssoc_major_title", ""),
            "n_jobs": n_jobs,
            "overall_mean_sim": float(group_sims.mean()),
        }

        for rank, m_idx in enumerate(top_mod_indices, 1):
            row[f"top{rank}_module"] = mod_idx.iloc[m_idx]["module code"]
            row[f"top{rank}_title"] = mod_idx.iloc[m_idx]["title"]
            row[f"top{rank}_sim"] = float(group_sims[m_idx])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("overall_mean_sim", ascending=False)

    # Add rank
    df = _add_rank(df, "overall_mean_sim", "rank_overall_mean_sim")

    print(f"  → {len(df)} SSOC groups at {ssoc_level} level")
    return df


# ══════════════════════════════════════════════
# STAGE 6: DEGREE-LEVEL ANALYSIS
# ══════════════════════════════════════════════

def _compute_degree_job_scores(sub_sim, top_n):
    """
    For a degree's sub-matrix (n_deg_modules × n_jobs), compute
    a per-job alignment score as mean of top-N most similar modules.
    """
    n_mods = sub_sim.shape[0]
    n_use = min(top_n, n_mods)

    if n_mods <= top_n:
        # All modules contribute
        return sub_sim.mean(axis=0)
    else:
        # Partition along module axis to get top-N per job
        top_n_sims = np.partition(sub_sim, -n_use, axis=0)[-n_use:, :]
        return top_n_sims.mean(axis=0)


def degree_level_analysis(sim_matrix, mod_idx, job_idx, deg_map, k, top_n,
                          threshold, ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """
    For each degree:
      - Top-K jobs (using mean-of-top-N-modules aggregation)
      - Breadth (distinct SSOC groups in top-K)
      - Coverage (% of jobs above threshold)
    """
    if deg_map is None:
        print("\n" + "=" * 70)
        print("STAGE 6: Degree-level analysis — SKIPPED (no degree mapping)")
        print("=" * 70)
        return None, None

    print("\n" + "=" * 70)
    print(f"STAGE 6: Degree-level analysis (top-{k} jobs, "
          f"mean of top-{top_n} modules per job)")
    print("=" * 70)

    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }
    ssoc_labels = job_idx[ssoc_level].values
    n_jobs = sim_matrix.shape[1]
    majors = deg_map["major"].unique()

    degree_summary_rows = []
    degree_topk_rows = []

    for major in majors:
        deg_modules = deg_map[deg_map["major"] == major]
        deg_faculty = deg_modules["degree_faculty"].iloc[0]

        codes = deg_modules["module_code"].unique()
        matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]

        if len(matched_indices) == 0:
            continue

        sub_sim = sim_matrix[matched_indices, :]  # (n_deg_modules, n_jobs)

        # Per-job score: mean of top-N modules
        degree_job_scores = _compute_degree_job_scores(sub_sim, top_n)

        # Top-K jobs
        capped_k = min(k, len(degree_job_scores))
        top_k_idx = np.argpartition(degree_job_scores, -capped_k)[-capped_k:]
        top_k_sims = degree_job_scores[top_k_idx]
        order = np.argsort(top_k_sims)[::-1]
        top_k_idx = top_k_idx[order]
        top_k_sims = top_k_sims[order]

        # Breadth: distinct SSOC groups in top-K
        top_k_ssoc = ssoc_labels[top_k_idx]
        breadth = len(set(g for g in top_k_ssoc if pd.notna(g)))

        # Coverage: % of jobs above threshold
        n_above = int((degree_job_scores >= threshold).sum())
        coverage_pct = round(100.0 * n_above / n_jobs, 2)

        # Long-form top-K rows
        for rank, (j_idx, score) in enumerate(zip(top_k_idx, top_k_sims), 1):
            job = job_idx.iloc[j_idx]
            # Find which module drove this match (highest individual sim)
            best_mod_local = sub_sim[:, j_idx].argmax()
            best_mod_code = mod_idx.iloc[matched_indices[best_mod_local]]["module code"]

            degree_topk_rows.append({
                "major": major,
                "degree_faculty": deg_faculty,
                "rank": rank,
                "job_id": job["job_id"],
                "job_title": job["title"],
                "ssoc_minor_title": job.get("ssoc_minor_title", ""),
                "ssoc_unit_title": job.get("ssoc_unit_title", ""),
                "similarity": float(score),
                "best_module": best_mod_code,
            })

        degree_summary_rows.append({
            "major": major,
            "degree_faculty": deg_faculty,
            "n_modules_total": len(codes),
            "n_modules_matched": len(matched_indices),
            "top_k_mean_sim": float(top_k_sims.mean()),
            "top_k_max_sim": float(top_k_sims.max()),
            "overall_mean_sim": float(degree_job_scores.mean()),
            "breadth_n_ssoc_groups": breadth,
            "coverage_n_jobs": n_above,
            "coverage_pct": coverage_pct,
        })

    degree_summary = pd.DataFrame(degree_summary_rows).sort_values(
        "top_k_mean_sim", ascending=False
    )
    degree_topk = pd.DataFrame(degree_topk_rows)

    # Add rank columns
    degree_summary = _add_rank(degree_summary, "top_k_mean_sim", "rank_top_k_sim")
    degree_summary = _add_rank(degree_summary, "breadth_n_ssoc_groups", "rank_breadth")
    degree_summary = _add_rank(degree_summary, "coverage_pct", "rank_coverage")

    print(f"  → {len(degree_summary)} degrees analysed")
    print(f"  → {len(degree_topk)} degree-job match rows")
    return degree_summary, degree_topk


# ══════════════════════════════════════════════
# STAGE 7: TARGETED SSOC ALIGNMENT
# ══════════════════════════════════════════════

def targeted_ssoc_alignment(sim_matrix, mod_idx, job_idx, deg_map,
                            major_ssoc, k, top_n,
                            ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """
    For each degree with a curated SSOC mapping:
      - Filter jobs to only those in the degree's mapped SSOC groups
      - Compute top-K mean similarity on the filtered subset
      - Compare to overall top-K mean
    """
    if deg_map is None or major_ssoc is None:
        print("\n" + "=" * 70)
        print("STAGE 7: Targeted SSOC alignment — SKIPPED")
        print("=" * 70)
        return None

    print("\n" + "=" * 70)
    print("STAGE 7: Targeted SSOC alignment")
    print("=" * 70)

    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }
    ssoc_labels = job_idx[ssoc_level].values

    rows = []
    for major in major_ssoc["major"].unique():
        # Check this major exists in degree mapping
        if major not in deg_map["major"].values:
            continue

        deg_modules = deg_map[deg_map["major"] == major]
        codes = deg_modules["module_code"].unique()
        matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]

        if len(matched_indices) == 0:
            continue

        # Get this degree's mapped SSOC groups
        mapped_groups = major_ssoc[major_ssoc["major"] == major]["ssoc_minor_title"].values
        mapped_mask = np.isin(ssoc_labels, mapped_groups)
        n_mapped_jobs = int(mapped_mask.sum())

        if n_mapped_jobs == 0:
            continue

        sub_sim = sim_matrix[matched_indices, :]
        degree_job_scores = _compute_degree_job_scores(sub_sim, top_n)

        # Overall top-K
        capped_k_all = min(k, len(degree_job_scores))
        overall_top_k = np.sort(degree_job_scores)[::-1][:capped_k_all]
        overall_top_k_mean = float(overall_top_k.mean())

        # Targeted: only jobs in mapped SSOC groups
        targeted_scores = degree_job_scores[mapped_mask]
        capped_k_targeted = min(k, len(targeted_scores))
        targeted_top_k = np.sort(targeted_scores)[::-1][:capped_k_targeted]
        targeted_top_k_mean = float(targeted_top_k.mean())

        rows.append({
            "major": major,
            "degree_faculty": deg_modules["degree_faculty"].iloc[0],
            "mapped_ssoc_groups": "; ".join(sorted(mapped_groups)),
            "n_mapped_jobs": n_mapped_jobs,
            "targeted_top_k_mean": targeted_top_k_mean,
            "overall_top_k_mean": overall_top_k_mean,
            "targeted_vs_overall_ratio": round(targeted_top_k_mean / overall_top_k_mean, 3)
                if overall_top_k_mean > 0 else None,
        })

    df = pd.DataFrame(rows).sort_values("targeted_top_k_mean", ascending=False)

    # Add rank columns
    df = _add_rank(df, "targeted_top_k_mean", "rank_targeted")
    df = _add_rank(df, "targeted_vs_overall_ratio", "rank_ratio")

    print(f"  → {len(df)} degrees with targeted SSOC analysis")

    if len(df) > 0:
        mean_ratio = df["targeted_vs_overall_ratio"].mean()
        print(f"  → Mean targeted/overall ratio: {mean_ratio:.3f}")
        print(f"     (>1 = degree is more aligned with its intended fields)")

    return df


# ══════════════════════════════════════════════
# STAGE 8: FACULTY / DEPARTMENT / DEGREE-FACULTY SUMMARIES
# ══════════════════════════════════════════════

def build_summaries(mod_summary, degree_summary):
    """Build aggregated summaries at faculty, department, and degree-faculty levels."""
    print("\n" + "=" * 70)
    print("STAGE 8: Faculty / Department / Degree-faculty summaries")
    print("=" * 70)

    results = {}

    # Module-home faculty summary
    faculty_summary = (
        mod_summary
        .groupby("module_faculty")
        .agg(
            n_modules=("module_code", "count"),
            mean_top_k_sim=("top_k_mean_sim", "mean"),
            median_top_k_sim=("top_k_mean_sim", "median"),
            mean_breadth=("breadth_n_ssoc_groups", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .reset_index()
        .sort_values("mean_top_k_sim", ascending=False)
    )
    results["faculty_summary"] = faculty_summary
    print(f"  → Faculty summary: {len(faculty_summary)} groups")

    # Department summary
    dept_summary = (
        mod_summary
        .groupby("module_department")
        .agg(
            n_modules=("module_code", "count"),
            mean_top_k_sim=("top_k_mean_sim", "mean"),
            median_top_k_sim=("top_k_mean_sim", "median"),
            mean_breadth=("breadth_n_ssoc_groups", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .reset_index()
        .sort_values("mean_top_k_sim", ascending=False)
    )
    results["department_summary"] = dept_summary
    print(f"  → Department summary: {len(dept_summary)} groups")

    # Degree-faculty summary
    if degree_summary is not None:
        deg_fac_summary = (
            degree_summary
            .groupby("degree_faculty")
            .agg(
                n_degrees=("major", "count"),
                mean_top_k_sim=("top_k_mean_sim", "mean"),
                median_top_k_sim=("top_k_mean_sim", "median"),
                mean_breadth=("breadth_n_ssoc_groups", "mean"),
                mean_coverage_pct=("coverage_pct", "mean"),
            )
            .reset_index()
            .sort_values("mean_top_k_sim", ascending=False)
        )
        results["degree_faculty_summary"] = deg_fac_summary
        print(f"  → Degree-faculty summary: {len(deg_fac_summary)} groups")

    return results


# ══════════════════════════════════════════════
# STAGE 9: THRESHOLD STABILITY CHECK
# ══════════════════════════════════════════════

def threshold_stability_check(sim_matrix, mod_idx, deg_map, dist_stats, top_n):
    """
    Compute degree-level coverage rankings at 3 different thresholds
    and report Spearman rank correlation to demonstrate robustness.
    """
    if deg_map is None:
        return None

    print("\n" + "=" * 70)
    print("STAGE 9: Threshold stability check")
    print("=" * 70)

    mean = dist_stats["global_mean"]
    std = dist_stats["global_std"]
    thresholds = {
        "mean+0.5sd": mean + 0.5 * std,
        "mean+1.0sd": mean + 1.0 * std,
        "mean+1.5sd": mean + 1.5 * std,
    }

    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }
    n_jobs = sim_matrix.shape[1]
    majors = deg_map["major"].unique()

    # Compute coverage for each threshold
    coverage_by_threshold = {}
    for label, thresh in thresholds.items():
        coverages = {}
        for major in majors:
            deg_modules = deg_map[deg_map["major"] == major]
            codes = deg_modules["module_code"].unique()
            matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]
            if len(matched_indices) == 0:
                continue
            sub_sim = sim_matrix[matched_indices, :]
            degree_job_scores = _compute_degree_job_scores(sub_sim, top_n)
            coverages[major] = (degree_job_scores >= thresh).sum() / n_jobs
        coverage_by_threshold[label] = coverages
        print(f"  Threshold {label} = {thresh:.4f}: "
              f"mean coverage = {np.mean(list(coverages.values())):.3%}")

    # Spearman correlations between rankings
    labels = list(thresholds.keys())
    stability = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            common = set(coverage_by_threshold[labels[i]].keys()) & \
                     set(coverage_by_threshold[labels[j]].keys())
            vals_i = [coverage_by_threshold[labels[i]][m] for m in common]
            vals_j = [coverage_by_threshold[labels[j]][m] for m in common]
            rho, pval = spearmanr(vals_i, vals_j)
            pair = f"{labels[i]} vs {labels[j]}"
            stability[pair] = {"spearman_rho": round(rho, 4), "p_value": round(pval, 6)}
            print(f"  Rank correlation {pair}: ρ={rho:.4f} (p={pval:.6f})")

    return {
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "rank_correlations": stability,
    }


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    start = datetime.now()
    k = ANALYSIS_TOP_K
    top_n = ANALYSIS_DEGREE_AGG_TOP_N

    # ── Stage 1: Load ──
    sim_matrix, mod_idx, job_idx, deg_map, major_ssoc = load_data()

    # ── Create output directories ──
    NPY_OUT.mkdir(parents=True, exist_ok=True)
    CSV_OUT.mkdir(parents=True, exist_ok=True)

    # ── Save similarity matrix to OneDrive if not already saved ──
    sim_matrix_path = NPY_OUT / "similarity_matrix.npy"
    if not sim_matrix_path.exists():
        np.save(sim_matrix_path, sim_matrix.astype(np.float16))
        print(f"  Saved similarity matrix → {sim_matrix_path}")

    # ── Stage 2: Distribution stats ──
    dist_stats, threshold = compute_distribution_stats(sim_matrix)

    # ── Stage 3: Module-level top-K ──
    top_k_df, mod_summary = module_top_k(sim_matrix, mod_idx, job_idx, k)
    top_k_df.to_csv(CSV_OUT / "top_k_matches_per_module.csv", index=False)
    print(f"  Saved → {CSV_OUT / 'top_k_matches_per_module.csv'}")

    # ── Stage 4: Module breadth & coverage ──
    breadth_coverage = module_breadth_and_coverage(
        sim_matrix, mod_idx, top_k_df, threshold
    )
    mod_summary = mod_summary.merge(breadth_coverage, on="module_code", how="left")
    mod_summary = mod_summary.sort_values("top_k_mean_sim", ascending=False)

    # Add rank columns to module summary
    mod_summary = _add_rank(mod_summary, "top_k_mean_sim", "rank_top_k_sim")
    mod_summary = _add_rank(mod_summary, "breadth_n_ssoc_groups", "rank_breadth")
    mod_summary = _add_rank(mod_summary, "coverage_pct", "rank_coverage")

    mod_summary.to_csv(CSV_OUT / "module_summary.csv", index=False)
    print(f"  Saved → {CSV_OUT / 'module_summary.csv'}")

    # ── Stage 5: SSOC-group alignment ──
    ssoc_alignment = ssoc_group_alignment(
        sim_matrix, mod_idx, job_idx, ssoc_level="ssoc_minor_title"
    )
    ssoc_alignment.to_csv(CSV_OUT / "ssoc_minor_alignment.csv", index=False)
    print(f"  Saved → {CSV_OUT / 'ssoc_minor_alignment.csv'}")

    # ── Stage 6: Degree-level analysis ──
    degree_summary, degree_topk = degree_level_analysis(
        sim_matrix, mod_idx, job_idx, deg_map, k, top_n, threshold
    )
    if degree_topk is not None:
        degree_topk.to_csv(CSV_OUT / "top_k_matches_per_degree.csv", index=False)
        print(f"  Saved → {CSV_OUT / 'top_k_matches_per_degree.csv'}")
    if degree_summary is not None:
        degree_summary.to_csv(CSV_OUT / "degree_summary.csv", index=False)
        print(f"  Saved → {CSV_OUT / 'degree_summary.csv'}")

    # ── Stage 7: Targeted SSOC alignment ──
    targeted_df = targeted_ssoc_alignment(
        sim_matrix, mod_idx, job_idx, deg_map, major_ssoc, k, top_n
    )
    if targeted_df is not None:
        targeted_df.to_csv(CSV_OUT / "targeted_ssoc_alignment.csv", index=False)
        print(f"  Saved → {CSV_OUT / 'targeted_ssoc_alignment.csv'}")

    # ── Stage 8: Summaries ──
    summaries = build_summaries(mod_summary, degree_summary)
    for name, df in summaries.items():
        path = CSV_OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")

    # ── Stage 9: Threshold stability ──
    stability = threshold_stability_check(
        sim_matrix, mod_idx, deg_map, dist_stats, top_n
    )

    # ── Stage 10: Metadata JSON + console summary ──
    print("\n" + "=" * 70)
    print("STAGE 10: Saving metadata & final summary")
    print("=" * 70)

    metadata = {
        "run_timestamp": start.isoformat(),
        "parameters": {
            "top_k": k,
            "degree_agg_top_n": top_n,
            "coverage_threshold": threshold,
            "breadth_ssoc_level": ANALYSIS_BREADTH_SSOC_LEVEL,
            "embedding_model": EMBEDDING_MODEL,
        },
        "data_shape": {
            "n_modules": int(sim_matrix.shape[0]),
            "n_jobs": int(sim_matrix.shape[1]),
            "n_degrees": int(deg_map["major"].nunique()) if deg_map is not None else 0,
        },
        "distribution_stats": dist_stats,
        "threshold_stability": stability,
    }

    with open(CSV_OUT / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = (datetime.now() - start).total_seconds()

    # ── Console summary ──
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 70}")

    print(f"\n  Similarity matrix (.npy): {NPY_OUT}")
    print(f"  All CSVs + metadata:     {CSV_OUT}")
    for f in sorted(CSV_OUT.glob("*")):
        print(f"    {f.name}")

    # Top modules
    print(f"\n  TOP 10 MODULES BY ALIGNMENT:")
    for _, r in mod_summary.head(10).iterrows():
        print(f"    #{int(r['rank_top_k_sim']):<4d} {r['module_code']:10s}  "
              f"top-K={r['top_k_mean_sim']:.4f}  "
              f"breadth={int(r['breadth_n_ssoc_groups'])}  "
              f"coverage={r['coverage_pct']:.1f}%  "
              f"{r['module_title'][:40]}")

    # Top degrees
    if degree_summary is not None:
        print(f"\n  TOP 10 DEGREES BY ALIGNMENT:")
        for _, r in degree_summary.head(10).iterrows():
            print(f"    #{int(r['rank_top_k_sim']):<4d} {r['major']:<35s}  "
                  f"top-K={r['top_k_mean_sim']:.4f}  "
                  f"breadth={int(r['breadth_n_ssoc_groups'])}  "
                  f"coverage={r['coverage_pct']:.1f}%  "
                  f"[{r['degree_faculty']}]")

    # Targeted SSOC
    if targeted_df is not None and len(targeted_df) > 0:
        print(f"\n  TARGETED SSOC ALIGNMENT (top 10):")
        for _, r in targeted_df.head(10).iterrows():
            print(f"    #{int(r['rank_targeted']):<4d} {r['major']:<35s}  "
                  f"targeted={r['targeted_top_k_mean']:.4f}  "
                  f"overall={r['overall_top_k_mean']:.4f}  "
                  f"ratio={r['targeted_vs_overall_ratio']:.3f}")

    # Faculty summary
    if "faculty_summary" in summaries:
        print(f"\n  FACULTY SUMMARY:")
        for _, r in summaries["faculty_summary"].iterrows():
            print(f"    {r['module_faculty']:<40s}  "
                  f"top-K={r['mean_top_k_sim']:.4f}  "
                  f"breadth={r['mean_breadth']:.1f}  "
                  f"coverage={r['mean_coverage_pct']:.1f}%  "
                  f"(n={int(r['n_modules'])})")


if __name__ == "__main__":
    main()
