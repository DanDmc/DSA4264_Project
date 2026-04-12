"""
query_tool.py — CLI tool for exploring module/degree job alignment
-------------------------------------------------------------------
Look up any module or degree and see its top-K job matches, breadth,
coverage, and SSOC breakdown. Useful for spot-checking results and
taking screenshots for presentations.

Usage (from repo root):
    python -m src.analysis.query_tool module CS3244
    python -m src.analysis.query_tool degree "Computer Science"
    python -m src.analysis.query_tool degree "Business Analytics" --k 20
    python -m src.analysis.query_tool ssoc "SOFTWARE AND APPLICATIONS DEVELOPERS AND ANALYSTS"

All parameters sourced from config.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project config ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    DEGREE_MODULE_MAPPING,
    SIMILARITY_RESULTS_DIR,
    ANALYSIS_TOP_K,
    ANALYSIS_DEGREE_AGG_TOP_N,
    ANALYSIS_BREADTH_SSOC_LEVEL,
)

EMB_DIR = EMBEDDINGS_DIR / "whole_text"
MODEL_TAG = EMBEDDING_MODEL.split("/")[-1]
REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_OUT = REPO_ROOT / "outputs" / "similarity_analysis_outputs"


def load_data():
    """Load similarity matrix and index files."""
    # Try OneDrive location first, then old repo fallback
    sim_path = SIMILARITY_RESULTS_DIR / "similarity_matrix.npy"
    if not sim_path.exists():
        sim_path = CSV_OUT / "similarity_matrix.npy"
    if not sim_path.exists():
        print("ERROR: No precomputed similarity matrix found.")
        print("Run the analysis pipeline first: python -m src.analysis.similarity_analysis")
        sys.exit(1)

    sim_matrix = np.load(sim_path).astype(np.float32)
    mod_idx = pd.read_csv(EMB_DIR / "module_index.csv")
    job_idx = pd.read_csv(EMB_DIR / "job_index.csv")

    deg_map = None
    if DEGREE_MODULE_MAPPING.exists():
        deg_map = pd.read_csv(DEGREE_MODULE_MAPPING)

    # Load metadata for threshold
    metadata_path = CSV_OUT / "analysis_metadata.json"
    threshold = None
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            meta = json.load(f)
        threshold = meta["parameters"].get("coverage_threshold")

    if threshold is None:
        flat = sim_matrix.ravel()
        threshold = float(flat.mean() + flat.std())

    return sim_matrix, mod_idx, job_idx, deg_map, threshold


def query_module(sim_matrix, mod_idx, job_idx, module_code, k, threshold,
                 ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """Look up a single module's alignment profile."""
    # Find the module
    mask = mod_idx["module code"].str.upper() == module_code.upper()
    if not mask.any():
        print(f"ERROR: Module '{module_code}' not found in embedding index.")
        # Suggest close matches
        all_codes = mod_idx["module code"].values
        prefix = module_code[:2].upper()
        suggestions = [c for c in all_codes if c.upper().startswith(prefix)][:10]
        if suggestions:
            print(f"  Did you mean one of: {', '.join(suggestions)}")
        return

    idx = mod_idx.index[mask][0]
    mod = mod_idx.iloc[idx]
    sims = sim_matrix[idx]

    print(f"\n{'=' * 70}")
    print(f"MODULE: [{mod['module code']}] {mod['title']}")
    print(f"{'=' * 70}")
    print(f"  Faculty:    {mod['faculty']}")
    print(f"  Department: {mod.get('department', 'N/A')}")

    # Top-K matches
    capped_k = min(k, len(sims))
    top_k_idx = np.argpartition(sims, -capped_k)[-capped_k:]
    top_k_sims = sims[top_k_idx]
    order = np.argsort(top_k_sims)[::-1]
    top_k_idx = top_k_idx[order]
    top_k_sims = top_k_sims[order]

    # Stats
    top_k_mean = float(top_k_sims.mean())
    n_above = int((sims >= threshold).sum())
    coverage_pct = 100.0 * n_above / len(sims)

    # Breadth
    ssoc_labels = job_idx[ssoc_level].values
    top_k_ssoc = ssoc_labels[top_k_idx]
    ssoc_groups = sorted(set(g for g in top_k_ssoc if pd.notna(g)))
    breadth = len(ssoc_groups)

    print(f"\n  Top-{capped_k} mean similarity: {top_k_mean:.4f}")
    print(f"  Breadth ({ssoc_level}): {breadth} distinct groups")
    print(f"  Coverage: {n_above} jobs ({coverage_pct:.1f}%) above threshold {threshold:.4f}")

    print(f"\n  TOP-{capped_k} JOB MATCHES:")
    print(f"  {'Rank':<5s} {'Sim':>6s}  {'Job Title':<45s} {'SSOC Minor Group'}")
    print(f"  {'-'*5} {'-'*6}  {'-'*45} {'-'*40}")
    for rank, (j_idx, score) in enumerate(zip(top_k_idx, top_k_sims), 1):
        job = job_idx.iloc[j_idx]
        print(f"  {rank:<5d} {score:>6.4f}  {str(job['title'])[:45]:<45s} "
              f"{str(job.get('ssoc_minor_title', 'N/A'))[:40]}")

    if ssoc_groups:
        print(f"\n  SSOC GROUPS IN TOP-{capped_k}:")
        from collections import Counter
        group_counts = Counter(g for g in top_k_ssoc if pd.notna(g))
        for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"    ({count:>2d} jobs) {group}")


def query_degree(sim_matrix, mod_idx, job_idx, deg_map, degree_name, k, top_n,
                 threshold, ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """Look up a degree's alignment profile."""
    if deg_map is None:
        print("ERROR: No degree mapping available.")
        return

    # Fuzzy match on degree name
    all_majors = deg_map["major"].unique()
    exact = [m for m in all_majors if m.lower() == degree_name.lower()]
    if exact:
        major = exact[0]
    else:
        partial = [m for m in all_majors if degree_name.lower() in m.lower()]
        if len(partial) == 1:
            major = partial[0]
        elif len(partial) > 1:
            print(f"Multiple matches for '{degree_name}':")
            for m in sorted(partial):
                print(f"  - {m}")
            print("Please be more specific.")
            return
        else:
            print(f"ERROR: Degree '{degree_name}' not found.")
            print(f"Available degrees:")
            for m in sorted(all_majors):
                print(f"  - {m}")
            return

    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }

    deg_modules = deg_map[deg_map["major"] == major]
    deg_faculty = deg_modules["degree_faculty"].iloc[0]
    codes = deg_modules["module_code"].unique()
    matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]
    unmatched = [c for c in codes if c not in mod_code_to_idx]

    print(f"\n{'=' * 70}")
    print(f"DEGREE: {major}")
    print(f"{'=' * 70}")
    print(f"  Faculty:           {deg_faculty}")
    print(f"  Total modules:     {len(codes)}")
    print(f"  Matched in index:  {len(matched_indices)}")
    if unmatched:
        print(f"  Unmatched:         {len(unmatched)} (not in embedding index)")

    if len(matched_indices) == 0:
        print("  No modules found in embedding index — cannot compute alignment.")
        return

    sub_sim = sim_matrix[matched_indices, :]
    n_jobs = sim_matrix.shape[1]

    # Per-job score: mean of top-N modules
    n_use = min(top_n, len(matched_indices))
    if len(matched_indices) <= top_n:
        degree_job_scores = sub_sim.mean(axis=0)
    else:
        top_n_sims = np.partition(sub_sim, -n_use, axis=0)[-n_use:, :]
        degree_job_scores = top_n_sims.mean(axis=0)

    # Top-K jobs
    capped_k = min(k, len(degree_job_scores))
    top_k_idx = np.argpartition(degree_job_scores, -capped_k)[-capped_k:]
    top_k_sims = degree_job_scores[top_k_idx]
    order = np.argsort(top_k_sims)[::-1]
    top_k_idx = top_k_idx[order]
    top_k_sims = top_k_sims[order]

    top_k_mean = float(top_k_sims.mean())

    # Coverage
    n_above = int((degree_job_scores >= threshold).sum())
    coverage_pct = 100.0 * n_above / n_jobs

    # Breadth
    ssoc_labels = job_idx[ssoc_level].values
    top_k_ssoc = ssoc_labels[top_k_idx]
    ssoc_groups = sorted(set(g for g in top_k_ssoc if pd.notna(g)))
    breadth = len(ssoc_groups)

    print(f"\n  Aggregation: mean of top-{n_use} modules per job")
    print(f"  Top-{capped_k} mean similarity: {top_k_mean:.4f}")
    print(f"  Breadth ({ssoc_level}): {breadth} distinct groups")
    print(f"  Coverage: {n_above} jobs ({coverage_pct:.1f}%) above threshold {threshold:.4f}")

    print(f"\n  TOP-{capped_k} JOB MATCHES:")
    print(f"  {'Rank':<5s} {'Sim':>6s}  {'Job Title':<40s} {'Best Module':<10s} {'SSOC Minor Group'}")
    print(f"  {'-'*5} {'-'*6}  {'-'*40} {'-'*10} {'-'*35}")
    for rank, (j_idx, score) in enumerate(zip(top_k_idx, top_k_sims), 1):
        job = job_idx.iloc[j_idx]
        best_mod_local = sub_sim[:, j_idx].argmax()
        best_mod_code = mod_idx.iloc[matched_indices[best_mod_local]]["module code"]
        print(f"  {rank:<5d} {score:>6.4f}  {str(job['title'])[:40]:<40s} "
              f"{best_mod_code:<10s} {str(job.get('ssoc_minor_title', 'N/A'))[:35]}")

    if ssoc_groups:
        print(f"\n  SSOC GROUPS IN TOP-{capped_k}:")
        from collections import Counter
        group_counts = Counter(g for g in top_k_ssoc if pd.notna(g))
        for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"    ({count:>2d} jobs) {group}")

    # Show contributing modules (ranked by their individual top-K mean)
    print(f"\n  TOP CONTRIBUTING MODULES (by individual alignment):")
    mod_scores = []
    for local_i, global_i in enumerate(matched_indices):
        mod = mod_idx.iloc[global_i]
        mod_type = deg_modules[deg_modules["module_code"] == mod["module code"]]
        mtype = mod_type["module_type"].iloc[0] if len(mod_type) > 0 else "?"
        row_sims = sim_matrix[global_i]
        tk = np.sort(row_sims)[::-1][:min(k, len(row_sims))]
        mod_scores.append((mod["module code"], mod["title"], mtype, float(tk.mean())))
    mod_scores.sort(key=lambda x: -x[3])
    for code, title, mtype, score in mod_scores[:15]:
        print(f"    {code:<10s} [{mtype:<8s}] top-K={score:.4f}  {title[:45]}")


def query_ssoc(sim_matrix, mod_idx, job_idx, ssoc_name, k,
               ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """Look up which modules are most aligned with an SSOC group."""
    ssoc_values = job_idx[ssoc_level].values
    unique_groups = sorted(job_idx[ssoc_level].dropna().unique())

    # Fuzzy match
    exact = [g for g in unique_groups if g.lower() == ssoc_name.lower()]
    if exact:
        group = exact[0]
    else:
        partial = [g for g in unique_groups if ssoc_name.lower() in g.lower()]
        if len(partial) == 1:
            group = partial[0]
        elif len(partial) > 1:
            print(f"Multiple matches for '{ssoc_name}':")
            for g in partial:
                print(f"  - {g}")
            return
        else:
            print(f"ERROR: SSOC group '{ssoc_name}' not found.")
            print(f"Available groups:")
            for g in unique_groups:
                print(f"  - {g}")
            return

    mask = ssoc_values == group
    n_jobs = int(mask.sum())

    print(f"\n{'=' * 70}")
    print(f"SSOC GROUP: {group}")
    print(f"{'=' * 70}")
    print(f"  Number of jobs: {n_jobs}")

    # Mean similarity of each module to jobs in this group
    group_sims = sim_matrix[:, mask].mean(axis=1)

    # Top modules
    top_indices = np.argsort(group_sims)[::-1][:min(k, len(group_sims))]

    print(f"\n  TOP-{len(top_indices)} MOST ALIGNED MODULES:")
    print(f"  {'Rank':<5s} {'Sim':>6s}  {'Code':<10s} {'Faculty':<30s} {'Title'}")
    print(f"  {'-'*5} {'-'*6}  {'-'*10} {'-'*30} {'-'*40}")
    for rank, m_idx in enumerate(top_indices, 1):
        mod = mod_idx.iloc[m_idx]
        print(f"  {rank:<5d} {group_sims[m_idx]:>6.4f}  {mod['module code']:<10s} "
              f"{str(mod['faculty'])[:30]:<30s} {str(mod['title'])[:40]}")


def main():
    parser = argparse.ArgumentParser(
        description="Query module/degree/SSOC alignment results"
    )
    parser.add_argument(
        "query_type",
        choices=["module", "degree", "ssoc"],
        help="Type of query"
    )
    parser.add_argument(
        "query",
        help="Module code, degree name, or SSOC group name"
    )
    parser.add_argument(
        "--k", type=int, default=ANALYSIS_TOP_K,
        help=f"Number of top matches to show (default: {ANALYSIS_TOP_K})"
    )
    parser.add_argument(
        "--top-n", type=int, default=ANALYSIS_DEGREE_AGG_TOP_N,
        help=f"Top-N modules averaged per job for degree queries "
             f"(default: {ANALYSIS_DEGREE_AGG_TOP_N})"
    )
    args = parser.parse_args()

    print("Loading data...")
    sim_matrix, mod_idx, job_idx, deg_map, threshold = load_data()
    print(f"  Modules: {len(mod_idx)}, Jobs: {len(job_idx)}, "
          f"Threshold: {threshold:.4f}")

    if args.query_type == "module":
        query_module(sim_matrix, mod_idx, job_idx, args.query, args.k, threshold)
    elif args.query_type == "degree":
        query_degree(sim_matrix, mod_idx, job_idx, deg_map, args.query,
                     args.k, args.top_n, threshold)
    elif args.query_type == "ssoc":
        query_ssoc(sim_matrix, mod_idx, job_idx, args.query, args.k)


if __name__ == "__main__":
    main()
