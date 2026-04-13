"""
query_tool.py — CLI tool for exploring module/degree/job alignment
-------------------------------------------------------------------
Look up any module, degree, SSOC group, or individual job posting and
see its top-K matches, breadth, coverage, and SSOC breakdown. Useful
for spot-checking results and taking screenshots for presentations.

Usage (from repo root):
    python -m src.analysis.query_tool module CS3244
    python -m src.analysis.query_tool degree "Computer Science"
    python -m src.analysis.query_tool degree "Business Analytics" --k 20
    python -m src.analysis.query_tool ssoc "SOFTWARE AND APPLICATIONS DEVELOPERS AND ANALYSTS"
    python -m src.analysis.query_tool job "data analyst"
    python -m src.analysis.query_tool job 12345              # by job_id

All parameters sourced from config.py.
"""

import argparse
import sys
from collections import Counter
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
    JOBS_FILTERED,
)

EMB_DIR = EMBEDDINGS_DIR / "whole_text"
MODEL_TAG = EMBEDDING_MODEL.split("/")[-1]
REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_OUT = REPO_ROOT / "outputs" / "similarity_analysis_outputs"


def load_data():
    """
    Load similarity matrix, index files, and degree mapping.

    Applies the same salary + experience filter as similarity_analysis.py
    so that query results are consistent with the batch analysis outputs.
    """
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

    # ── Filter jobs to current JOBS_FILTERED population ──
    # Same logic as similarity_analysis.py — keeps query results consistent
    if JOBS_FILTERED.exists():
        filtered_jobs = pd.read_csv(JOBS_FILTERED, usecols=["job_id"])
        valid_ids = set(filtered_jobs["job_id"].values)
        mask = job_idx["job_id"].isin(valid_ids).values

        n_before = len(job_idx)
        job_idx = job_idx[mask].reset_index(drop=True)
        sim_matrix = sim_matrix[:, mask]
        n_after = len(job_idx)
        print(f"  Applied salary + experience filter: {n_before:,} → {n_after:,} jobs")
    else:
        print(f"  WARNING: {JOBS_FILTERED} not found — using all embedded jobs")

    # Degree mapping (optional)
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

    # Fallback: compute threshold from filtered matrix
    if threshold is None:
        flat = sim_matrix.ravel()
        threshold = float(flat.mean() + flat.std())

    return sim_matrix, mod_idx, job_idx, deg_map, threshold


# ──────────────────────────────────────────────
# MODULE QUERY
# ──────────────────────────────────────────────

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
        group_counts = Counter(g for g in top_k_ssoc if pd.notna(g))
        for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"    ({count:>2d} jobs) {group}")


# ──────────────────────────────────────────────
# DEGREE QUERY
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# SSOC QUERY
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# JOB QUERY
# ──────────────────────────────────────────────

def query_job(sim_matrix, mod_idx, job_idx, query, k,
              ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """
    Look up a job posting and find its top-K most aligned NUS modules.

    The query can be:
      - A numeric job_id (exact match)
      - A text string (fuzzy-matched against job titles)
    If multiple jobs match a text query, the user picks one.
    """
    # Determine if query is a job_id (numeric) or a title search
    try:
        job_id = int(query)
        is_id = True
    except ValueError:
        is_id = False

    if is_id:
        # Exact match on job_id
        mask = job_idx["job_id"] == job_id
        if not mask.any():
            print(f"ERROR: Job ID {job_id} not found in filtered job set.")
            return
        matches = job_idx[mask]
    else:
        # Fuzzy match on job title (case-insensitive substring)
        query_lower = query.lower()
        mask = job_idx["title"].str.lower().str.contains(query_lower, na=False)
        matches = job_idx[mask]

        if len(matches) == 0:
            print(f"ERROR: No jobs matching '{query}' found.")
            # Show a few example titles for guidance
            sample = job_idx["title"].dropna().sample(min(5, len(job_idx))).values
            print(f"  Example job titles in dataset:")
            for t in sample:
                print(f"    - {t}")
            return
        elif len(matches) > 10:
            # Too many matches — ask user to be more specific
            print(f"Found {len(matches)} jobs matching '{query}'. Showing first 10:")
            for _, row in matches.head(10).iterrows():
                ssoc = str(row.get("ssoc_minor_title", ""))[:35]
                print(f"  ID={row['job_id']:<8}  {str(row['title'])[:45]}  [{ssoc}]")
            print(f"\n  Use a job_id for an exact match, e.g.:")
            print(f"    python -m src.analysis.query_tool job {matches.iloc[0]['job_id']}")
            return
        elif len(matches) > 1:
            # A few matches — show them all, pick the first
            print(f"Found {len(matches)} jobs matching '{query}':")
            for _, row in matches.iterrows():
                ssoc = str(row.get("ssoc_minor_title", ""))[:35]
                print(f"  ID={row['job_id']:<8}  {str(row['title'])[:45]}  [{ssoc}]")
            print(f"\nShowing results for the first match. "
                  f"Use job_id for a specific one.")

    # Use the first match
    job_row = matches.iloc[0]
    job_pos = matches.index[0]  # position in the filtered job_idx

    print(f"\n{'=' * 70}")
    print(f"JOB: [{job_row['job_id']}] {job_row['title']}")
    print(f"{'=' * 70}")

    # Print available metadata
    for col_label, col_name in [
        ("SSOC Code",  "ssoc_code"),
        ("SSOC Major", "ssoc_major_title"),
        ("SSOC Minor", "ssoc_minor_title"),
        ("SSOC Unit",  "ssoc_unit_title"),
        ("Category",   "category"),
        ("Salary Avg", "salary_avg"),
    ]:
        val = job_row.get(col_name, None)
        if pd.notna(val) and str(val).strip():
            print(f"  {col_label + ':':<14s} {val}")

    # Get this job's similarity to all modules (= one column of the sim matrix)
    job_sims = sim_matrix[:, job_pos]  # shape: (n_modules,)

    # Top-K most similar modules
    capped_k = min(k, len(job_sims))
    top_k_idx = np.argpartition(job_sims, -capped_k)[-capped_k:]
    top_k_sims = job_sims[top_k_idx]
    order = np.argsort(top_k_sims)[::-1]
    top_k_idx = top_k_idx[order]
    top_k_sims = top_k_sims[order]

    print(f"\n  TOP-{capped_k} MOST RELEVANT NUS MODULES:")
    print(f"  {'Rank':<5s} {'Sim':>6s}  {'Code':<10s} {'Faculty':<30s} {'Title'}")
    print(f"  {'-'*5} {'-'*6}  {'-'*10} {'-'*30} {'-'*40}")
    for rank, (m_idx, score) in enumerate(zip(top_k_idx, top_k_sims), 1):
        mod = mod_idx.iloc[m_idx]
        print(f"  {rank:<5d} {score:>6.4f}  {mod['module code']:<10s} "
              f"{str(mod['faculty'])[:30]:<30s} {str(mod['title'])[:40]}")

    # Show which faculties are represented in the top-K
    top_faculties = [mod_idx.iloc[m]["faculty"] for m in top_k_idx]
    faculty_counts = Counter(top_faculties)
    print(f"\n  FACULTIES IN TOP-{capped_k}:")
    for fac, count in sorted(faculty_counts.items(), key=lambda x: -x[1]):
        print(f"    ({count:>2d} modules) {fac}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query module/degree/SSOC/job alignment results"
    )
    parser.add_argument(
        "query_type",
        choices=["module", "degree", "ssoc", "job"],
        help="Type of query: module, degree, ssoc, or job"
    )
    parser.add_argument(
        "query",
        help="Module code, degree name, SSOC group name, or job title/ID"
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
    elif args.query_type == "job":
        query_job(sim_matrix, mod_idx, job_idx, args.query, args.k)


if __name__ == "__main__":
    main()
