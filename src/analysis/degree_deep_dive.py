"""
degree_deep_dive.py — Deep-dive into a specific degree's coverage profile
---------------------------------------------------------------------------
Investigates *why* a degree has high or low job market coverage by:

  1. Score distribution — histogram of this degree's per-job similarity
     scores vs. the global distribution. Shows whether the degree has
     a long tail (generalist) or sharp peak (specialist).

  2. SSOC breakdown — which occupational groups contribute most to the
     degree's coverage, and which ones it misses entirely.

  3. Top contributing modules — which courses in the degree drive the
     highest alignment scores, and what kinds of jobs they match.

Useful for answering "why does Law have high coverage?" or "why does
Data Science have low coverage?" in the Discussion section.

Usage (from repo root):
    python -m src.analysis.degree_deep_dive "Law"
    python -m src.analysis.degree_deep_dive "Data Science and Analytics"
    python -m src.analysis.degree_deep_dive "Computer Science" --k 15

Outputs:
    Console report + figure saved to outputs/report_figures/deep_dive_{degree}.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
FIG_OUT = REPO_ROOT / "outputs" / "report_figures"

# ── Chart style ──
ACCENT   = "#2563EB"
ACCENT2  = "#7C3AED"
MUTED    = "#CBD5E1"
BAD      = "#EF4444"
BG       = "#F8FAFC"
TEXT_CLR = "#1E293B"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "text.color":        TEXT_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
})


def load_data():
    """Load sim matrix, indices, degree mapping (with salary filter applied)."""
    # Sim matrix
    sim_path = SIMILARITY_RESULTS_DIR / "similarity_matrix.npy"
    if not sim_path.exists():
        sim_path = CSV_OUT / "similarity_matrix.npy"
    if not sim_path.exists():
        print("ERROR: Similarity matrix not found. Run similarity_analysis.py first.")
        sys.exit(1)

    sim_matrix = np.load(sim_path).astype(np.float32)
    mod_idx = pd.read_csv(EMB_DIR / "module_index.csv")
    job_idx = pd.read_csv(EMB_DIR / "job_index.csv")

    # Apply salary + experience filter (same as similarity_analysis.py)
    if JOBS_FILTERED.exists():
        filtered_jobs = pd.read_csv(JOBS_FILTERED, usecols=["job_id"])
        valid_ids = set(filtered_jobs["job_id"].values)
        mask = job_idx["job_id"].isin(valid_ids).values
        n_before = len(job_idx)
        job_idx = job_idx[mask].reset_index(drop=True)
        sim_matrix = sim_matrix[:, mask]
        print(f"  Filter: {n_before:,} → {len(job_idx):,} jobs")

    # Degree mapping
    deg_map = pd.read_csv(DEGREE_MODULE_MAPPING) if DEGREE_MODULE_MAPPING.exists() else None

    # Threshold from metadata
    threshold = None
    meta_path = CSV_OUT / "analysis_metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            threshold = json.load(f)["parameters"].get("coverage_threshold")
    if threshold is None:
        flat = sim_matrix.ravel()
        threshold = float(flat.mean() + flat.std())

    return sim_matrix, mod_idx, job_idx, deg_map, threshold


def find_degree(deg_map, degree_name):
    """Fuzzy-match a degree name and return the exact major string."""
    all_majors = deg_map["major"].unique()

    # Try exact match first
    exact = [m for m in all_majors if m.lower() == degree_name.lower()]
    if exact:
        return exact[0]

    # Try substring match
    partial = [m for m in all_majors if degree_name.lower() in m.lower()]
    if len(partial) == 1:
        return partial[0]
    elif len(partial) > 1:
        print(f"Multiple matches for '{degree_name}':")
        for m in sorted(partial):
            print(f"  - {m}")
        sys.exit(1)
    else:
        print(f"ERROR: Degree '{degree_name}' not found.")
        print("Available degrees:")
        for m in sorted(all_majors):
            print(f"  - {m}")
        sys.exit(1)


def deep_dive(sim_matrix, mod_idx, job_idx, deg_map, major, k, top_n,
              threshold, ssoc_level=ANALYSIS_BREADTH_SSOC_LEVEL):
    """
    Run the full deep-dive analysis for a single degree.
    Prints a detailed console report and saves a multi-panel figure.
    """
    # ── Resolve degree modules ──
    mod_code_to_idx = {
        code: i for i, code in enumerate(mod_idx["module code"].values)
    }

    deg_modules = deg_map[deg_map["major"] == major]
    deg_faculty = deg_modules["degree_faculty"].iloc[0]
    codes = deg_modules["module_code"].unique()
    matched_indices = [mod_code_to_idx[c] for c in codes if c in mod_code_to_idx]

    if len(matched_indices) == 0:
        print(f"ERROR: No modules for {major} found in embedding index.")
        return

    sub_sim = sim_matrix[matched_indices, :]
    n_jobs = sim_matrix.shape[1]

    # ── Compute degree-level per-job scores ──
    n_use = min(top_n, len(matched_indices))
    if len(matched_indices) <= top_n:
        degree_scores = sub_sim.mean(axis=0)
    else:
        top_n_sims = np.partition(sub_sim, -n_use, axis=0)[-n_use:, :]
        degree_scores = top_n_sims.mean(axis=0)

    # Global distribution for comparison
    global_mean = float(sim_matrix.ravel().mean())
    global_std = float(sim_matrix.ravel().std())

    # Coverage
    above_mask = degree_scores >= threshold
    n_above = int(above_mask.sum())
    coverage_pct = 100.0 * n_above / n_jobs

    # ── Console report ──
    print(f"\n{'=' * 70}")
    print(f"DEEP DIVE: {major}")
    print(f"{'=' * 70}")
    print(f"  Faculty:        {deg_faculty}")
    print(f"  Modules:        {len(matched_indices)} matched / {len(codes)} total")
    print(f"  Aggregation:    mean of top-{n_use} modules per job")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Coverage:       {n_above:,} / {n_jobs:,} jobs ({coverage_pct:.1f}%)")

    deg_mean = float(degree_scores.mean())
    deg_std = float(degree_scores.std())
    print(f"\n  SCORE DISTRIBUTION:")
    print(f"    Degree mean:  {deg_mean:.4f} (global mean: {global_mean:.4f})")
    print(f"    Degree SD:    {deg_std:.4f} (global SD:   {global_std:.4f})")
    print(f"    Degree min:   {degree_scores.min():.4f}")
    print(f"    Degree max:   {degree_scores.max():.4f}")
    for p in [25, 50, 75, 90, 95]:
        print(f"    P{p}:          {np.percentile(degree_scores, p):.4f}")

    # ── SSOC group breakdown ──
    ssoc_labels = job_idx[ssoc_level].values
    unique_groups = sorted(job_idx[ssoc_level].dropna().unique())

    print(f"\n  SSOC GROUP BREAKDOWN (coverage contributors):")
    print(f"  {'SSOC Minor Group':<55s} {'Jobs':>5s} {'Above':>5s} {'Rate':>6s} {'Mean Sim':>8s}")
    print(f"  {'-'*55} {'-'*5} {'-'*5} {'-'*6} {'-'*8}")

    ssoc_rows = []
    for group in unique_groups:
        group_mask = ssoc_labels == group
        n_group = int(group_mask.sum())
        if n_group == 0:
            continue

        group_scores = degree_scores[group_mask]
        n_group_above = int((group_scores >= threshold).sum())
        group_rate = 100.0 * n_group_above / n_group
        group_mean = float(group_scores.mean())

        ssoc_rows.append({
            "ssoc_group": group,
            "n_jobs": n_group,
            "n_above": n_group_above,
            "coverage_rate": group_rate,
            "mean_sim": group_mean,
        })

    # Sort by number of jobs above threshold (biggest contributors first)
    ssoc_rows.sort(key=lambda x: -x["n_above"])

    for row in ssoc_rows[:20]:  # top 20 SSOC groups
        print(f"  {row['ssoc_group'][:55]:<55s} {row['n_jobs']:>5d} "
              f"{row['n_above']:>5d} {row['coverage_rate']:>5.1f}% "
              f"{row['mean_sim']:>8.4f}")

    # Groups with zero coverage
    zero_groups = [r for r in ssoc_rows if r["n_above"] == 0]
    if zero_groups:
        print(f"\n  SSOC GROUPS WITH ZERO COVERAGE ({len(zero_groups)}):")
        for row in zero_groups:
            print(f"    {row['ssoc_group'][:60]} ({row['n_jobs']} jobs, "
                  f"mean sim={row['mean_sim']:.4f})")

    # ── Top contributing modules ──
    print(f"\n  TOP CONTRIBUTING MODULES (by individual top-{k} alignment):")
    print(f"  {'Code':<10s} {'Type':<8s} {'Top-K':>6s}  {'Title'}")
    print(f"  {'-'*10} {'-'*8} {'-'*6}  {'-'*45}")

    mod_scores = []
    for local_i, global_i in enumerate(matched_indices):
        mod = mod_idx.iloc[global_i]
        mod_type_rows = deg_modules[deg_modules["module_code"] == mod["module code"]]
        mtype = mod_type_rows["module_type"].iloc[0] if len(mod_type_rows) > 0 else "?"
        row_sims = sim_matrix[global_i]
        tk = np.sort(row_sims)[::-1][:min(k, len(row_sims))]
        mod_scores.append({
            "code": mod["module code"],
            "title": mod["title"],
            "type": mtype,
            "top_k_mean": float(tk.mean()),
        })

    mod_scores.sort(key=lambda x: -x["top_k_mean"])
    for ms in mod_scores[:15]:
        print(f"  {ms['code']:<10s} {ms['type']:<8s} {ms['top_k_mean']:>6.4f}  "
              f"{ms['title'][:45]}")

    # Bottom 5 modules (weakest contributors)
    print(f"\n  WEAKEST MODULES (bottom 5):")
    for ms in mod_scores[-5:]:
        print(f"  {ms['code']:<10s} {ms['type']:<8s} {ms['top_k_mean']:>6.4f}  "
              f"{ms['title'][:45]}")

    # ── Generate multi-panel figure ──
    _plot_deep_dive(major, degree_scores, threshold, global_mean, global_std,
                    ssoc_rows, mod_scores, n_jobs)


def _plot_deep_dive(major, degree_scores, threshold, global_mean, global_std,
                    ssoc_rows, mod_scores, n_jobs):
    """
    Three-panel figure:
      (a) Score distribution histogram vs global normal curve
      (b) Top SSOC groups by coverage contribution (horizontal bars)
      (c) Top modules by alignment score (horizontal bars)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Safe filename
    safe_name = major.lower().replace(" ", "_").replace("&", "and")

    # ── Panel (a): Score distribution ──
    ax = axes[0]
    ax.hist(degree_scores, bins=50, color=ACCENT, alpha=0.6, density=True,
            edgecolor="white", linewidth=0.5, label=f"{major}")

    # Global normal overlay
    x = np.linspace(degree_scores.min(), degree_scores.max(), 200)
    y = (1 / (global_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - global_mean) / global_std) ** 2
    )
    ax.plot(x, y, color=MUTED, linewidth=2, linestyle="--", label="Global (approx)")

    ax.axvline(x=threshold, color=BAD, linewidth=1.5, linestyle="--", label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("Per-Job Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("(a) Score Distribution", fontweight="bold")
    ax.legend(fontsize=7)

    # ── Panel (b): Top SSOC groups by coverage contribution ──
    ax = axes[1]
    top_ssoc = sorted(ssoc_rows, key=lambda x: -x["n_above"])[:12]
    top_ssoc.reverse()  # so highest is at top of horizontal bar

    if top_ssoc:
        labels = [r["ssoc_group"][:35] for r in top_ssoc]
        values = [r["n_above"] for r in top_ssoc]
        colours = [ACCENT if v > 0 else MUTED for v in values]

        ax.barh(range(len(top_ssoc)), values, color=colours, height=0.6)
        ax.set_yticks(range(len(top_ssoc)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("Jobs Above Threshold")
    ax.set_title("(b) Top SSOC Groups by Coverage", fontweight="bold")

    # ── Panel (c): Top modules by alignment ──
    ax = axes[2]
    top_mods = mod_scores[:12]
    top_mods_rev = list(reversed(top_mods))

    if top_mods_rev:
        labels = [f"{m['code']} [{m['type'][:4]}]" for m in top_mods_rev]
        values = [m["top_k_mean"] for m in top_mods_rev]

        ax.barh(range(len(top_mods_rev)), values, color=ACCENT2, height=0.6)
        ax.set_yticks(range(len(top_mods_rev)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Top-K Mean Similarity")
    ax.set_title("(c) Top Contributing Modules", fontweight="bold")

    fig.suptitle(f"Deep Dive: {major}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out_path = FIG_OUT / f"deep_dive_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deep-dive analysis for a specific degree's coverage profile"
    )
    parser.add_argument(
        "degree",
        help="Degree name (e.g. 'Law', 'Data Science and Analytics')"
    )
    parser.add_argument(
        "--k", type=int, default=ANALYSIS_TOP_K,
        help=f"Top-K for module-level alignment (default: {ANALYSIS_TOP_K})"
    )
    parser.add_argument(
        "--top-n", type=int, default=ANALYSIS_DEGREE_AGG_TOP_N,
        help=f"Top-N modules per job for degree aggregation "
             f"(default: {ANALYSIS_DEGREE_AGG_TOP_N})"
    )
    args = parser.parse_args()

    print("Loading data...")
    sim_matrix, mod_idx, job_idx, deg_map, threshold = load_data()

    if deg_map is None:
        print("ERROR: No degree mapping found.")
        sys.exit(1)

    major = find_degree(deg_map, args.degree)
    print(f"  Matched degree: {major}")

    deep_dive(sim_matrix, mod_idx, job_idx, deg_map, major,
              args.k, args.top_n, threshold)


if __name__ == "__main__":
    main()
