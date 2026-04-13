"""
combined_degree_priority.py — Degree-level alignment priority ranking
----------------------------------------------------------------------
Combines two complementary methodologies to identify NUS degree programmes
that are misaligned with their target career tracks:

  1. Semantic alignment  (friend's analysis) — do module descriptions match
     job descriptions holistically? Source: targeted_ssoc_alignment.csv
     (Top-K mean cosine similarity against SSOC-mapped jobs only)

  2. Skill coverage      (skill-space analysis) — do courses teach the
     explicit skills demanded by target-occupation job postings?
     Source: major.csv from breakdown_analysis (SSOC-aligned Soft-SCR)

Both metrics are restricted to each degree's mapped SSOC occupational
groups, so the comparison is fair across disciplines.

Formula
-------
Both metrics are percentile-ranked (0–100) across all 55 degrees, then
combined with equal (50-50) weight:

    alignment_score = 0.5 × sem_pctile + 0.5 × skill_pctile

Low score = misaligned on both dimensions = higher MOE review priority.

Outputs
-------
  combined_degree_priority.csv  — full ranked table
  fig_degree_priority_scatter.png — scatter plot of all 55 degrees

Usage
-----
    python -m src.analysis.combined_degree_priority
    python -m src.analysis.combined_degree_priority --output-dir path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.config import RESULTS_DIR
except ModuleNotFoundError:
    from config import RESULTS_DIR

SEM_CSV   = REPO_ROOT / "outputs" / "similarity_analysis_outputs" / "targeted_ssoc_alignment.csv"
SKILL_CSV_DEFAULT = RESULTS_DIR / "breakdown_analysis_skills_list" / "major.csv"

# ── chart style ────────────────────────────────────────────────────────────
FACULTY_COLOURS = {
    "School of Computing":                "#2563EB",
    "NUS Business School":                "#7C3AED",
    "College of Design and Engineering":  "#0891B2",
    "Faculty of Science":                 "#059669",
    "Faculty of Arts and Social Sciences":"#D97706",
    "Faculty of Law":                     "#DC2626",
    "Yong Loo Lin School of Medicine":    "#DB2777",
}
DEFAULT_COLOUR = "#64748B"
BG = "#F8FAFC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── data loading ───────────────────────────────────────────────────────────

def load_and_merge(sem_path: Path, skill_path: Path) -> pd.DataFrame:
    sem = pd.read_csv(sem_path)[
        ["major", "degree_faculty", "targeted_top_k_mean", "n_mapped_jobs"]
    ]
    skill = pd.read_csv(skill_path)[
        ["major", "soft_scr_all", "n_relevant_job_skills", "relevant_ssoc_minors"]
    ]
    df = sem.merge(skill, on="major", how="inner")

    missing = len(sem) + len(skill) - 2 * len(df)
    if missing > 0:
        print(f"  [warning] {missing} majors could not be matched — check name alignment")

    return df


# ── scoring ────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Percentile ranks (0–100, higher = better aligned)
    df["sem_pctile"]   = (df["targeted_top_k_mean"].rank(pct=True) * 100).round(1)
    df["skill_pctile"] = (df["soft_scr_all"].rank(pct=True) * 100).round(1)

    # Degree Alignment Score — 50-50 arithmetic mean of percentile ranks (0–100)
    df["alignment_score"] = (0.5 * df["sem_pctile"] + 0.5 * df["skill_pctile"]).round(1)

    # Priority rank for alignment score — rank 1 = most misaligned (lowest score)
    df["priority_rank"] = df["alignment_score"].rank(method="min", ascending=True).astype(int)

    # Degree-level PGI — analogous to job-category PGI
    #   job_demand_weight = n_mapped_jobs relative to the most demand-facing degree (0–1)
    #                       Normalised by max, not sum, to avoid double-counting: many
    #                       degrees share the same SSOC groups so summing inflates the
    #                       denominator and makes every degree look like a tiny fraction.
    #   misalignment_rate = 1 − (alignment_score / 100)  [0 = perfect, 1 = fully misaligned]
    #   degree_pgi        = demand × misalignment
    max_jobs = df["n_mapped_jobs"].max()
    df["job_demand_weight"] = (df["n_mapped_jobs"] / max_jobs).round(4)
    df["misalignment_rate"] = ((1 - df["alignment_score"] / 100)).round(4)
    df["degree_pgi"]        = (df["job_demand_weight"] * df["misalignment_rate"]).round(6)

    # PGI rank — rank 1 = highest degree_pgi = most urgent
    df["pgi_rank"] = df["degree_pgi"].rank(method="min", ascending=False).astype(int)

    return df.sort_values("priority_rank")


# ── visualisation ──────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))

    # Soft shading in bottom-left (purely visual — no hard threshold)
    ax.fill_between([0, 30], [0, 0], [30, 30], alpha=0.05, color="red", zorder=0)

    # Soft reference lines at 25th percentile
    ax.axhline(25, color="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(25, color="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)

    for _, row in df.iterrows():
        col = FACULTY_COLOURS.get(row["degree_faculty"], DEFAULT_COLOUR)
        ax.scatter(row["sem_pctile"], row["skill_pctile"],
                   color=col, s=65, zorder=3, alpha=0.9)

        label = row["major"] if len(row["major"]) <= 22 else row["major"][:20] + ".."
        weight = "bold" if row["alignment_score"] <= 25 else "normal"
        ax.annotate(
            label,
            (row["sem_pctile"], row["skill_pctile"]),
            fontsize=6.5, fontweight=weight, color="#1E293B",
            xytext=(4, 3), textcoords="offset points",
        )

    ax.set_xlabel(
        "Semantic Alignment Percentile\n"
        "(Targeted SSOC Top-K Mean Similarity — higher = better)",
        fontsize=10,
    )
    ax.set_ylabel(
        "Skill Coverage Percentile\n"
        "(SSOC-aligned Soft-SCR — higher = better)",
        fontsize=10,
    )
    ax.set_title(
        "Degree Curriculum Alignment — Skill-Space + Semantic\n"
        "Bottom-left: misaligned on both dimensions",
        fontsize=12, fontweight="bold",
    )

    # Quadrant annotations
    ax.text(0.02, 0.97, "Review Priority", transform=ax.transAxes,
            fontsize=8, color="#EF4444", fontweight="bold", va="top")
    ax.text(0.60, 0.03, "Skill Gap only", transform=ax.transAxes,
            fontsize=8, color="#D97706", va="bottom")
    ax.text(0.02, 0.45, "Framing\nIssue only", transform=ax.transAxes,
            fontsize=8, color="#D97706", va="bottom")
    ax.text(0.60, 0.97, "Well-Aligned", transform=ax.transAxes,
            fontsize=8, color="#059669", va="top")

    patches = [
        mpatches.Patch(color=c, label=f)
        for f, c in FACULTY_COLOURS.items()
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              framealpha=0.85, title="Faculty", title_fontsize=7)

    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Degree-level curriculum alignment priority ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sem-csv",   type=Path, default=SEM_CSV,
                   help="Path to targeted_ssoc_alignment.csv")
    p.add_argument("--skill-csv", type=Path, default=SKILL_CSV_DEFAULT,
                   help="Path to breakdown_analysis major.csv")
    p.add_argument("--output-dir", type=Path,
                   default=RESULTS_DIR / "combined_degree_priority",
                   help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"  Semantic : {args.sem_csv}")
    print(f"  Skill    : {args.skill_csv}")

    df = load_and_merge(args.sem_csv, args.skill_csv)
    df = compute_scores(df)

    # Save CSV
    out_cols = [
        "priority_rank", "pgi_rank", "major", "degree_faculty",
        "sem_pctile", "skill_pctile", "alignment_score",
        "job_demand_weight", "misalignment_rate", "degree_pgi",
        "targeted_top_k_mean", "soft_scr_all",
        "n_mapped_jobs", "n_relevant_job_skills", "relevant_ssoc_minors",
    ]
    csv_path = args.output_dir / "combined_degree_priority.csv"
    df[out_cols].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save scatter plot
    plot_scatter(df, args.output_dir / "fig_degree_priority_scatter.png")

    # Console report
    print("\n" + "=" * 75)
    print("DEGREE ALIGNMENT SCORE — TOP 15 (most misaligned)")
    print("  Score = 0.5 × semantic percentile + 0.5 × skill coverage percentile")
    print("  Lower score = more misaligned with target career track")
    print("=" * 75)
    print(f"{'Rank':<5} {'Major':<42} {'Sem%':>5} {'Skill%':>6} {'Score':>6}")
    print("-" * 75)
    for _, row in df.head(15).iterrows():
        print(f"{int(row['priority_rank']):<5} {row['major']:<42} "
              f"{row['sem_pctile']:>5.1f} {row['skill_pctile']:>6.1f} "
              f"{row['alignment_score']:>6.1f}")

    pgi_sorted = df.sort_values("pgi_rank")
    print("\n" + "=" * 75)
    print("DEGREE PGI — TOP 15 (highest impact misalignment)")
    print("  PGI = job_demand_weight × misalignment_rate")
    print("  Higher PGI = large job-market footprint AND poorly aligned")
    print("=" * 75)
    print(f"{'PGI Rank':<9} {'Major':<42} {'Demand%':>8} {'Gap%':>6} {'PGI':>8}")
    print("-" * 75)
    for _, row in pgi_sorted.head(15).iterrows():
        print(f"{int(row['pgi_rank']):<9} {row['major']:<42} "
              f"{row['job_demand_weight']*100:>7.2f}% "
              f"{row['misalignment_rate']*100:>5.1f}% "
              f"{row['degree_pgi']*1000:>7.3f}‰")

    print("\n" + "=" * 70)
    print(f"Full table saved to: {csv_path}")


if __name__ == "__main__":
    main()
