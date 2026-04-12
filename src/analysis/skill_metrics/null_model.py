"""
Null Model for Skill Coverage Rate — Step 2 of the coverage pipeline.

What this script does
---------------------
Tests whether NUS's *pattern* of skill coverage across job categories is
non-random — i.e. does NUS strategically cover the right skills in the
right job categories, or could the observed per-category SCR be explained
by chance?

Why not "does NUS cover more job skills than random?"
------------------------------------------------------
NUS has 31,949 unique canonicalized course skill terms, while the total
combined vocabulary (course ∪ job skills) is only 38,722.  A random draw
of 31,949 skills from this vocabulary would cover ~82 % of job demands by
chance alone — making that comparison meaningless.

The right null model
--------------------
NUS covers exactly 1,811 of the 7,893 unique job skills (22.9 % micro SCR).
We hold this coverage *count* fixed and ask: if NUS happened to cover 1,811
job skills at random (with no deliberate curriculum alignment), what would
per-category SCR look like?

This is a permutation test on coverage labels:
  - Take all 7,893 unique job skills.
  - In each iteration, randomly mark 1,811 of them as "covered".
  - Compute per-category SCR under this random labelling.
  - Repeat B = 1,000 times to build the null distribution.
  - Compare NUS's actual per-category SCR to this null.

Null hypothesis
---------------
    H₀: The 1,811 job skills covered by NUS are a random subset of all
        job-demanded skills — i.e. coverage is not concentrated in any
        particular job category.

    H₁: Coverage is non-randomly distributed — NUS covers the right skills
        in the right categories more than chance would predict.

Interpretation
--------------
  - A category with actual SCR *above* the null 97.5th percentile is
    significantly better covered than chance (p < 0.05).
  - A category *below* the null 2.5th percentile is significantly worse
    covered than chance — a genuine curriculum gap signal.
  - Categories within the null CI are covered at roughly the level that
    random coverage of the same size would produce.

Outputs
-------
results/null_model/
  ├── per_category_null.csv        — null stats + actual SCR per category
  ├── summary_null.json            — headline null model results
  ├── fig_actual_vs_null.png       — actual SCR vs null CI per category
  └── fig_null_distribution.png   — histogram of macro null SCR distribution

Usage
-----
    python -m src.analysis.skill_metrics.null_model
    python -m src.analysis.skill_metrics.null_model --iterations 2000
    python -m src.analysis.skill_metrics.null_model --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR

from src.analysis.skill_metrics.baseline_scr import load_course_skills, load_job_skills_auto

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_JOB_SKILLS    = JOBS_PROCESSED_DIR    / "job_skill_pair_skillner.csv"
DEFAULT_COURSE_SKILLS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_OUTPUT_DIR    = RESULTS_DIR / "null_model"
DEFAULT_ITERATIONS    = 1_000
DEFAULT_SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Null model core
# ─────────────────────────────────────────────────────────────────────────────

def build_null_distribution(
    job_skills:       pd.DataFrame,
    course_skill_set: set[str],
    n_iterations:     int,
    rng:              np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Permutation-test null distribution of SCR per job category.

    For each iteration we randomly select K job skills as "covered"
    (where K = actual number of job skills covered by NUS courses), then
    compute per-category SCR for this random selection.

    This holds the *size* of coverage fixed and tests only whether
    the *pattern* of coverage across categories is non-random.

    Parameters
    ----------
    job_skills : DataFrame
        Output of ``load_job_skills()``.
    course_skill_set : set[str]
        Actual NUS course skills (canonicalized).
    n_iterations : int
        Number of random permutations.
    rng : numpy Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    dict mapping category → 1-D array of length n_iterations (null SCR values)
    """
    # All unique job skills (the pool we permute over)
    all_job_skills = np.array(sorted(job_skills["skill_canon"].unique()))
    n_job_skills   = len(all_job_skills)

    # Actual covered skills and coverage count
    actually_covered = np.array(
        sorted(set(all_job_skills) & course_skill_set)
    )
    k_covered = len(actually_covered)   # fix this count for the null

    # Per-category job skill sets (pre-built once for efficiency)
    cat_job_sets: dict[str, set[str]] = {
        cat: set(grp["skill_canon"].unique())
        for cat, grp in job_skills.groupby("category")
    }

    # Allocate result arrays
    null_scr: dict[str, np.ndarray] = {
        cat: np.zeros(n_iterations) for cat in cat_job_sets
    }

    print(f"  Total unique job skills : {n_job_skills:,}")
    print(f"  Actually covered by NUS : {k_covered:,}  "
          f"({k_covered / n_job_skills:.1%} micro SCR)")
    print(f"  Running {n_iterations:,} permutations …")

    for i in range(n_iterations):
        # Randomly choose k_covered job skills as "covered"
        random_covered = set(
            all_job_skills[rng.choice(n_job_skills, size=k_covered, replace=False)]
        )

        for cat, job_set in cat_job_sets.items():
            n_cov = len(job_set & random_covered)
            null_scr[cat][i] = n_cov / len(job_set) if job_set else 0.0

        if (i + 1) % 200 == 0:
            print(f"    {i + 1:,} / {n_iterations:,} done")

    return null_scr


def compute_null_stats(
    null_scr:   dict[str, np.ndarray],
    actual_scr: dict[str, float],
) -> pd.DataFrame:
    """
    Compute per-category null statistics and compare to actual SCR.

    Returns a DataFrame with one row per category:
        category, actual_scr,
        null_mean, null_sd, null_ci_low, null_ci_high,
        z_score,       ← (actual − null_mean) / null_sd
        p_value,       ← two-sided: P(|null − null_mean| ≥ |actual − null_mean|)
        direction      ← "above" | "below" | "within" null CI
    """
    rows: list[dict[str, Any]] = []

    for cat, null_vals in null_scr.items():
        actual  = actual_scr.get(cat, 0.0)
        mean_   = float(null_vals.mean())
        sd_     = float(null_vals.std(ddof=1))
        ci_low  = float(np.percentile(null_vals, 2.5))
        ci_high = float(np.percentile(null_vals, 97.5))

        z = (actual - mean_) / sd_ if sd_ > 0 else float("nan")

        # Two-sided p-value
        p_val = float(2 * min(
            (null_vals >= actual).mean(),
            (null_vals <= actual).mean(),
        ))
        p_val = min(p_val, 1.0)

        if actual > ci_high:
            direction = "above"    # better covered than chance
        elif actual < ci_low:
            direction = "below"    # worse covered than chance (gap signal)
        else:
            direction = "within"

        rows.append({
            "category":     cat,
            "actual_scr":   round(actual,   4),
            "null_mean":    round(mean_,    4),
            "null_sd":      round(sd_,      4),
            "null_ci_low":  round(ci_low,   4),
            "null_ci_high": round(ci_high,  4),
            "z_score":      round(z,        2),
            "p_value":      round(p_val,    4),
            "direction":    direction,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_DIR_COLOURS = {
    "above":  "#27ae60",   # green  — significantly better than chance
    "below":  "#e74c3c",   # red    — significantly worse than chance
    "within": "#7f8c8d",   # grey   — within null CI
}


def plot_actual_vs_null(df: pd.DataFrame, output_path: Path) -> None:
    """
    Horizontal chart: actual SCR (dot) vs null 95 % CI (error bar) per category.

    Categories sorted by z-score (most above null at top).
    Colour encodes whether the actual SCR is above, within, or below the null CI.

    Key insight for MOE: green categories are ones where NUS specifically
    concentrates coverage beyond what random chance would predict.
    Red categories are genuine relative gaps — NUS covers these *less* than
    random chance would suggest.
    """
    df_plot = df.sort_values("z_score", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(8, len(df_plot) * 0.38)))

    for _, row in df_plot.iterrows():
        y     = row["category"]
        color = _DIR_COLOURS[row["direction"]]

        # Null 95 % CI as a horizontal span
        ax.barh(
            y, row["null_ci_high"] - row["null_ci_low"],
            left=row["null_ci_low"], height=0.55,
            color="#ecf0f1", edgecolor="#bdc3c7", linewidth=0.8,
            zorder=2,
        )
        # Null mean tick
        ax.vlines(row["null_mean"], ymin=df_plot["category"].tolist().index(y) - 0.3,
                  ymax=df_plot["category"].tolist().index(y) + 0.3,
                  colors="#95a5a6", linewidth=1.5, zorder=3)
        # Actual SCR dot
        ax.scatter(row["actual_scr"], y, marker="D", s=55,
                   color=color, zorder=4)
        # Percentage label
        ax.text(
            max(row["actual_scr"], row["null_ci_high"]) + 0.006,
            y, f"{row['actual_scr']:.1%}",
            va="center", ha="left", fontsize=7.5, color=color,
        )

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Skill Coverage Rate (SCR)", fontsize=10)
    ax.set_title(
        "NUS Curriculum Coverage vs Null Model — Permutation Test\n"
        "Null: same number of job skills covered (1 811) distributed randomly "
        "across categories  (1 000 permutations)",
        fontsize=11, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    patches = [
        mpatches.Patch(color=_DIR_COLOURS["above"],  label="Above null CI — NUS over-concentrates coverage here"),
        mpatches.Patch(color=_DIR_COLOURS["within"], label="Within null CI — coverage consistent with chance"),
        mpatches.Patch(color=_DIR_COLOURS["below"],  label="Below null CI — NUS under-covers relative to chance"),
        mpatches.Patch(color="#ecf0f1", ec="#bdc3c7", label="Null 95 % CI"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_macro_null_distribution(
    macro_null:       np.ndarray,
    actual_macro_scr: float,
    output_path:      Path,
) -> None:
    """
    Histogram of the macro-average null SCR distribution vs actual macro SCR.

    Under the permutation null (randomly distribute the same number of
    covered skills), the expected macro SCR equals the micro SCR (~22.9 %)
    because coverage is spread uniformly across categories.  If the actual
    macro SCR differs, coverage is non-uniformly distributed.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(macro_null, bins=50, color="#3498db", alpha=0.75,
            edgecolor="white", label="Null macro SCR (1 000 permutations)")
    ax.axvline(actual_macro_scr, color="#e74c3c", linewidth=2.2,
               linestyle="--", label=f"Actual macro SCR: {actual_macro_scr:.1%}")
    ax.axvline(np.percentile(macro_null, 2.5),  color="#7f8c8d",
               linewidth=1.2, linestyle=":", label="Null 95 % CI bounds")
    ax.axvline(np.percentile(macro_null, 97.5), color="#7f8c8d",
               linewidth=1.2, linestyle=":")

    p_val = float(2 * min(
        (macro_null >= actual_macro_scr).mean(),
        (macro_null <= actual_macro_scr).mean(),
    ))
    z = (actual_macro_scr - macro_null.mean()) / macro_null.std()

    ax.set_xlabel("Macro-average SCR across all job categories", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(
        "Permutation Null: Macro SCR Distribution\n"
        f"p = {p_val:.4f}  |  z = {z:+.2f}  |  "
        f"Null mean = {macro_null.mean():.1%}",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Permutation null model for SCR coverage pattern.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-skills",    type=Path, default=None)
    p.add_argument("--job-source",    type=str,  default="skills_list",
                   choices=["skillner", "skills_list"])
    p.add_argument("--course-skills", type=Path, default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--output",        type=Path, default=None)
    p.add_argument("--iterations",    type=int,  default=DEFAULT_ITERATIONS)
    p.add_argument("--seed",          type=int,  default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.output is None:
        args.output = RESULTS_DIR / f"null_model_{args.job_source}"
    out  = args.output
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading data  [job-source: {args.job_source}] …")
    courses = load_course_skills(args.course_skills)
    jobs    = load_job_skills_auto(args.job_skills, args.job_source)

    course_skill_set = set(courses["skill_canon"].unique())

    # ── Actual SCR per category ────────────────────────────────────────────────
    actual_scr_per_cat: dict[str, float] = {}
    for cat, grp in jobs.groupby("category"):
        job_set = set(grp["skill_canon"].unique())
        actual_scr_per_cat[cat] = len(job_set & course_skill_set) / len(job_set)

    actual_macro_scr = float(np.mean(list(actual_scr_per_cat.values())))

    # ── Run null model ─────────────────────────────────────────────────────────
    print("\nRunning permutation null model …")
    null_scr = build_null_distribution(jobs, course_skill_set, args.iterations, rng)

    macro_null = np.array([
        np.mean([null_scr[cat][i] for cat in null_scr])
        for i in range(args.iterations)
    ])

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\nComputing null statistics …")
    df_stats = compute_null_stats(null_scr, actual_scr_per_cat)

    n_above  = int((df_stats["direction"] == "above").sum())
    n_below  = int((df_stats["direction"] == "below").sum())
    n_within = int((df_stats["direction"] == "within").sum())

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\nSaving outputs …")
    df_stats.to_csv(out / "per_category_null.csv", index=False)
    print(f"  Saved: per_category_null.csv")

    macro_p = float(2 * min(
        (macro_null >= actual_macro_scr).mean(),
        (macro_null <= actual_macro_scr).mean(),
    ))
    macro_z = float((actual_macro_scr - macro_null.mean()) / macro_null.std())

    summary = {
        "null_type":           "permutation — fixed coverage count",
        "iterations":          args.iterations,
        "seed":                args.seed,
        "k_covered":           int(len(set(jobs["skill_canon"].unique()) & course_skill_set)),
        "n_unique_job_skills": int(jobs["skill_canon"].nunique()),
        "actual_macro_scr":    round(actual_macro_scr,           4),
        "null_macro_mean":     round(float(macro_null.mean()),   4),
        "null_macro_sd":       round(float(macro_null.std()),    4),
        "null_macro_ci_low":   round(float(np.percentile(macro_null, 2.5)),  4),
        "null_macro_ci_high":  round(float(np.percentile(macro_null, 97.5)), 4),
        "macro_z_score":       round(macro_z, 2),
        "macro_p_value":       round(macro_p, 4),
        "n_categories_above_null":  n_above,
        "n_categories_within_null": n_within,
        "n_categories_below_null":  n_below,
    }
    with (out / "summary_null.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary_null.json")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_actual_vs_null(df_stats, out / "fig_actual_vs_null.png")
    plot_macro_null_distribution(macro_null, actual_macro_scr,
                                 out / "fig_null_distribution.png")

    # ── Headline ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  NULL MODEL — HEADLINE RESULTS")
    print("=" * 62)
    print(f"  Actual macro SCR         : {actual_macro_scr:.1%}")
    print(f"  Null macro SCR (mean)    : {macro_null.mean():.1%}")
    print(f"  Null 95 % CI             : "
          f"[{np.percentile(macro_null, 2.5):.1%}, "
          f"{np.percentile(macro_null, 97.5):.1%}]")
    print(f"  z-score                  : {macro_z:+.2f}")
    print(f"  p-value (two-sided)      : {macro_p:.4f}")
    print()
    print(f"  Categories above null CI : {n_above}  (NUS over-concentrates here)")
    print(f"  Categories within null CI: {n_within}")
    print(f"  Categories below null CI : {n_below}  (relative gaps)")
    print("=" * 62)

    if n_below > 0:
        below_cats = df_stats[df_stats["direction"] == "below"]["category"].tolist()
        print(f"\n  Relative gap categories: {below_cats}")
    print(f"\n  Outputs written to: {out}\n")


if __name__ == "__main__":
    main()
