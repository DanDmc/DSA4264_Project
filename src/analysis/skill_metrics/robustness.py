"""
Robustness analysis for improved skill coverage metrics.

This script stress-tests Soft-IDF-SCR with three checks:
1) Threshold sensitivity across cosine thresholds.
2) Coverage depth (modules per covered job skill) by category.
3) Bootstrap 95% confidence intervals for per-category Soft-IDF-SCR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR

from src.analysis.skill_metrics.baseline_scr import load_course_skills, load_job_skills_auto
from src.analysis.skill_metrics.improved_scr import (
    build_skill_coverage_scores,
    compute_idf,
    compute_improved_scr,
)

DEFAULT_COURSE_SKILLS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"


def _resolve_paths(args: argparse.Namespace) -> None:
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.nn_path is None:
        args.nn_path = (
            RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}" / "all_uncovered_nearest_neighbour.csv"
        )
    if args.output is None:
        args.output = RESULTS_DIR / f"robustness_{args.job_source}"


def _load_nn_files(nn_path: Path) -> tuple[dict[str, float], dict[str, str]]:
    if not nn_path.exists():
        raise FileNotFoundError(
            f"Nearest-neighbour file not found: {nn_path}\n"
            "Run vocabulary_mismatch.py first."
        )
    nn_df = pd.read_csv(nn_path)
    sim_map = dict(zip(nn_df["job_skill"], nn_df["similarity"]))
    match_map = dict(zip(nn_df["job_skill"], nn_df["course_skill_match"]))
    return sim_map, match_map


def threshold_sensitivity(
    job_skills: pd.DataFrame,
    course_skill_set: set[str],
    nn_similarities: dict[str, float],
    idf_map: dict[str, float],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for t in thresholds:
        coverage_scores = build_skill_coverage_scores(
            job_skills=job_skills,
            course_skill_set=course_skill_set,
            nn_similarities=nn_similarities,
            threshold=float(t),
        )
        per_cat = compute_improved_scr(job_skills, coverage_scores, idf_map)

        macro_soft_idf = float(per_cat["soft_idf_scr"].mean())
        micro_soft_idf = float(
            (per_cat["soft_idf_scr"] * per_cat["n_unique_job_skills"]).sum()
            / per_cat["n_unique_job_skills"].sum()
        )

        n_total = len(coverage_scores)
        n_semantic = int(coverage_scores["hard_covered"].sum())

        rows.append(
            {
                "threshold": round(float(t), 2),
                "macro_soft_idf_scr": round(macro_soft_idf, 4),
                "micro_soft_idf_scr": round(micro_soft_idf, 4),
                "n_semantic_covered": n_semantic,
                "n_genuine_gaps": n_total - n_semantic,
                "pct_genuine_gaps": round((n_total - n_semantic) / n_total, 4),
            }
        )

    return pd.DataFrame(rows)


def compute_coverage_depth(
    job_skills: pd.DataFrame,
    courses: pd.DataFrame,
    coverage_scores: pd.DataFrame,
    nn_match_map: dict[str, str],
) -> pd.DataFrame:
    # Number of distinct modules teaching each course skill
    modules_per_course_skill = (
        courses.groupby("skill_canon")["module_code"]
        .nunique()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []

    for category, grp in job_skills.groupby("category"):
        skills = grp["skill_canon"].unique()
        depth_values: list[int] = []

        for skill in skills:
            if skill not in coverage_scores.index:
                continue
            srow = coverage_scores.loc[skill]
            if not bool(srow["hard_covered"]):
                continue

            if bool(srow["exact_match"]):
                depth_values.append(int(modules_per_course_skill.get(skill, 0)))
                continue

            # Semantic-covered but non-exact: map via nearest course skill label
            nearest_course_skill = nn_match_map.get(skill, "")
            depth_values.append(int(modules_per_course_skill.get(nearest_course_skill, 0)))

        n_cov = len(depth_values)
        mean_depth = float(np.mean(depth_values)) if n_cov > 0 else 0.0
        median_depth = float(np.median(depth_values)) if n_cov > 0 else 0.0

        rows.append(
            {
                "category": category,
                "n_covered_skills": n_cov,
                "mean_modules_per_covered_skill": round(mean_depth, 4),
                "median_modules_per_covered_skill": round(median_depth, 4),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mean_modules_per_covered_skill", ascending=False
    ).reset_index(drop=True)


def bootstrap_soft_idf_ci(
    job_skills: pd.DataFrame,
    coverage_scores: pd.DataFrame,
    idf_map: dict[str, float],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for category, grp in job_skills.groupby("category"):
        unique_skills = grp["skill_canon"].unique()
        if len(unique_skills) == 0:
            continue

        soft = np.array(
            [
                float(coverage_scores.loc[s, "soft_score"])
                if s in coverage_scores.index
                else 0.0
                for s in unique_skills
            ],
            dtype=float,
        )
        idf = np.array([float(idf_map.get(s, 1.0)) for s in unique_skills], dtype=float)

        denom = float(idf.sum())
        point = float((soft * idf).sum() / denom) if denom > 0 else 0.0

        sims = np.zeros(n_bootstrap, dtype=float)
        n = len(unique_skills)
        idx = np.arange(n)

        for i in range(n_bootstrap):
            sample_idx = rng.choice(idx, size=n, replace=True)
            soft_s = soft[sample_idx]
            idf_s = idf[sample_idx]
            d = float(idf_s.sum())
            sims[i] = float((soft_s * idf_s).sum() / d) if d > 0 else 0.0

        rows.append(
            {
                "category": category,
                "point_estimate_soft_idf_scr": round(point, 4),
                "ci95_low": round(float(np.percentile(sims, 2.5)), 4),
                "ci95_high": round(float(np.percentile(sims, 97.5)), 4),
                "bootstrap_sd": round(float(np.std(sims, ddof=1)), 4),
                "n_unique_job_skills": n,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "point_estimate_soft_idf_scr", ascending=False
    ).reset_index(drop=True)


def plot_threshold_sweep(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["threshold"], df["macro_soft_idf_scr"], marker="o", linewidth=2, label="Macro Soft-IDF-SCR")
    ax.plot(df["threshold"], df["micro_soft_idf_scr"], marker="s", linewidth=2, label="Micro Soft-IDF-SCR")
    ax.set_xlabel("Cosine similarity threshold (theta)")
    ax.set_ylabel("Soft-IDF-SCR")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Threshold Sensitivity: Soft-IDF-SCR vs theta")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_depth(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("mean_modules_per_covered_skill", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(8, 0.34 * len(d))))
    ax.barh(d["category"], d["mean_modules_per_covered_skill"], color="#2E86AB", alpha=0.9)
    ax.set_xlabel("Mean modules per covered skill")
    ax.set_title("Coverage Depth by Category")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_ci(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("point_estimate_soft_idf_scr", ascending=True)
    y = np.arange(len(d))

    lo = d["point_estimate_soft_idf_scr"] - d["ci95_low"]
    hi = d["ci95_high"] - d["point_estimate_soft_idf_scr"]

    fig, ax = plt.subplots(figsize=(11, max(8, 0.34 * len(d))))
    ax.errorbar(
        d["point_estimate_soft_idf_scr"],
        y,
        xerr=[lo, hi],
        fmt="o",
        capsize=3,
        color="#117A65",
        ecolor="#117A65",
        elinewidth=1,
        markersize=4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["category"], fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Soft-IDF-SCR with 95% bootstrap CI")
    ax.set_title("Per-Category Uncertainty (Bootstrap 95% CI)")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robustness checks: threshold sweep, depth, and bootstrap CI for Soft-IDF-SCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skillner", "skills_list"])
    p.add_argument("--job-skills", type=Path, default=None)
    p.add_argument("--course-skills", type=Path, default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--nn-path", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--threshold-start", type=float, default=0.50)
    p.add_argument("--threshold-end", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
    p.add_argument("--depth-threshold", type=float, default=0.72)
    p.add_argument("--bootstrap-threshold", type=float, default=0.72)
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _resolve_paths(args)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Job source  : {args.job_source}")
    print(f"Job skills  : {args.job_skills}")
    print(f"Course skill: {args.course_skills}")
    print(f"NN file     : {args.nn_path}")
    print(f"Output dir  : {args.output}\n")

    print("Loading data …")
    courses = load_course_skills(args.course_skills)
    jobs = load_job_skills_auto(args.job_skills, args.job_source)
    course_skill_set = set(courses["skill_canon"].unique())
    nn_sim, nn_match = _load_nn_files(args.nn_path)
    idf_map = compute_idf(jobs)

    print(f"  Unique course skills : {len(course_skill_set):,}")
    print(f"  Unique job skills    : {jobs['skill_canon'].nunique():,}")

    print("\nRunning threshold sensitivity …")
    thresholds = np.round(
        np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step),
        2,
    )
    sweep_df = threshold_sensitivity(
        job_skills=jobs,
        course_skill_set=course_skill_set,
        nn_similarities=nn_sim,
        idf_map=idf_map,
        thresholds=thresholds,
    )
    sweep_df.to_csv(args.output / "threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep_df, args.output / "fig_threshold_sweep.png")
    print("  Saved: threshold_sweep.csv")

    print(f"\nComputing coverage depth (theta={args.depth_threshold:.2f}) …")
    depth_cov = build_skill_coverage_scores(jobs, course_skill_set, nn_sim, args.depth_threshold)
    depth_df = compute_coverage_depth(jobs, courses, depth_cov, nn_match)
    depth_df.to_csv(args.output / "coverage_depth_by_category.csv", index=False)
    plot_coverage_depth(depth_df, args.output / "fig_coverage_depth.png")
    print("  Saved: coverage_depth_by_category.csv")

    print(f"\nBootstrapping 95% CI (theta={args.bootstrap_threshold:.2f}, n={args.bootstrap_iterations}) …")
    boot_cov = build_skill_coverage_scores(jobs, course_skill_set, nn_sim, args.bootstrap_threshold)
    boot_df = bootstrap_soft_idf_ci(
        job_skills=jobs,
        coverage_scores=boot_cov,
        idf_map=idf_map,
        n_bootstrap=args.bootstrap_iterations,
        seed=args.seed,
    )
    boot_df.to_csv(args.output / "bootstrap_ci_by_category.csv", index=False)
    plot_bootstrap_ci(boot_df, args.output / "fig_bootstrap_ci.png")
    print("  Saved: bootstrap_ci_by_category.csv")

    summary = {
        "job_source": args.job_source,
        "threshold_sweep": {
            "start": args.threshold_start,
            "end": args.threshold_end,
            "step": args.threshold_step,
            "best_macro_soft_idf": float(sweep_df["macro_soft_idf_scr"].max()),
            "best_macro_threshold": float(sweep_df.loc[sweep_df["macro_soft_idf_scr"].idxmax(), "threshold"]),
        },
        "depth_threshold": args.depth_threshold,
        "mean_depth_overall": float(depth_df["mean_modules_per_covered_skill"].mean()),
        "bootstrap_threshold": args.bootstrap_threshold,
        "bootstrap_iterations": args.bootstrap_iterations,
        "macro_soft_idf_at_bootstrap_threshold": float(
            compute_improved_scr(jobs, boot_cov, idf_map)["soft_idf_scr"].mean()
        ),
    }
    with (args.output / "summary_robustness.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  Saved: summary_robustness.json")

    print("\n" + "=" * 64)
    print("ROBUSTNESS — HEADLINE")
    print("=" * 64)
    print(
        "Threshold sweep (macro Soft-IDF-SCR): "
        f"min={sweep_df['macro_soft_idf_scr'].min():.1%}, "
        f"max={sweep_df['macro_soft_idf_scr'].max():.1%}"
    )
    print(
        f"Coverage depth mean (theta={args.depth_threshold:.2f}): "
        f"{depth_df['mean_modules_per_covered_skill'].mean():.2f} modules/covered skill"
    )
    ci_width = (boot_df["ci95_high"] - boot_df["ci95_low"]).mean()
    print(
        f"Bootstrap avg CI width (theta={args.bootstrap_threshold:.2f}): {ci_width:.1%}"
    )
    print("=" * 64)
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
