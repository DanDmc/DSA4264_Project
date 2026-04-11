"""
Policy-priority outputs for MOE briefing.

Builds two decision-facing artifacts:
1) Priority Gap Index (category + skill level)
2) Confidence Flag by category (from bootstrap CI width)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from src.analysis.skill_metrics.improved_scr import build_skill_coverage_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Priority Gap Index and Confidence Flags for MOE prioritization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skillner", "skills_list"])
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--job-skills", type=Path, default=None)
    p.add_argument("--course-skills", type=Path, default=COURSES_PROCESSED_DIR / "module_skill_pairs.csv")
    p.add_argument("--nn-path", type=Path, default=None)
    p.add_argument("--improved-per-category", type=Path, default=None)
    p.add_argument("--bootstrap-ci", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _threshold_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def resolve_paths(args: argparse.Namespace) -> None:
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "01b_jobs_cleaned.csv"
        )
    if args.nn_path is None:
        args.nn_path = RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}" / "all_uncovered_nearest_neighbour.csv"
    if args.improved_per_category is None:
        suffix = f"_t{_threshold_tag(args.threshold)}" if args.threshold != 0.70 else ""
        args.improved_per_category = RESULTS_DIR / f"improved_scr_{args.job_source}{suffix}" / "per_category_comparison.csv"
    if args.bootstrap_ci is None:
        args.bootstrap_ci = RESULTS_DIR / f"robustness_{args.job_source}" / "bootstrap_ci_by_category.csv"
    if args.output is None:
        args.output = RESULTS_DIR / f"policy_priority_{args.job_source}_t{_threshold_tag(args.threshold)}"


def load_nn_sim_map(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NN file: {path}")
    nn = pd.read_csv(path)
    return dict(zip(nn["job_skill"], nn["similarity"]))


def confidence_label(ci_width: float, q33: float, q66: float) -> str:
    if ci_width <= q33:
        return "High"
    if ci_width <= q66:
        return "Medium"
    return "Low"


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Job source: {args.job_source}")
    print(f"Threshold : {args.threshold}")
    print(f"Output    : {args.output}\n")

    # Load core datasets
    courses = load_course_skills(args.course_skills)
    jobs = load_job_skills_auto(args.job_skills, args.job_source)
    course_skill_set = set(courses["skill_canon"].unique())

    nn_sim_map = load_nn_sim_map(args.nn_path)
    cov = build_skill_coverage_scores(jobs, course_skill_set, nn_sim_map, args.threshold).reset_index()

    # ---------- Skill-level Priority Gap Index ----------
    # Demand weight: share of mentions for each skill within category.
    mention_counts = (
        jobs.groupby(["category", "skill_canon"]).size().rename("skill_mentions").reset_index()
    )
    total_mentions_by_cat = (
        mention_counts.groupby("category")["skill_mentions"].sum().rename("total_mentions_cat").reset_index()
    )
    skill_df = mention_counts.merge(total_mentions_by_cat, on="category", how="left")
    skill_df = skill_df.merge(cov[["skill_canon", "soft_score", "hard_covered", "exact_match"]], on="skill_canon", how="left")

    # gap_rate per skill: 1 - soft_score (captures partial coverage)
    skill_df["gap_rate"] = 1.0 - skill_df["soft_score"].fillna(0.0)
    skill_df["job_demand_weight"] = skill_df["skill_mentions"] / skill_df["total_mentions_cat"]
    skill_df["priority_gap_index"] = skill_df["gap_rate"] * skill_df["job_demand_weight"]

    skill_ranked = skill_df.sort_values(["priority_gap_index", "skill_mentions"], ascending=[False, False]).reset_index(drop=True)
    skill_ranked.to_csv(args.output / "priority_gap_index_skill_level.csv", index=False)

    top10_per_cat = (
        skill_ranked.groupby("category", group_keys=False).head(10)
        .sort_values(["category", "priority_gap_index"], ascending=[True, False])
        .reset_index(drop=True)
    )
    top10_per_cat.to_csv(args.output / "priority_gap_index_top10_per_category.csv", index=False)

    # ---------- Category-level Priority Gap Index ----------
    if not args.improved_per_category.exists():
        raise FileNotFoundError(f"Missing per-category improved file: {args.improved_per_category}")
    per_cat = pd.read_csv(args.improved_per_category)

    demand_by_cat = jobs.groupby("category").size().rename("job_mentions").reset_index()
    demand_by_cat["job_demand_weight"] = demand_by_cat["job_mentions"] / demand_by_cat["job_mentions"].sum()

    cat = per_cat.merge(demand_by_cat[["category", "job_mentions", "job_demand_weight"]], on="category", how="left")
    cat["gap_rate"] = 1.0 - cat["soft_idf_scr"]
    cat["priority_gap_index"] = cat["gap_rate"] * cat["job_demand_weight"]

    cat = cat.sort_values("priority_gap_index", ascending=False).reset_index(drop=True)

    # ---------- Confidence flag from bootstrap CI ----------
    if not args.bootstrap_ci.exists():
        raise FileNotFoundError(f"Missing bootstrap CI file: {args.bootstrap_ci}")
    b = pd.read_csv(args.bootstrap_ci)
    b["ci_width"] = b["ci95_high"] - b["ci95_low"]

    q33 = float(b["ci_width"].quantile(1/3))
    q66 = float(b["ci_width"].quantile(2/3))

    b["confidence_flag"] = b["ci_width"].apply(lambda x: confidence_label(float(x), q33, q66))
    b = b[["category", "point_estimate_soft_idf_scr", "ci95_low", "ci95_high", "ci_width", "n_unique_job_skills", "confidence_flag"]]

    cat = cat.merge(b[["category", "ci_width", "confidence_flag"]], on="category", how="left")
    cat.to_csv(args.output / "priority_gap_index_category_level.csv", index=False)
    b.sort_values("ci_width", ascending=True).to_csv(args.output / "confidence_flags_by_category.csv", index=False)

    summary = {
        "job_source": args.job_source,
        "threshold": args.threshold,
        "n_categories": int(cat["category"].nunique()),
        "top5_priority_categories": cat[["category", "priority_gap_index"]].head(5).to_dict(orient="records"),
        "confidence_cutoffs_ci_width": {"high_max": round(q33, 4), "medium_max": round(q66, 4)},
        "confidence_counts": b["confidence_flag"].value_counts().to_dict(),
    }
    with (args.output / "summary_policy_priority.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved: priority_gap_index_skill_level.csv")
    print("Saved: priority_gap_index_top10_per_category.csv")
    print("Saved: priority_gap_index_category_level.csv")
    print("Saved: confidence_flags_by_category.csv")
    print("Saved: summary_policy_priority.json")


if __name__ == "__main__":
    main()
