"""
Foundational layer analysis.

Goal:
- Identify theoretical skills taught by NUS that are not explicitly demanded
  in job postings.
- Compute per-faculty foundational ratio:
    foundational_theoretical_only_skills / total_unique_skills_taught
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.config import (
        COURSES_PROCESSED_DIR,
        JOBS_PROCESSED_DIR,
        MODULES_CLEANED,
        RESULTS_DIR,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import (
        COURSES_PROCESSED_DIR,
        JOBS_PROCESSED_DIR,
        MODULES_CLEANED,
        RESULTS_DIR,
    )

from src.analysis.skill_metrics.baseline_scr import load_course_skills, load_job_skills_auto

DEFAULT_COURSE_SKILLS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Foundational layer: theoretical-only skills not explicitly demanded in job postings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skillner", "skills_list"])
    p.add_argument("--job-skills", type=Path, default=None)
    p.add_argument("--course-skills", type=Path, default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--modules-meta", type=Path, default=MODULES_CLEANED)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.output is None:
        args.output = RESULTS_DIR / f"foundational_layer_{args.job_source}"


def load_module_faculty_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"module code", "faculty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in modules metadata: {sorted(missing)}. Found: {list(df.columns)}"
        )

    out = (
        df[["module code", "faculty"]]
        .rename(columns={"module code": "module_code"})
        .dropna(subset=["module_code", "faculty"])
        .copy()
    )
    out["module_code"] = out["module_code"].astype(str).str.strip()
    out["faculty"] = out["faculty"].astype(str).str.strip()
    return out.drop_duplicates()


def make_faculty_ratio_table(course_fac: pd.DataFrame, demand_set: set[str]) -> pd.DataFrame:
    rows: list[dict] = []

    for faculty, grp in course_fac.groupby("faculty"):
        total_skills = set(grp["skill_canon"].dropna().astype(str))
        theo_skills = set(
            grp.loc[
                grp["knowledge_type"].astype(str).str.lower().eq("theoretical"),
                "skill_canon",
            ].dropna().astype(str)
        )
        foundational_theo = theo_skills - demand_set

        n_total = len(total_skills)
        n_theo = len(theo_skills)
        n_foundational = len(foundational_theo)
        ratio = (n_foundational / n_total) if n_total > 0 else 0.0

        rows.append(
            {
                "faculty": faculty,
                "n_total_unique_skills": n_total,
                "n_theoretical_unique_skills": n_theo,
                "n_foundational_theoretical_only_skills": n_foundational,
                "foundational_ratio": round(ratio, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("foundational_ratio", ascending=False).reset_index(drop=True)


def build_foundational_skill_table(
    theoretical: pd.DataFrame,
    foundational_set: set[str],
    course_fac: pd.DataFrame,
) -> pd.DataFrame:
    f = theoretical[theoretical["skill_canon"].isin(foundational_set)].copy()
    if f.empty:
        return pd.DataFrame(
            columns=[
                "skill_canon",
                "example_skill_label",
                "n_modules",
                "n_faculties",
                "faculties",
            ]
        )

    fac_map = (
        course_fac[["module_code", "faculty"]]
        .drop_duplicates()
        .set_index("module_code")["faculty"]
        .to_dict()
    )
    f["faculty"] = f["module_code"].map(fac_map).fillna("Unknown")

    grouped = []
    for skill, grp in f.groupby("skill_canon"):
        faculties = sorted(set(grp["faculty"].astype(str)))
        grouped.append(
            {
                "skill_canon": skill,
                "example_skill_label": str(grp["skill_label"].iloc[0]),
                "n_modules": int(grp["module_code"].nunique()),
                "n_faculties": int(len(faculties)),
                "faculties": " | ".join(faculties),
            }
        )

    return (
        pd.DataFrame(grouped)
        .sort_values(["n_modules", "n_faculties", "skill_canon"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def plot_faculty_ratio(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("foundational_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(d))))
    ax.barh(d["faculty"], d["foundational_ratio"], color="#1f77b4", alpha=0.9)
    ax.set_xlabel("Foundational ratio (theoretical-only / total unique skills)")
    ax.set_title("Foundational Skill Ratio by Faculty")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Job source  : {args.job_source}")
    print(f"Job skills  : {args.job_skills}")
    print(f"Course skill: {args.course_skills}")
    print(f"Modules meta: {args.modules_meta}")
    print(f"Output dir  : {args.output}\n")

    courses = load_course_skills(args.course_skills)
    jobs = load_job_skills_auto(args.job_skills, args.job_source)
    module_fac = load_module_faculty_map(args.modules_meta)

    # Attach faculty to each module-skill row
    course_fac = courses.merge(module_fac, on="module_code", how="left")
    course_fac["faculty"] = course_fac["faculty"].fillna("Unknown")

    demand_set = set(jobs["skill_canon"].dropna().astype(str).unique())

    theoretical = course_fac[
        course_fac["knowledge_type"].astype(str).str.lower().eq("theoretical")
    ].copy()
    theoretical_set = set(theoretical["skill_canon"].dropna().astype(str).unique())
    foundational_set = theoretical_set - demand_set

    foundational_table = build_foundational_skill_table(theoretical, foundational_set, course_fac)
    faculty_ratio = make_faculty_ratio_table(course_fac, demand_set)

    foundational_table.to_csv(args.output / "foundational_theoretical_skills_no_job_demand.csv", index=False)
    faculty_ratio.to_csv(args.output / "foundational_ratio_by_faculty.csv", index=False)
    plot_faculty_ratio(faculty_ratio, args.output / "fig_foundational_ratio_by_faculty.png")

    summary = {
        "job_source": args.job_source,
        "n_unique_job_skills": len(demand_set),
        "n_unique_theoretical_course_skills": len(theoretical_set),
        "n_foundational_theoretical_skills_no_job_demand": len(foundational_set),
        "pct_theoretical_skills_without_job_demand": round(
            (len(foundational_set) / len(theoretical_set)) if theoretical_set else 0.0, 4
        ),
        "n_faculties": int(faculty_ratio["faculty"].nunique()),
        "mean_foundational_ratio_across_faculties": round(float(faculty_ratio["foundational_ratio"].mean()), 4),
    }
    with (args.output / "summary_foundational_layer.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved: foundational_theoretical_skills_no_job_demand.csv")
    print("Saved: foundational_ratio_by_faculty.csv")
    print("Saved: fig_foundational_ratio_by_faculty.png")
    print("Saved: summary_foundational_layer.json")

    print("\n" + "=" * 64)
    print("FOUNDATIONAL LAYER — HEADLINE")
    print("=" * 64)
    print(f"Unique theoretical course skills           : {len(theoretical_set):,}")
    print(f"Theoretical skills with zero job demand   : {len(foundational_set):,}")
    print(
        "Share of theoretical skills with zero demand: "
        f"{((len(foundational_set) / len(theoretical_set)) if theoretical_set else 0.0):.1%}"
    )
    print(f"Mean faculty foundational ratio           : {faculty_ratio['foundational_ratio'].mean():.1%}")
    print("=" * 64)
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
