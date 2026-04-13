"""
Foundational layer analysis.

Goal:
- Identify theoretical skills taught by NUS that are not explicitly demanded
  in job postings.
- Compute per-faculty foundational ratio:
    foundational_theoretical_only_skills / total_unique_skills_taught
- Classify foundational skills into relevance bands:
    "Vocabulary Gap"      — concept exists in job market under a different name
                            (nearest job-skill similarity 0.50–0.72)
    "Truly Foundational"  — no close equivalent in job postings
                            (nearest job-skill similarity < 0.50)
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

DEFAULT_COURSE_SKILLS  = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_DEGREE_MAPPING = COURSES_PROCESSED_DIR / "degree_module_mapping.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Foundational layer: theoretical-only skills not explicitly demanded in job postings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skillner", "skills_list"])
    p.add_argument("--job-skills", type=Path, default=None)
    p.add_argument("--course-skills", type=Path, default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--modules-meta", type=Path, default=MODULES_CLEANED)
    p.add_argument("--degree-map", type=Path, default=DEFAULT_DEGREE_MAPPING,
                   help="Path to degree_module_mapping.csv for major-level breakdown.")
    p.add_argument("--nn-file", type=Path, default=None,
                   help="Path to all_uncovered_nearest_neighbour.csv for relevance band classification.")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.nn_file is None:
        args.nn_file = RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}" / "all_uncovered_nearest_neighbour.csv"
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


def make_group_ratio_table(
    course_skills: pd.DataFrame,
    group_col: str,
    demand_set: set[str],
) -> pd.DataFrame:
    """Compute foundational ratio grouped by any column (faculty, major, etc.)."""
    rows: list[dict] = []
    for group_val, grp in course_skills.groupby(group_col):
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
        rows.append({
            group_col: group_val,
            "n_total_unique_skills": n_total,
            "n_theoretical_unique_skills": n_theo,
            "n_foundational_theoretical_only_skills": n_foundational,
            "foundational_ratio": round(ratio, 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("foundational_ratio", ascending=False)
        .reset_index(drop=True)
    )


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


_VOCAB_GAP_THRESHOLD = 0.50   # below this → Truly Foundational; above → Vocabulary Gap
_MATCH_THRESHOLD     = 0.72   # must match threshold used in improved_scr

_BAND_VOCAB_GAP   = "Vocabulary Gap"
_BAND_FOUNDATIONAL = "Truly Foundational"


def build_course_to_job_sim_map(nn_path: Path) -> dict[str, tuple[str, float]]:
    """
    Reverse the job→course NN file to get course_skill → (nearest_job_skill, max_similarity).

    The NN file maps each uncovered job skill to its nearest course skill.
    Inverting it: for each course skill, take the highest-similarity job skill
    that points to it.  This gives a lower-bound on how job-relevant a course
    skill is (lower-bound because the true nearest job skill for a course skill
    might not have chosen it as its nearest neighbour).
    """
    if not nn_path.exists():
        return {}

    nn = pd.read_csv(nn_path)
    required = {"job_skill", "course_skill_match", "similarity"}
    if not required.issubset(nn.columns):
        return {}

    # Keep only sub-threshold rows (genuine gaps, sim < _MATCH_THRESHOLD)
    nn = nn[nn["similarity"] < _MATCH_THRESHOLD].copy()

    # For each course skill, find the job skill with highest similarity
    idx = nn.groupby("course_skill_match")["similarity"].idxmax()
    best = nn.loc[idx, ["course_skill_match", "job_skill", "similarity"]].copy()
    best.columns = ["course_skill", "nearest_job_skill", "nearest_similarity"]

    return {
        row["course_skill"]: (row["nearest_job_skill"], float(row["nearest_similarity"]))
        for _, row in best.iterrows()
    }


def _relevance_band(sim: float | None) -> str:
    if sim is None or sim < _VOCAB_GAP_THRESHOLD:
        return _BAND_FOUNDATIONAL
    return _BAND_VOCAB_GAP


def build_foundational_skill_table(
    theoretical: pd.DataFrame,
    foundational_set: set[str],
    course_fac: pd.DataFrame,
    course_job_sim: dict[str, tuple[str, float]],
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
                "relevance_band",
                "nearest_job_skill",
                "nearest_similarity_pct",
                "interpretation",
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
        sim_info  = course_job_sim.get(skill)
        nearest_job  = sim_info[0] if sim_info else ""
        nearest_sim  = sim_info[1] if sim_info else None
        band         = _relevance_band(nearest_sim)
        interp = (
            f"Concept exists in job market — nearest posting term: '{nearest_job}'"
            if band == _BAND_VOCAB_GAP
            else "No close equivalent found in job postings"
        )
        grouped.append(
            {
                "skill_canon": skill,
                "example_skill_label": str(grp["skill_label"].iloc[0]),
                "n_modules": int(grp["module_code"].nunique()),
                "n_faculties": int(len(faculties)),
                "faculties": " | ".join(faculties),
                "relevance_band": band,
                "nearest_job_skill": nearest_job,
                "nearest_similarity_pct": (
                    f"{nearest_sim:.0%}" if nearest_sim is not None else ""
                ),
                "interpretation": interp,
            }
        )

    return (
        pd.DataFrame(grouped)
        .sort_values(["relevance_band", "n_modules", "n_faculties", "skill_canon"],
                     ascending=[True, False, False, True])
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
    print(f"NN file     : {args.nn_file}")
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

    course_job_sim = build_course_to_job_sim_map(args.nn_file)
    if not course_job_sim:
        print("  [warning] NN file not found or unreadable — relevance bands will default to 'Truly Foundational'")

    foundational_table = build_foundational_skill_table(
        theoretical, foundational_set, course_fac, course_job_sim
    )
    faculty_ratio = make_faculty_ratio_table(course_fac, demand_set)

    # Major-level ratio — join course skills with degree_module_mapping on module_code
    major_ratio: pd.DataFrame | None = None
    if args.degree_map.exists():
        deg_map = pd.read_csv(args.degree_map)[["major", "module_code"]].drop_duplicates()
        course_major = courses.merge(deg_map, on="module_code", how="inner")
        major_ratio = make_group_ratio_table(course_major, "major", demand_set)
    else:
        print(f"  [skip] degree_module_mapping not found at {args.degree_map} — skipping major ratio")

    # Band counts for summary
    n_vocab_gap    = int((foundational_table["relevance_band"] == _BAND_VOCAB_GAP).sum())
    n_truly_found  = int((foundational_table["relevance_band"] == _BAND_FOUNDATIONAL).sum())
    n_total_found  = len(foundational_table)
    pct_vocab_gap  = round(n_vocab_gap  / n_total_found, 4) if n_total_found else 0.0
    pct_truly_found = round(n_truly_found / n_total_found, 4) if n_total_found else 0.0

    foundational_table.to_csv(args.output / "foundational_theoretical_skills_no_job_demand.csv", index=False)
    faculty_ratio.to_csv(args.output / "foundational_ratio_by_faculty.csv", index=False)
    plot_faculty_ratio(faculty_ratio, args.output / "fig_foundational_ratio_by_faculty.png")
    if major_ratio is not None:
        major_ratio.to_csv(args.output / "foundational_ratio_by_major.csv", index=False)
        print("Saved: foundational_ratio_by_major.csv")

    summary = {
        "job_source": args.job_source,
        "n_unique_job_skills": len(demand_set),
        "n_unique_theoretical_course_skills": len(theoretical_set),
        "n_foundational_theoretical_skills_no_job_demand": len(foundational_set),
        "pct_theoretical_skills_without_job_demand": round(
            (len(foundational_set) / len(theoretical_set)) if theoretical_set else 0.0, 4
        ),
        "relevance_bands": {
            "vocabulary_gap": {
                "description": "Concept exists in job market under a different name (NN sim 0.50–0.72)",
                "n": n_vocab_gap,
                "pct_of_foundational": pct_vocab_gap,
            },
            "truly_foundational": {
                "description": "No close equivalent found in job postings (NN sim < 0.50)",
                "n": n_truly_found,
                "pct_of_foundational": pct_truly_found,
            },
        },
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
    print()
    print("  Of those foundational skills:")
    print(f"    Vocabulary Gap      : {n_vocab_gap:,} ({pct_vocab_gap:.1%})")
    print(f"      → Concept is job-relevant but described differently in postings")
    print(f"    Truly Foundational  : {n_truly_found:,} ({pct_truly_found:.1%})")
    print(f"      → No close equivalent found in job postings")
    print()
    print(f"Mean faculty foundational ratio           : {faculty_ratio['foundational_ratio'].mean():.1%}")
    print("=" * 64)
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
