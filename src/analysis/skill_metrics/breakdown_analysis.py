"""
Breakdown Analysis — Pipeline 3 component.

Computes Soft-SCR / Soft-IDF-SCR across six breakdown dimensions:

  Job-side (Soft-IDF-SCR — IDF recomputed per grouping level):
    1. Job category          (43 groups)
    2. SSOC submajor         (39 groups)
    3. SSOC minor            (126 groups)

  Course-side (Soft-SCR, no IDF — fair cross-programme comparison):
    4. Faculty               (8 groups)
    5. Major / Degree        (55 groups)
    6. Department            (variable)

  For course-side breakdowns, two coverage variants are reported:
    - Maximum coverage   : all modules (core + elective)
    - Guaranteed coverage: core modules only

  Applied vs theoretical split is reported for ALL breakdowns:
    - Overall Soft-SCR
    - Applied-only Soft-SCR  (knowledge_type == "applied")
    - Theoretical-only Soft-SCR

Metric definitions
------------------
Soft-IDF-SCR (job-side):
    sim(j) = 1.0 if exact match, nn_sim if ≥ θ, else 0.0
    idf(j) = log(N/df) + 1  (N = number of groups in this breakdown level)
    Soft-IDF-SCR = Σ sim(j)×idf(j) / Σ idf(j)

Soft-SCR (course-side):
    Same sim(j) formula, but equal weights (no IDF).
    IDF is not meaningful when comparing programmes — it would penalise
    skills that are demanded across many industries, which is irrelevant
    when asking "how much of ALL job demand does this faculty cover?"

Outputs
-------
results/breakdown_analysis_{source}/
  ├── job_category.csv
  ├── ssoc_submajor.csv
  ├── ssoc_minor.csv
  ├── faculty.csv
  ├── major.csv
  ├── department.csv
  ├── summary_breakdown.json
  └── fig_*.png

Usage
-----
    python -m src.analysis.skill_metrics.breakdown_analysis
    python -m src.analysis.skill_metrics.breakdown_analysis --job-source skills_list
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
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RAW_DIR, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RAW_DIR, RESULTS_DIR

from src.analysis.skill_metrics.baseline_scr import (
    canonicalize,
    load_course_skills,
    load_job_skills_auto,
)

DEFAULT_COURSE_SKILLS    = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_MODULES_CLEANED  = COURSES_PROCESSED_DIR / "modules_cleaned.csv"
DEFAULT_DEGREE_MAPPING   = COURSES_PROCESSED_DIR / "degree_module_mapping.csv"
DEFAULT_MAJOR_SSOC_MAP   = RAW_DIR / "major_ssoc_mapping.csv"
DEFAULT_THRESHOLD        = 0.72


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nn_data(nn_path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Returns (sim_map, match_map) where match_map[job_skill] = nearest course_skill_match."""
    if not nn_path.exists():
        raise FileNotFoundError(
            f"Nearest-neighbour file not found: {nn_path}\n"
            "Run pipeline_build_metric.py first."
        )
    nn_df = pd.read_csv(nn_path)
    sim_map   = dict(zip(nn_df["job_skill"], nn_df["similarity"]))
    match_map = dict(zip(nn_df["job_skill"], nn_df["course_skill_match"]))
    return sim_map, match_map


def load_degree_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["module_code"] = df["module_code"].astype(str).str.strip()
    df["module_type"] = df["module_type"].astype(str).str.strip().str.lower()
    return df


def load_modules_cleaned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={"module_code": "module_code"})
    # handle "module code" with space
    if "module_code" not in df.columns and "module code" in df.columns:
        df = df.rename(columns={"module code": "module_code"})
    df["module_code"] = df["module_code"].astype(str).str.strip()
    return df[["module_code", "department", "faculty"]].drop_duplicates()


# ─────────────────────────────────────────────────────────────────────────────
# Coverage score per skill (soft matching)
# ─────────────────────────────────────────────────────────────────────────────

def get_soft_score(skill: str, course_skill_set: set[str],
                   nn_similarities: dict[str, float], threshold: float,
                   nn_match_map: dict[str, str] | None = None) -> float:
    if skill in course_skill_set:
        return 1.0
    sim = nn_similarities.get(skill, 0.0)
    if sim < threshold:
        return 0.0
    # For course-side breakdowns: only credit if the matched course skill is in this group
    if nn_match_map is not None and nn_match_map.get(skill, "") not in course_skill_set:
        return 0.0
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# Job-side breakdown (Soft-IDF-SCR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_job_side_breakdown(
    jobs:             pd.DataFrame,
    course_skills:    pd.DataFrame,
    nn_similarities:  dict[str, float],
    nn_match_map:     dict[str, str],
    threshold:        float,
    group_col:        str,
    knowledge_type:   str | None = None,
) -> pd.DataFrame:
    """
    Compute Soft-IDF-SCR grouping job skills by group_col.
    IDF is recomputed within this grouping level.
    knowledge_type: None=all, 'applied'=applied only, 'theoretical'=theoretical only
    """
    # Filter course skills by knowledge_type if specified
    cs = course_skills.copy()
    if knowledge_type:
        cs = cs[cs["knowledge_type"].astype(str).str.lower() == knowledge_type]
    course_skill_set = set(cs["skill_canon"].unique())
    # When knowledge_type is filtered, enforce NN match is also in the filtered set
    match_map = nn_match_map if knowledge_type else None

    # IDF: N = number of unique groups
    groups = jobs[group_col].dropna().unique()
    N = len(groups)

    df_counts = jobs.groupby("skill_canon")[group_col].nunique()
    idf_map = {skill: np.log(N / df) + 1.0 for skill, df in df_counts.items()}

    rows: list[dict[str, Any]] = []

    for grp_val, grp in jobs.groupby(group_col):
        skills = grp["skill_canon"].unique()
        n_total = len(skills)
        if n_total == 0:
            continue

        n_exact = 0
        n_semantic = 0
        sum_soft_idf = 0.0
        sum_denom_idf = 0.0

        for skill in skills:
            soft = get_soft_score(skill, course_skill_set, nn_similarities, threshold,
                                  nn_match_map=match_map)
            idf_w = idf_map.get(skill, 1.0)
            exact = skill in course_skill_set
            semantic = soft > 0.0

            if exact:
                n_exact += 1
            if semantic:
                n_semantic += 1

            sum_soft_idf  += soft * idf_w
            sum_denom_idf += idf_w

        baseline_scr = n_exact    / n_total if n_total > 0 else 0.0
        soft_idf_scr = sum_soft_idf / sum_denom_idf if sum_denom_idf > 0 else 0.0

        rows.append({
            group_col:              grp_val,
            "n_unique_job_skills":  n_total,
            "n_exact_covered":      n_exact,
            "n_semantic_covered":   n_semantic,
            "n_genuine_gaps":       n_total - n_semantic,
            "baseline_scr":         round(baseline_scr, 4),
            "soft_idf_scr":         round(soft_idf_scr, 4),
            "knowledge_type_filter":knowledge_type or "all",
        })

    return pd.DataFrame(rows).sort_values("soft_idf_scr", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Course-side breakdown (Soft-SCR, no IDF)
# ─────────────────────────────────────────────────────────────────────────────

def compute_course_side_breakdown(
    jobs:             pd.DataFrame,
    course_skills:    pd.DataFrame,
    nn_similarities:  dict[str, float],
    nn_match_map:     dict[str, str],
    threshold:        float,
    group_col:        str,
    module_type:      str | None = None,
    knowledge_type:   str | None = None,
) -> pd.DataFrame:
    """
    Compute Soft-SCR (no IDF) grouping course skills by group_col.
    Denominator = ALL unique job-demanded skills across all categories.

    module_type:    None=all, 'core'=guaranteed coverage, 'elective'=elective only
    knowledge_type: None=all, 'applied', 'theoretical'
    """
    all_job_skills = jobs["skill_canon"].unique()
    n_total = len(all_job_skills)

    rows: list[dict[str, Any]] = []

    for grp_val, grp in course_skills.groupby(group_col):
        # Filter by module_type if specified
        cs = grp.copy()
        if module_type:
            cs = cs[cs["module_type"].astype(str).str.lower() == module_type]
        # Filter by knowledge_type if specified
        if knowledge_type:
            cs = cs[cs["knowledge_type"].astype(str).str.lower() == knowledge_type]

        course_skill_set = set(cs["skill_canon"].unique())

        n_exact = 0
        n_semantic = 0
        sum_soft = 0.0

        for skill in all_job_skills:
            soft = get_soft_score(skill, course_skill_set, nn_similarities, threshold,
                                  nn_match_map=nn_match_map)
            exact = skill in course_skill_set
            semantic = soft > 0.0

            if exact:
                n_exact += 1
            if semantic:
                n_semantic += 1
            sum_soft += soft

        baseline_scr = n_exact    / n_total if n_total > 0 else 0.0
        soft_scr     = sum_soft   / n_total if n_total > 0 else 0.0

        rows.append({
            group_col:              grp_val,
            "n_unique_job_skills":  n_total,
            "n_exact_covered":      n_exact,
            "n_semantic_covered":   n_semantic,
            "n_genuine_gaps":       n_total - n_semantic,
            "baseline_scr":         round(baseline_scr, 4),
            "soft_scr":             round(soft_scr, 4),
            "module_type_filter":   module_type or "all",
            "knowledge_type_filter":knowledge_type or "all",
        })

    return pd.DataFrame(rows).sort_values("soft_scr", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build full breakdown table with all splits
# ─────────────────────────────────────────────────────────────────────────────

def build_course_side_full(
    jobs:            pd.DataFrame,
    course_skills:   pd.DataFrame,
    nn_similarities: dict[str, float],
    nn_match_map:    dict[str, str],
    threshold:       float,
    group_col:       str,
    include_core:    bool = True,
) -> pd.DataFrame:
    """
    Returns one combined DataFrame with columns for each variant.
    include_core=False drops core-only variants — use for faculty/department
    where module_type is only meaningful at the major level.
    """
    combos = [
        ("all",         None,          None),
        ("applied",     None,          "applied"),
        ("theoretical", None,          "theoretical"),
    ]
    if include_core:
        combos += [
            ("all_core",    "core",    None),
            ("applied_core","core",    "applied"),
            ("theo_core",   "core",    "theoretical"),
        ]

    base = None
    for label, mt, kt in combos:
        df = compute_course_side_breakdown(
            jobs, course_skills, nn_similarities, nn_match_map, threshold,
            group_col, module_type=mt, knowledge_type=kt,
        )
        df = df.rename(columns={
            "soft_scr":      f"soft_scr_{label}",
            "baseline_scr":  f"baseline_scr_{label}",
            "n_exact_covered":   f"n_exact_{label}",
            "n_semantic_covered":f"n_semantic_{label}",
            "n_genuine_gaps":    f"n_gaps_{label}",
        }).drop(columns=["module_type_filter", "knowledge_type_filter",
                          "n_unique_job_skills"], errors="ignore")

        if base is None:
            base = df
        else:
            base = base.merge(df, on=group_col, how="outer")

    # Re-attach job skill count
    base["n_unique_job_skills"] = jobs["skill_canon"].nunique()
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_breakdown(df: pd.DataFrame, group_col: str,
                   metric_col: str, title: str, out: Path,
                   top_n: int = 30) -> None:
    d = df.nlargest(top_n, metric_col).sort_values(metric_col, ascending=True)
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(d))))
    ax.barh(d[group_col].astype(str), d[metric_col], color="#2980b9", alpha=0.88)
    macro = float(df[metric_col].mean())
    ax.axvline(macro, color="#e74c3c", linestyle="--", linewidth=1.3,
               label=f"Macro avg: {macro:.1%}")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Coverage Rate", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_core_vs_max(df: pd.DataFrame, group_col: str, out: Path, top_n: int = 20) -> None:
    """Side-by-side: maximum coverage (core+elective) vs guaranteed (core only)."""
    if "soft_scr_all" not in df.columns or "soft_scr_all_core" not in df.columns:
        return
    d = df.nlargest(top_n, "soft_scr_all").sort_values("soft_scr_all", ascending=True)

    y      = np.arange(len(d))
    height = 0.38
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(d))))
    ax.barh(y + height/2, d["soft_scr_all"],      height=height,
            color="#27ae60", alpha=0.85, label="Maximum (core + elective)")
    ax.barh(y - height/2, d["soft_scr_all_core"], height=height,
            color="#e74c3c", alpha=0.85, label="Guaranteed (core only)")
    ax.set_yticks(y)
    ax.set_yticklabels(d[group_col].astype(str), fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Soft-SCR", fontsize=10)
    ax.set_title(f"Maximum vs Guaranteed Coverage by {group_col.replace('_', ' ').title()}",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_applied_vs_theoretical(df: pd.DataFrame, group_col: str, out: Path,
                                 top_n: int = 20) -> None:
    """Stacked view: applied and theoretical coverage."""
    if "soft_scr_applied" not in df.columns or "soft_scr_theoretical" not in df.columns:
        return
    d = df.nlargest(top_n, "soft_scr_all").sort_values("soft_scr_all", ascending=True)

    y      = np.arange(len(d))
    height = 0.28
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(d))))
    ax.barh(y + height,    d["soft_scr_all"],         height=height,
            color="#2980b9", alpha=0.85, label="Overall")
    ax.barh(y,             d["soft_scr_applied"],     height=height,
            color="#27ae60", alpha=0.85, label="Applied only")
    ax.barh(y - height,    d["soft_scr_theoretical"], height=height,
            color="#e67e22", alpha=0.85, label="Theoretical only")
    ax.set_yticks(y)
    ax.set_yticklabels(d[group_col].astype(str), fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Soft-SCR", fontsize=10)
    ax.set_title(f"Applied vs Theoretical Coverage by {group_col.replace('_', ' ').title()}",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SSOC-aligned denominator for course-side breakdowns
# ─────────────────────────────────────────────────────────────────────────────

def load_major_ssoc_map(path: Path) -> dict[str, list[str]]:
    """Returns {major: [ssoc_minor_title, ...]}."""
    df = pd.read_csv(path)
    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["major"], []).append(row["ssoc_minor_title"])
    return mapping


def derive_group_ssoc_map(
    degree_map:     pd.DataFrame,
    major_ssoc_map: dict[str, list[str]],
    group_col:      str,
) -> dict[str, list[str]]:
    """
    Roll up major→SSOC to faculty/department→SSOC.
    For each faculty/dept, take the union of SSOC minors across all its majors.
    """
    group_ssoc: dict[str, set[str]] = {}
    pairs = degree_map[["major", group_col]].drop_duplicates().dropna()
    for _, row in pairs.iterrows():
        group = str(row[group_col])
        ssoc_list = major_ssoc_map.get(str(row["major"]), [])
        group_ssoc.setdefault(group, set()).update(ssoc_list)
    return {g: list(s) for g, s in group_ssoc.items() if s}


def compute_course_breakdown_ssoc_aligned(
    jobs:            pd.DataFrame,
    course_skills:   pd.DataFrame,
    nn_similarities: dict[str, float],
    nn_match_map:    dict[str, str],
    threshold:       float,
    group_col:       str,
    group_ssoc_map:  dict[str, list[str]],
    module_type:     str | None = None,
    knowledge_type:  str | None = None,
) -> pd.DataFrame:
    """
    Soft-SCR per group where denominator = skills from SSOC-mapped job roles only.
    Works for faculty, major, or department — group_col determines the grouping.
    """
    rows: list[dict[str, Any]] = []

    for grp_val, grp in course_skills.groupby(group_col):
        relevant_ssoc = group_ssoc_map.get(str(grp_val))
        if not relevant_ssoc:
            continue

        if "ssoc_minor_title" not in jobs.columns:
            relevant_jobs = jobs
        else:
            relevant_jobs = jobs[jobs["ssoc_minor_title"].isin(relevant_ssoc)]

        relevant_skills = relevant_jobs["skill_canon"].unique()
        n_total = len(relevant_skills)
        if n_total == 0:
            continue

        cs = grp.copy()
        if module_type:
            cs = cs[cs["module_type"].astype(str).str.lower() == module_type]
        if knowledge_type:
            cs = cs[cs["knowledge_type"].astype(str).str.lower() == knowledge_type]
        course_skill_set = set(cs["skill_canon"].unique())

        n_exact = 0
        n_semantic = 0
        sum_soft = 0.0

        for skill in relevant_skills:
            soft = get_soft_score(skill, course_skill_set, nn_similarities, threshold,
                                  nn_match_map=nn_match_map)
            if skill in course_skill_set:
                n_exact += 1
            if soft > 0.0:
                n_semantic += 1
            sum_soft += soft

        rows.append({
            group_col:               grp_val,
            "relevant_ssoc_minors":  "; ".join(relevant_ssoc),
            "n_relevant_job_skills": n_total,
            "n_exact_covered":       n_exact,
            "n_semantic_covered":    n_semantic,
            "n_genuine_gaps":        n_total - n_semantic,
            "baseline_scr":          round(n_exact / n_total, 4),
            "soft_scr":              round(sum_soft / n_total, 4),
            "module_type_filter":    module_type or "all",
            "knowledge_type_filter": knowledge_type or "all",
        })

    return pd.DataFrame(rows).sort_values("soft_scr", ascending=False).reset_index(drop=True)


def build_course_breakdown_ssoc_full(
    jobs:            pd.DataFrame,
    course_skills:   pd.DataFrame,
    nn_similarities: dict[str, float],
    nn_match_map:    dict[str, str],
    threshold:       float,
    group_col:       str,
    group_ssoc_map:  dict[str, list[str]],
    include_core:    bool = True,
) -> pd.DataFrame:
    """Runs all variants (with/without core × all/applied/theoretical)."""
    combos = [("all", None, None), ("applied", None, "applied"), ("theoretical", None, "theoretical")]
    if include_core:
        combos += [("all_core", "core", None), ("applied_core", "core", "applied"),
                   ("theo_core", "core", "theoretical")]

    merge_on = [group_col, "relevant_ssoc_minors", "n_relevant_job_skills"]
    base = None
    for label, mt, kt in combos:
        df = compute_course_breakdown_ssoc_aligned(
            jobs, course_skills, nn_similarities, nn_match_map,
            threshold, group_col, group_ssoc_map, module_type=mt, knowledge_type=kt,
        )
        df = df.rename(columns={
            "soft_scr":           f"soft_scr_{label}",
            "baseline_scr":       f"baseline_scr_{label}",
            "n_exact_covered":    f"n_exact_{label}",
            "n_semantic_covered": f"n_semantic_{label}",
            "n_genuine_gaps":     f"n_gaps_{label}",
        }).drop(columns=["module_type_filter", "knowledge_type_filter"], errors="ignore")

        keep = merge_on + [f"soft_scr_{label}", f"baseline_scr_{label}",
                           f"n_exact_{label}", f"n_semantic_{label}", f"n_gaps_{label}"]
        df = df[[c for c in keep if c in df.columns]]

        if base is None:
            base = df
        else:
            base = base.merge(df, on=merge_on, how="outer")

    return base


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Breakdown analysis across job categories, SSOC, faculty, major, department.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source",     type=str,   default="skills_list",
                   choices=["skills_list", "skillner"])
    p.add_argument("--job-skills",     type=Path,  default=None)
    p.add_argument("--course-skills",  type=Path,  default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--modules-meta",   type=Path,  default=DEFAULT_MODULES_CLEANED)
    p.add_argument("--degree-mapping", type=Path,  default=DEFAULT_DEGREE_MAPPING)
    p.add_argument("--major-ssoc-map", type=Path,  default=DEFAULT_MAJOR_SSOC_MAP)
    p.add_argument("--nn-path",        type=Path,  default=None)
    p.add_argument("--output",         type=Path,  default=None)
    p.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.nn_path is None:
        args.nn_path = (
            RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}"
            / "all_uncovered_nearest_neighbour.csv"
        )
    if args.output is None:
        args.output = RESULTS_DIR / f"breakdown_analysis_{args.job_source}"

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Job source : {args.job_source}")
    print(f"Threshold  : {args.threshold}")
    print(f"Output dir : {args.output}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    courses      = load_course_skills(args.course_skills)
    jobs         = load_job_skills_auto(args.job_skills, args.job_source)
    nn_sim, nn_match = load_nn_data(args.nn_path)
    degree_map    = load_degree_mapping(args.degree_mapping)
    modules_meta  = load_modules_cleaned(args.modules_meta)
    major_ssoc    = load_major_ssoc_map(args.major_ssoc_map)

    # Normalise faculty names in modules_cleaned to match degree_module_mapping
    _FACULTY_NORM = {
        "Arts and Social Science":      "Faculty of Arts and Social Sciences",
        "Computing":                    "School of Computing",
        "Law":                          "Faculty of Law",
        "Science":                      "Faculty of Science",
        "Yong Loo Lin Sch of Medicine": "Yong Loo Lin School of Medicine",
    }
    modules_meta["faculty"] = modules_meta["faculty"].replace(_FACULTY_NORM)

    print(f"  Course skills (unique): {courses['skill_canon'].nunique():,}")
    print(f"  Job skills    (unique): {jobs['skill_canon'].nunique():,}")
    print(f"  NN similarities       : {len(nn_sim):,}")

    # ── Attach module metadata to course skills ────────────────────────────────
    # From degree_module_mapping: major, degree_faculty, module_type
    courses_with_meta = courses.merge(
        degree_map[["module_code", "major", "degree_faculty", "module_type",
                    "module_department"]].drop_duplicates(),
        on="module_code", how="left",
    )
    # From modules_cleaned: department, faculty (fallback)
    courses_with_meta = courses_with_meta.merge(
        modules_meta[["module_code", "department", "faculty"]].drop_duplicates(),
        on="module_code", how="left",
    )
    # Use degree_module_mapping faculty preferentially
    courses_with_meta["faculty_final"] = courses_with_meta["degree_faculty"].fillna(
        courses_with_meta["faculty"]
    )
    courses_with_meta["dept_final"] = courses_with_meta["module_department"].fillna(
        courses_with_meta["department"]
    )

    print(f"  Unique majors   : {courses_with_meta['major'].nunique()}")
    print(f"  Unique faculties: {courses_with_meta['faculty_final'].nunique()}")
    print(f"  Unique depts    : {courses_with_meta['dept_final'].nunique()}")

    # ── Attach SSOC columns to job skills ─────────────────────────────────────
    # 03_jobs_filtered already has ssoc columns; merge on job_id
    raw_jobs = pd.read_csv(args.job_skills)
    ssoc_cols = ["job_id", "ssoc_submajor_title", "ssoc_minor_title"]
    available = [c for c in ssoc_cols if c in raw_jobs.columns]
    if len(available) > 1:
        jobs = jobs.merge(raw_jobs[available].drop_duplicates("job_id"),
                          on="job_id", how="left")

    # ── JOB-SIDE BREAKDOWNS ───────────────────────────────────────────────────
    print("\nComputing job-side breakdowns …")

    job_breakdowns = {}

    for group_col, label in [
        ("category",           "job_category"),
        ("ssoc_submajor_title","ssoc_submajor"),
        ("ssoc_minor_title",   "ssoc_minor"),
    ]:
        if group_col not in jobs.columns:
            print(f"  [skip] {group_col} not in jobs data")
            continue
        print(f"  → {label}")

        # All / applied / theoretical
        df_all   = compute_job_side_breakdown(jobs, courses, nn_sim, nn_match, args.threshold, group_col)
        df_app   = compute_job_side_breakdown(jobs, courses, nn_sim, nn_match, args.threshold, group_col, "applied")
        df_theo  = compute_job_side_breakdown(jobs, courses, nn_sim, nn_match, args.threshold, group_col, "theoretical")

        merged = df_all.merge(
            df_app[  [group_col, "soft_idf_scr"]].rename(columns={"soft_idf_scr": "soft_idf_scr_applied"}),
            on=group_col, how="left",
        ).merge(
            df_theo[[group_col, "soft_idf_scr"]].rename(columns={"soft_idf_scr": "soft_idf_scr_theoretical"}),
            on=group_col, how="left",
        )

        merged.to_csv(args.output / f"{label}.csv", index=False)
        job_breakdowns[label] = merged

        plot_breakdown(df_all, group_col, "soft_idf_scr",
                       f"Soft-IDF-SCR by {label.replace('_', ' ').title()}",
                       args.output / f"fig_{label}.png")
        print(f"    Saved: {label}.csv")

    # ── COURSE-SIDE BREAKDOWNS (SSOC-aligned denominator) ────────────────────
    print("\nComputing course-side breakdowns (SSOC-aligned denominator) …")

    # Derive faculty and department SSOC maps by rolling up from major→SSOC
    faculty_ssoc = derive_group_ssoc_map(degree_map, major_ssoc, "degree_faculty")
    dept_ssoc    = derive_group_ssoc_map(degree_map, major_ssoc, "module_department")

    print(f"  Faculty SSOC coverage : {len(faculty_ssoc)} faculties mapped")
    print(f"  Department SSOC coverage: {len(dept_ssoc)} departments mapped")

    course_breakdowns = {}

    for group_col, label, ssoc_map, include_core in [
        ("faculty_final", "faculty",    faculty_ssoc, False),
        ("major",         "major",      major_ssoc,   True),
        ("dept_final",    "department", dept_ssoc,    False),
    ]:
        cs = courses_with_meta[[
            "module_code", "skill_canon", "skill_type", "knowledge_type",
            "module_type", group_col,
        ]]
        cs = cs[cs[group_col].notna()].copy()

        if cs[group_col].nunique() == 0:
            print(f"  [skip] {group_col} has no data")
            continue
        print(f"  → {label} ({cs[group_col].nunique()} groups)")

        df = build_course_breakdown_ssoc_full(
            jobs, cs, nn_sim, nn_match, args.threshold,
            group_col, ssoc_map, include_core=include_core,
        )
        df.to_csv(args.output / f"{label}.csv", index=False)
        course_breakdowns[label] = df

        if include_core:
            plot_core_vs_max(df, group_col, args.output / f"fig_{label}_core_vs_max.png")
        plot_applied_vs_theoretical(df, group_col,
                                    args.output / f"fig_{label}_applied_vs_theoretical.png")
        print(f"    Saved: {label}.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "job_source": args.job_source,
        "threshold":  args.threshold,
        "job_side": {
            label: {
                "n_groups": len(df),
                "macro_soft_idf_scr": round(float(df["soft_idf_scr"].mean()), 4),
                "min_soft_idf_scr":   round(float(df["soft_idf_scr"].min()),  4),
                "max_soft_idf_scr":   round(float(df["soft_idf_scr"].max()),  4),
            }
            for label, df in job_breakdowns.items()
        },
        "course_side": {
            label: {
                "n_groups": len(df),
                "macro_soft_scr_max":  round(float(df["soft_scr_all"].mean()),      4),
                "macro_soft_scr_core": round(float(df["soft_scr_all_core"].mean()), 4) if "soft_scr_all_core" in df else None,
                "macro_soft_scr_applied": round(
                    float(df["soft_scr_applied"].mean()), 4) if "soft_scr_applied" in df else None,
            }
            for label, df in course_breakdowns.items()
        },
    }
    with (args.output / "summary_breakdown.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: summary_breakdown.json")

    print("\n" + "=" * 60)
    print("BREAKDOWN ANALYSIS — HEADLINE")
    print("=" * 60)
    for label, info in summary["job_side"].items():
        print(f"  Job-side {label:<20} macro Soft-IDF-SCR = {info['macro_soft_idf_scr']:.1%}")
    for label, info in summary["course_side"].items():
        max_cov  = info.get("macro_soft_scr_max",  0) or 0
        core_cov = info.get("macro_soft_scr_core")
        if core_cov is not None:
            print(f"  Course-side {label:<18} max={max_cov:.1%}  core={core_cov:.1%}")
        else:
            print(f"  Course-side {label:<18} max={max_cov:.1%}")
    print("=" * 60)
    print(f"\nOutputs written to: {args.output}")


if __name__ == "__main__":
    main()
