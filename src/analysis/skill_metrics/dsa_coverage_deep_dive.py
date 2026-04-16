"""
DSA Coverage Deep-Dive
----------------------
Measures how well DSA course skills cover the skills demanded in jobs that
DSA graduates actually go into, and produces a skill-level alignment table.

What this script does
---------------------
1. Loads DSA modules from degree_module_mapping.csv → DSA course skill set.
2. Identifies DSA-relevant jobs via major_ssoc_mapping.csv:
     major "Data Science and Analytics" maps to specific SSOC minor titles.
     Only job postings in those SSOC categories are considered.
3. Ranks job skills by raw demand frequency (posting count) within those jobs.
   IDF is intentionally NOT used here — we want to know which skills are most
   demanded in the jobs DSA graduates compete for, regardless of how
   specialised they are.
4. For every demanded skill, computes soft coverage against DSA's skill set:
     exact    — skill appears verbatim in DSA courses
     semantic — nearest DSA course skill has cosine similarity ≥ threshold
     gap      — not covered by DSA
5. Outputs an alignment table: each DSA course skill and which job skills it
   covers (both exact and semantic).

Why Soft-SCR (not Soft-IDF-SCR)?
---------------------------------
Soft-IDF-SCR is designed for job-side analysis: it down-weights skills that
appear across many job categories, amplifying category-specific signals.
Here we have already restricted to a specific set of job categories (DSA-
relevant SSOC groups), so IDF reweighting would compound the filtering and
obscure which skills are genuinely most demanded.
Soft-SCR (Variant B: continuous similarity, equal weights) is the right metric
for "how much of what DSA-relevant employers want does the DSA degree teach?".

Outputs
-------
results/dsa_coverage/
  ├── headline.json                      — Soft-SCR for DSA on relevant jobs
  ├── top_demanded_skills.csv            — all demanded skills ranked by freq,
  │                                        with coverage status and DSA match
  ├── skill_alignment_course_to_job.csv  — DSA skill → job skills it covers
  ├── skill_alignment_job_to_course.csv  — covered job skill → DSA skill matched
  ├── gap_skills.csv                     — uncovered skills ranked by demand freq
  └── fig_top_demanded.png               — bar chart of top skills, coloured
                                           by coverage status

Usage
-----
    python -m src.analysis.skill_metrics.dsa_coverage_deep_dive
    python -m src.analysis.skill_metrics.dsa_coverage_deep_dive --threshold 0.72
    python -m src.analysis.skill_metrics.dsa_coverage_deep_dive --module-type core
    python -m src.analysis.skill_metrics.dsa_coverage_deep_dive --top-n 40
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

# ── Path resolution ───────────────────────────────────────────────────────────
try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RAW_DIR, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RAW_DIR, RESULTS_DIR

from src.analysis.skill_metrics.baseline_scr import canonicalize, load_course_skills

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DEGREE_MAPPING  = COURSES_PROCESSED_DIR / "degree_module_mapping.csv"
DEFAULT_COURSE_SKILLS   = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_JOB_PATH        = JOBS_PROCESSED_DIR    / "03_jobs_filtered.csv"
DEFAULT_SSOC_MAP        = RAW_DIR / "major_ssoc_mapping.csv"
DEFAULT_NN_PATH         = RESULTS_DIR / "vocabulary_mismatch_skills_list" / "all_uncovered_nearest_neighbour.csv"
DEFAULT_OUTPUT_DIR      = RESULTS_DIR / "dsa_coverage"
DEFAULT_THRESHOLD       = 0.72
DSA_DEGREE_NAME         = "Data Science and Analytics"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dsa_modules(degree_mapping_path: Path, module_type: str | None) -> set[str]:
    """Return module codes for the DSA degree (optionally filtered by type)."""
    df = pd.read_csv(degree_mapping_path)
    df.columns      = df.columns.str.strip()
    df["module_code"] = df["module_code"].astype(str).str.strip()
    df["major"]       = df["major"].astype(str).str.strip()
    df["module_type"] = df["module_type"].astype(str).str.strip().str.lower()

    mask = df["major"] == DSA_DEGREE_NAME
    if module_type:
        mask &= df["module_type"] == module_type.lower()

    codes = set(df.loc[mask, "module_code"].unique())
    if not codes:
        raise ValueError(
            f"No modules found for degree '{DSA_DEGREE_NAME}' "
            f"(module_type={module_type!r}).\n"
            f"Available majors: {sorted(df['major'].unique())}"
        )
    return codes


def load_dsa_relevant_ssoc(ssoc_map_path: Path) -> list[str]:
    """
    Return the SSOC minor titles that DSA graduates are mapped to.

    These come from major_ssoc_mapping.csv which maps each major to the
    SSOC minor group(s) its graduates typically enter.
    """
    df = pd.read_csv(ssoc_map_path)
    df.columns = df.columns.str.strip().str.strip('"')
    df["major"] = df["major"].astype(str).str.strip().str.strip('"')
    dsa_rows = df[df["major"] == DSA_DEGREE_NAME]
    ssoc_col = [c for c in df.columns if "ssoc" in c.lower()][0]
    titles = dsa_rows[ssoc_col].dropna().str.strip().str.strip('"').tolist()
    if not titles:
        raise ValueError(
            f"No SSOC mapping found for '{DSA_DEGREE_NAME}' in {ssoc_map_path}."
        )
    return titles


def load_relevant_jobs(job_path: Path, relevant_ssoc: list[str]) -> pd.DataFrame:
    """
    Load 03_jobs_filtered.csv, restrict to DSA-relevant SSOC categories,
    parse skills_list, and return one row per (job_id, skill_canon).
    """
    df = pd.read_csv(job_path, usecols=["job_id", "skills_list", "ssoc_minor_title", "category"])
    df = df[df["ssoc_minor_title"].isin(relevant_ssoc)].dropna(subset=["skills_list"])

    # Explode comma-separated skills
    df["skill"] = df["skills_list"].str.split(",")
    df = df.explode("skill")
    df["skill"] = df["skill"].str.strip()
    df = df[df["skill"].astype(bool)].copy()
    df["skill_canon"] = df["skill"].map(canonicalize)
    df = df[df["skill_canon"].astype(bool)].copy()

    return df[["job_id", "skill", "skill_canon", "ssoc_minor_title", "category"]].reset_index(drop=True)


def load_nn(nn_path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Load pre-computed nearest-neighbour table → (sim_map, match_map)."""
    if not nn_path.exists():
        raise FileNotFoundError(
            f"Nearest-neighbour file not found: {nn_path}\n"
            "Run pipeline_build_metric.py first."
        )
    nn_df     = pd.read_csv(nn_path)
    sim_map   = dict(zip(nn_df["job_skill"], nn_df["similarity"]))
    match_map = dict(zip(nn_df["job_skill"], nn_df["course_skill_match"]))
    return sim_map, match_map


# ─────────────────────────────────────────────────────────────────────────────
# Coverage scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_skill(
    job_skill:    str,
    dsa_skill_set: set[str],
    nn_sim_map:   dict[str, float],
    nn_match_map: dict[str, str],
    threshold:    float,
) -> tuple[float, str, str]:
    """
    Return (soft_score, match_type, matched_dsa_skill) for one job skill.

    match_type: "exact" | "semantic" | "gap"
    matched_dsa_skill: the DSA course skill it aligns to (empty string for gaps)

    Soft score:
      1.0       — exact match
      sim ≥ θ   — semantic match where nearest course skill is a DSA skill
      0.0       — gap
    """
    if job_skill in dsa_skill_set:
        return 1.0, "exact", job_skill

    sim   = nn_sim_map.get(job_skill, 0.0)
    match = nn_match_map.get(job_skill, "")

    if sim >= threshold and match in dsa_skill_set:
        return sim, "semantic", match

    return 0.0, "gap", ""


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def run_coverage(
    jobs:         pd.DataFrame,
    dsa_skill_set: set[str],
    nn_sim_map:   dict[str, float],
    nn_match_map: dict[str, str],
    threshold:    float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute DSA coverage over the (already filtered) relevant job postings.

    Returns
    -------
    headline       : overall Soft-SCR and counts
    demanded       : all unique job skills ranked by freq, with coverage info
    course_to_job  : DSA skill → job skills it covers
    job_to_course  : covered job skill → DSA skill matched
    """
    # ── Demand frequency per unique skill ─────────────────────────────────────
    freq = jobs.groupby("skill_canon")["job_id"].nunique().rename("job_freq")

    # ── Score every unique job skill ──────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for js, job_freq in freq.items():
        soft, mtype, matched = score_skill(
            js, dsa_skill_set, nn_sim_map, nn_match_map, threshold
        )
        rows.append({
            "job_skill":     js,
            "job_freq":      int(job_freq),
            "match_type":    mtype,
            "soft_score":    round(soft,    4),
            "matched_dsa":   matched,
        })

    demanded = (
        pd.DataFrame(rows)
        .sort_values("job_freq", ascending=False)
        .reset_index(drop=True)
    )

    # ── Headline Soft-SCR (equal weights over unique skills) ──────────────────
    scores = demanded["soft_score"].values
    soft_scr = float(scores.mean()) if len(scores) else 0.0

    n_exact    = int((demanded["match_type"] == "exact").sum())
    n_semantic = int((demanded["match_type"] == "semantic").sum())
    n_gap      = int((demanded["match_type"] == "gap").sum())
    n_total    = len(demanded)

    headline: dict[str, Any] = {
        "degree":              DSA_DEGREE_NAME,
        "threshold":           threshold,
        "n_dsa_course_skills": len(dsa_skill_set),
        "n_unique_job_skills": n_total,
        "n_exact_covered":     n_exact,
        "n_semantic_covered":  n_semantic,
        "n_genuine_gaps":      n_gap,
        "pct_exact":           round(n_exact    / n_total, 4) if n_total else 0,
        "pct_semantic":        round(n_semantic / n_total, 4) if n_total else 0,
        "pct_gap":             round(n_gap      / n_total, 4) if n_total else 0,
        "soft_scr":            round(soft_scr, 4),
        "metric_note":         (
            "Soft-SCR (Variant B): continuous similarity, equal weights over "
            "unique skills in DSA-relevant SSOC job categories."
        ),
    }

    # ── Skill alignment: DSA skill → job skills it covers ─────────────────────
    skill_info = demanded.set_index("job_skill").to_dict("index")

    c2j_rows: list[dict[str, Any]] = []
    for dsa_skill in sorted(dsa_skill_set):
        aligned_exact = [
            js for js, info in skill_info.items()
            if info["match_type"] == "exact" and info["matched_dsa"] == dsa_skill
        ]
        aligned_sem = [
            js for js, info in skill_info.items()
            if info["match_type"] == "semantic" and info["matched_dsa"] == dsa_skill
        ]
        all_aligned = aligned_exact + aligned_sem
        total_freq  = sum(skill_info[js]["job_freq"] for js in all_aligned)

        # Annotate each aligned job skill with its frequency for readability
        def _annotate(skill_list: list[str]) -> str:
            return " | ".join(
                f"{js} (freq={skill_info[js]['job_freq']})"
                for js in sorted(skill_list, key=lambda s: -skill_info[s]["job_freq"])
            )

        c2j_rows.append({
            "dsa_skill":             dsa_skill,
            "n_job_skills_covered":  len(all_aligned),
            "n_exact":               len(aligned_exact),
            "n_semantic":            len(aligned_sem),
            "total_job_freq":        total_freq,
            "exact_job_skills":      _annotate(aligned_exact),
            "semantic_job_skills":   _annotate(aligned_sem),
        })

    course_to_job = (
        pd.DataFrame(c2j_rows)
        .sort_values("total_job_freq", ascending=False)
        .reset_index(drop=True)
    )

    # ── Skill alignment: covered job skill → DSA skill ────────────────────────
    j2c_rows = [
        {
            "job_skill":         row["job_skill"],
            "job_freq":          row["job_freq"],
            "match_type":        row["match_type"],
            "soft_score":        row["soft_score"],
            "matched_dsa_skill": row["matched_dsa"],
        }
        for _, row in demanded.iterrows()
        if row["match_type"] != "gap"
    ]
    job_to_course = (
        pd.DataFrame(j2c_rows)
        .sort_values("job_freq", ascending=False)
        .reset_index(drop=True)
    )

    return headline, demanded, course_to_job, job_to_course


def build_top_demanded_report_table(demanded: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Build a report-ready Top-N demanded skills table.

    Columns:
      job_skill, freq, status, score, matched_dsa_skill

    status:
      EXACT      -> exact string match
      NON-EXACT  -> semantic match or gap

    score:
      1.0 for exact matches, otherwise embedding-based soft score.
    """
    top = demanded.head(top_n).copy()
    top["status"] = np.where(top["match_type"] == "exact", "EXACT", "NON-EXACT")
    top["score"] = np.where(top["match_type"] == "exact", 1.0, top["soft_score"])
    top["matched_dsa_skill"] = top["matched_dsa"].replace("", "—")

    out = top.rename(columns={"job_freq": "freq"})
    return out[["job_skill", "freq", "status", "score", "matched_dsa_skill"]]


def build_top_dsa_semantic_report_table(course_to_job: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Build Top-N DSA skills ranked by semantic alignment to demanded job skills.

    Keeps only DSA skills with at least one semantic match.
    """
    x = course_to_job[course_to_job["n_semantic"] > 0].copy()
    if x.empty:
        return pd.DataFrame(
            columns=["dsa_skill", "n_semantic_matches", "total_job_freq", "semantic_job_skills"]
        )
    x = x.sort_values(["n_semantic", "total_job_freq"], ascending=[False, False]).head(top_n)
    return x.rename(columns={"n_semantic": "n_semantic_matches"})[
        ["dsa_skill", "n_semantic_matches", "total_job_freq", "semantic_job_skills"]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_COLOURS = {
    "exact":    "#27ae60",   # green
    "semantic": "#3498db",   # blue
    "gap":      "#e74c3c",   # red
}


def plot_top_demanded(
    demanded:   pd.DataFrame,
    threshold:  float,
    output_path: Path,
    top_n:      int = 40,
) -> None:
    """
    Horizontal bar chart of the top-N demanded skills in DSA-relevant jobs,
    coloured by coverage status (exact / semantic / gap).
    Bars are sorted by demand frequency descending (most demanded at top).
    """
    df_plot = demanded.head(top_n).sort_values("job_freq", ascending=True)
    colours = [_COLOURS[m] for m in df_plot["match_type"]]

    fig, ax = plt.subplots(figsize=(13, max(8, len(df_plot) * 0.38)))

    bars = ax.barh(
        df_plot["job_skill"], df_plot["job_freq"],
        color=colours, edgecolor="white", height=0.72,
    )
    for bar, row in zip(bars, df_plot.itertuples()):
        label = (
            f"exact → {row.matched_dsa}"
            if row.match_type == "exact"
            else f"~{row.matched_dsa}  (sim={row.soft_score:.2f})"
            if row.match_type == "semantic"
            else "GAP"
        )
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left", fontsize=6.5, color="#2c3e50",
        )

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=_COLOURS["exact"],    label="Exact match"),
        Patch(color=_COLOURS["semantic"], label=f"Semantic match (θ={threshold})"),
        Patch(color=_COLOURS["gap"],      label="Gap — not covered by DSA"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    ax.set_xlabel("Number of job postings demanding this skill", fontsize=10)
    ax.set_title(
        f"Top {top_n} Skills Demanded in DSA-Relevant Jobs\n"
        f"Coloured by DSA course coverage  (θ = {threshold})",
        fontsize=11, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    headline:     dict[str, Any],
    demanded:     pd.DataFrame,
    course_to_job: pd.DataFrame,
    relevant_ssoc: list[str],
    top_n:        int = 30,
) -> None:
    h   = headline
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  DSA COVERAGE DEEP-DIVE — {h['degree']}")
    print(sep)
    print(f"  Relevant SSOC categories  : {len(relevant_ssoc)}")
    for s in relevant_ssoc:
        print(f"    • {s}")
    print()
    print(f"  DSA course skills (unique) : {h['n_dsa_course_skills']:>5,}")
    print(f"  Demanded job skills        : {h['n_unique_job_skills']:>5,}")
    print()
    print(f"  Exact match                : {h['n_exact_covered']:>5,}  ({h['pct_exact']:.1%})")
    print(f"  Semantic match (θ={h['threshold']})    : {h['n_semantic_covered']:>5,}  ({h['pct_semantic']:.1%})")
    print(f"  Genuine gap                : {h['n_genuine_gaps']:>5,}  ({h['pct_gap']:.1%})")
    print()
    print(f"  Soft-SCR                   : {h['soft_scr']:.1%}")
    print(sep)

    print(f"\n{'─'*72}")
    print(f"  TOP {top_n} DEMANDED SKILLS — coverage status")
    print(f"{'─'*72}")
    report_top = build_top_demanded_report_table(demanded, top_n=top_n)
    print(f"  {'Rank':<5} {'Job skill':<35} {'Freq':>5}  {'Status':<10} {'Score':>6}  Matched DSA skill")
    print(f"  {'─'*4} {'─'*35} {'─'*5}  {'─'*10} {'─'*6}  {'─'*30}")
    for rank, row in report_top.iterrows():
        print(
            f"  {rank+1:<5} {row['job_skill']:<35} {row['freq']:>5}  "
            f"{row['status']:<10} {row['score']:>6.2f}  {row['matched_dsa_skill']}"
        )

    print(f"\n{'─'*72}")
    print(f"  TOP {top_n} DSA SKILLS — semantic matches to demanded job skills")
    print(f"{'─'*72}")
    dsa_sem_top = build_top_dsa_semantic_report_table(course_to_job, top_n=top_n)
    if dsa_sem_top.empty:
        print("  No semantic matches found.")
    else:
        print(f"  {'Rank':<5} {'DSA skill':<30} {'#Sem':>5} {'Total freq':>11}  Semantic job skills")
        print(f"  {'─'*4} {'─'*30} {'─'*5} {'─'*11}  {'─'*30}")
        for rank, row in dsa_sem_top.iterrows():
            print(
                f"  {rank+1:<5} {row['dsa_skill']:<30} {int(row['n_semantic_matches']):>5} "
                f"{int(row['total_job_freq']):>11}  {row['semantic_job_skills']}"
            )

    print(f"\n{'─'*72}")
    print(f"  DSA SKILLS AND THEIR ALIGNED JOB SKILLS (sorted by demand coverage)")
    print(f"{'─'*72}")
    covered = course_to_job[course_to_job["n_job_skills_covered"] > 0]
    for _, row in covered.iterrows():
        print(
            f"\n  [{row['total_job_freq']:4d} postings, "
            f"{row['n_job_skills_covered']} job skill(s)]  "
            f"DSA skill: {row['dsa_skill']}"
        )
        if row["exact_job_skills"]:
            print(f"    exact   : {row['exact_job_skills']}")
        if row["semantic_job_skills"]:
            print(f"    semantic: {row['semantic_job_skills']}")

    print(f"\n{'─'*72}")
    n_no_coverage = (course_to_job["n_job_skills_covered"] == 0).sum()
    print(f"  DSA skills with zero alignment to demanded job skills: {n_no_coverage}")
    print(f"  (These are valid academic skills not listed by employers in these SSOC roles)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DSA coverage deep-dive: how well DSA skills cover demanded job skills.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--degree-mapping", type=Path, default=DEFAULT_DEGREE_MAPPING)
    p.add_argument("--course-skills",  type=Path, default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--job-path",       type=Path, default=DEFAULT_JOB_PATH)
    p.add_argument("--ssoc-map",       type=Path, default=DEFAULT_SSOC_MAP)
    p.add_argument("--nn-path",        type=Path, default=DEFAULT_NN_PATH)
    p.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Cosine similarity threshold for semantic near-match",
    )
    p.add_argument(
        "--module-type", type=str, default=None,
        choices=["core", "elective"],
        help="Restrict to core or elective modules only (default: all)",
    )
    p.add_argument(
        "--top-n", type=int, default=5,
        help="Number of rows to show in console tables and figure",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"Degree       : {DSA_DEGREE_NAME}")
    print(f"Module type  : {args.module_type or 'all (core + elective)'}")
    print(f"Threshold    : {args.threshold}")
    print(f"Output dir   : {out}\n")

    # ── DSA modules → course skill set ───────────────────────────────────────
    print("Loading DSA modules …")
    dsa_modules = load_dsa_modules(args.degree_mapping, args.module_type)
    print(f"  DSA modules: {len(dsa_modules)}")

    print("Loading course skills …")
    all_courses   = load_course_skills(args.course_skills)
    dsa_courses   = all_courses[all_courses["module_code"].isin(dsa_modules)]
    dsa_skill_set = set(dsa_courses["skill_canon"].unique())
    print(f"  DSA course skills : {len(dsa_skill_set):,}  "
          f"(from {dsa_courses['module_code'].nunique()} modules with skills)")

    # ── DSA-relevant SSOC categories ─────────────────────────────────────────
    print("Loading SSOC mapping …")
    relevant_ssoc = load_dsa_relevant_ssoc(args.ssoc_map)
    print(f"  Relevant SSOC categories ({len(relevant_ssoc)}):")
    for s in relevant_ssoc:
        print(f"    • {s}")

    # ── Job skills filtered to relevant SSOC ─────────────────────────────────
    print("\nLoading job postings (filtered to DSA-relevant SSOC) …")
    jobs = load_relevant_jobs(args.job_path, relevant_ssoc)
    n_postings = jobs["job_id"].nunique()
    print(f"  Job postings       : {n_postings:,}")
    print(f"  Job skill rows     : {len(jobs):,}")
    print(f"  Unique job skills  : {jobs['skill_canon'].nunique():,}")

    # ── Nearest-neighbour data ────────────────────────────────────────────────
    print("\nLoading nearest-neighbour similarities …")
    nn_sim_map, nn_match_map = load_nn(args.nn_path)
    print(f"  Loaded {len(nn_sim_map):,} pre-computed NN similarities")

    # ── Coverage computation ──────────────────────────────────────────────────
    print("\nComputing DSA coverage …")
    headline, demanded, course_to_job, job_to_course = run_coverage(
        jobs, dsa_skill_set, nn_sim_map, nn_match_map, args.threshold
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs …")

    with (out / "headline.json").open("w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)
    print("  Saved: headline.json")

    demanded.to_csv(out / "top_demanded_skills.csv", index=False)
    print("  Saved: top_demanded_skills.csv")

    report_top = build_top_demanded_report_table(demanded, top_n=args.top_n)
    report_top_path = out / f"top{args.top_n}_demanded_skills_report.csv"
    report_top.to_csv(report_top_path, index=False)
    print(f"  Saved: {report_top_path.name}")

    dsa_sem_top = build_top_dsa_semantic_report_table(course_to_job, top_n=args.top_n)
    dsa_sem_top_path = out / f"top{args.top_n}_dsa_skills_semantic_to_job_skills.csv"
    dsa_sem_top.to_csv(dsa_sem_top_path, index=False)
    print(f"  Saved: {dsa_sem_top_path.name}")

    course_to_job.to_csv(out / "skill_alignment_course_to_job.csv", index=False)
    print("  Saved: skill_alignment_course_to_job.csv")

    job_to_course.to_csv(out / "skill_alignment_job_to_course.csv", index=False)
    print("  Saved: skill_alignment_job_to_course.csv")

    gaps = demanded[demanded["match_type"] == "gap"].copy()
    gaps.to_csv(out / "gap_skills.csv", index=False)
    print("  Saved: gap_skills.csv")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    plot_top_demanded(demanded, args.threshold, out / "fig_top_demanded.png", top_n=args.top_n)

    # ── Console report ────────────────────────────────────────────────────────
    print_report(headline, demanded, course_to_job, relevant_ssoc, top_n=args.top_n)

    print(f"\n  Outputs written to: {out}\n")


if __name__ == "__main__":
    main()
