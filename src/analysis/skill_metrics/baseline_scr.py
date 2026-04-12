"""
Baseline Skill Coverage Rate (SCR) — Step 1 of the coverage pipeline.

What this script does
---------------------
Measures how well NUS course skills cover the skills demanded by Singapore
employers using *exact string matching* after canonicalization.

This is deliberately the simplest possible metric. Its purpose is twofold:
  1. Establish a reproducible baseline to compare all improved metrics against.
  2. Expose the limitations of exact matching (synonym misses, surface-form
     variation) that motivate the probabilistic approach in later steps.

Metric definitions
------------------
SCR (Skill Coverage Rate):
    Of the unique skills demanded in a job category, what fraction appear
    anywhere in NUS course descriptions?

        SCR_c = |covered_c| / |unique_job_skills_c|

    where  covered_c = unique_job_skills_c  ∩  course_skill_set

DW-SCR (Demand-Weighted SCR):
    SCR weighted by how often each skill is mentioned across job postings.
    Skills demanded in many ads count proportionally more.

        DW-SCR_c = Σ_{j∈covered_c} freq(j,c)  /  Σ_{j∈all_c} freq(j,c)

Hard-skill vs soft-skill SCR:
    Computed separately to isolate the soft-skill description artefact:
    course descriptions rarely mention soft skills (teamwork, communication)
    even when they are taught, so soft-skill SCR is systematically
    understated and should not be conflated with the hard-skill signal.

Outputs
-------
results/baseline_scr/
  ├── per_category_all.csv        — full metrics for all skills
  ├── per_category_hard.csv       — hard skills only
  ├── per_category_soft.csv       — soft skills only
  ├── summary.json                — headline numbers
  ├── fig_scr_by_category.png     — horizontal bar chart (SCR + DW-SCR)
  └── fig_hard_vs_soft_scr.png    — hard vs soft coverage comparison

Usage
-----
    python -m src.analysis.baseline_scr
    python -m src.analysis.baseline_scr --output results/my_baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Resolve repo root whether invoked as a module or run directly ─────────────
try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

# ── Default I/O paths ─────────────────────────────────────────────────────────
DEFAULT_JOB_SKILLS    = JOBS_PROCESSED_DIR    / "job_skill_pair_skillner.csv"
DEFAULT_COURSE_SKILLS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_OUTPUT_DIR    = RESULTS_DIR / "baseline_scr"

# ─────────────────────────────────────────────────────────────────────────────
# Canonicalization
# ─────────────────────────────────────────────────────────────────────────────
# We normalise every skill label through the same deterministic pipeline so
# that surface-form variation ("Python programming" vs "python") does not
# create artificial mismatches.  This is the *maximum* cleaning we apply in
# the baseline — more aggressive fuzzy matching comes in later steps.

_PUNC_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_LEADING_WRAPPER_RE = re.compile(
    r"^(knowledge of|experience in|experience with|ability to|"
    r"proficiency in|understanding of|familiarity with)\s+",
    flags=re.IGNORECASE,
)
_TRAILING_SUFFIX_RE = re.compile(
    r"\s+(skills?|programming language|tool|tools|framework|frameworks|"
    r"methodology|methodologies|technique|techniques)$",
    flags=re.IGNORECASE,
)

# Hard-coded synonyms for the most common surface variants.
# Keys and values are already in the normalised form produced by steps 1–4.
_SYNONYM_MAP: dict[str, str] = {
    "ms excel":                        "microsoft excel",
    "ms word":                         "microsoft word",
    "ms powerpoint":                   "microsoft powerpoint",
    "python programming":              "python",
    "python programming language":     "python",
    "r programming":                   "r",
    "r programming language":          "r",
    "machine learning ml":             "machine learning",
    "artificial intelligence ai":      "artificial intelligence",
    "natural language processing nlp": "natural language processing",
    "structured query language sql":   "sql",
    "creative thinking":               "creativity",
    "critical thinking skills":        "critical thinking",
    "problem solving":                 "problem-solving",
}


def canonicalize(skill: str) -> str:
    """
    Normalise a raw skill label to a canonical matching form.

    Pipeline
    --------
    1. Lowercase + strip surrounding whitespace.
    2. Replace hyphens with spaces  ("problem-solving" → "problem solving").
    3. Remove non-alphanumeric punctuation.
    4. Collapse multiple spaces to one.
    5. Iteratively strip leading wrappers  ("knowledge of X" → "X").
    6. Iteratively strip trailing generic suffixes  ("X skills" → "X").
    7. Look up synonym map for known surface variants.

    Returns "" for null/empty inputs so callers can filter with ``bool()``.
    """
    if pd.isna(skill) or not str(skill).strip():
        return ""

    s = str(skill).strip().lower()
    s = s.replace("-", " ")
    s = _PUNC_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()

    # Iterative stripping handles nested cases, e.g.
    # "knowledge of machine learning skills" → "machine learning"
    prev = None
    while prev != s:
        prev = s
        s = _LEADING_WRAPPER_RE.sub("", s).strip()
        s = _TRAILING_SUFFIX_RE.sub("", s).strip()

    return _SYNONYM_MAP.get(s, s)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_course_skills(path: Path) -> pd.DataFrame:
    """
    Load module_skill_pairs.csv and canonicalize skill labels.

    Handles a known formatting quirk: some generated versions of this file
    have the filename itself as an extra first row before the real header.
    """
    df = pd.read_csv(path)
    required = {"module_code", "skill_label", "skill_type", "knowledge_type"}

    if not required.issubset(df.columns):
        # Try skipping the spurious first row
        df = pd.read_csv(path, skiprows=1)

    if not required.issubset(df.columns):
        raise ValueError(
            f"Cannot find required columns {sorted(required)} in {path}.\n"
            f"Found: {list(df.columns)}"
        )

    df["skill_canon"] = df["skill_label"].map(canonicalize)
    # Drop rows where canonicalization produced an empty string
    return df[df["skill_canon"].astype(bool)].copy()


def load_job_skills(path: Path) -> pd.DataFrame:
    """
    Load job_skill_pair_skillner.csv and canonicalize skill labels.

    Expected columns: job_id, skill, category, skill_type, knowledge_type.
    """
    df = pd.read_csv(path)
    required = {"job_id", "skill", "category", "skill_type", "knowledge_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Job skills file is missing columns: {sorted(missing)}.\n"
            f"Found: {list(df.columns)}"
        )

    df["skill_canon"] = df["skill"].map(canonicalize)
    # Drop rows with empty skills or missing category
    df = df[df["skill_canon"].astype(bool) & df["category"].notna()].copy()
    df["skill_type"] = df["skill_type"].str.strip().str.lower()
    return df


def load_job_skills_from_list(path: Path) -> pd.DataFrame:
    """
    Load job skills from the ``skills_list`` column of 01b_jobs_cleaned.csv.

    These are employer-provided skills — explicitly tagged in the job posting
    rather than NLP-extracted.  The list is a comma-separated string per row.

    Because this source has no skill_type / knowledge_type labelling, both
    columns default to "unknown".  All downstream scripts treat "unknown"
    the same as unlabelled and skip hard/soft filtering unless explicitly
    overridden.

    Returns a DataFrame in the same structure as ``load_job_skills()`` so
    that all downstream scripts are source-agnostic.
    """
    df = pd.read_csv(path)
    required = {"job_id", "skills_list", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Job skills file is missing columns: {sorted(missing)}.\n"
            f"Found: {list(df.columns)}"
        )

    df = df[["job_id", "skills_list", "category"]].dropna(subset=["skills_list", "category"])

    # Parse comma-separated skill strings into individual skills
    df["skill"] = df["skills_list"].str.split(",")
    df = df.explode("skill")
    df["skill"] = df["skill"].str.strip()
    df = df[df["skill"].astype(bool)]  # drop empty strings

    # Add placeholder label columns so downstream scripts work unchanged
    df["skill_type"]     = "unknown"
    df["knowledge_type"] = "unknown"

    df["skill_canon"] = df["skill"].map(canonicalize)
    df = df[df["skill_canon"].astype(bool) & df["category"].notna()].copy()
    return df[["job_id", "skill", "skill_canon", "category",
               "skill_type", "knowledge_type"]].reset_index(drop=True)


def load_job_skills_auto(
    path: Path,
    source: str,
) -> pd.DataFrame:
    """
    Dispatch to the correct loader based on ``source``.

    Parameters
    ----------
    path   : Path to the job skills file.
    source : ``"skillner"``    → job_skill_pair_skillner.csv (one skill per row)
             ``"skills_list"`` → 01b_jobs_cleaned.csv (comma-separated per row)

    This function is the single entry point used by all downstream scripts
    so that switching data sources requires only a ``--job-source`` flag.
    """
    if source == "skillner":
        return load_job_skills(path)
    if source == "skills_list":
        return load_job_skills_from_list(path)
    raise ValueError(f"Unknown job source: {source!r}. Use 'skillner' or 'skills_list'.")


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_scr(
    job_skills: pd.DataFrame,
    course_skill_set: set[str],
    skill_type_filter: str | None = None,
) -> pd.DataFrame:
    """
    Compute SCR and DW-SCR for every job category.

    Parameters
    ----------
    job_skills : DataFrame
        Output of ``load_job_skills()``.  Must contain columns
        ``skill_canon``, ``category``, ``skill_type``.
    course_skill_set : set[str]
        Canonicalized course skills to match against.
    skill_type_filter : "hard" | "soft" | None
        If given, only consider job skills of this type.  None → all skills.

    Returns
    -------
    DataFrame sorted by SCR descending, one row per category, with columns:
        category, n_unique_job_skills, n_covered, n_uncovered, scr,
        n_job_mentions, n_covered_mentions, dw_scr, top_10_uncovered
    """
    df = job_skills.copy()
    if skill_type_filter:
        df = df[df["skill_type"] == skill_type_filter]

    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for category, grp in df.groupby("category"):
        unique_skills = set(grp["skill_canon"].unique())
        covered   = unique_skills & course_skill_set
        uncovered = unique_skills - course_skill_set

        n_job = len(unique_skills)
        n_cov = len(covered)
        scr   = n_cov / n_job if n_job > 0 else 0.0

        # Demand-weight: count how often each skill appears across postings
        # in this category.  More mentions = higher demand weight.
        freq_series   = grp["skill_canon"].value_counts()
        total_freq    = int(freq_series.sum())
        covered_freq  = int(freq_series[freq_series.index.isin(covered)].sum())
        dw_scr        = covered_freq / total_freq if total_freq > 0 else 0.0

        # Top uncovered skills by demand frequency — the gap list for MOE
        uncov_series = freq_series[freq_series.index.isin(uncovered)]
        top_uncov = [
            {"skill": skill, "freq": int(freq)}
            for skill, freq in uncov_series.head(10).items()
        ]

        rows.append({
            "category":            category,
            "n_unique_job_skills": n_job,
            "n_covered":           n_cov,
            "n_uncovered":         n_job - n_cov,
            "scr":                 round(scr,    4),
            "n_job_mentions":      total_freq,
            "n_covered_mentions":  covered_freq,
            "dw_scr":              round(dw_scr, 4),
            "top_10_uncovered":    top_uncov,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("scr", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Coverage tier thresholds and colours (used consistently across all figures)
_TIER_HIGH   = 0.40   # ≥ 40 % — reasonably well covered
_TIER_MEDIUM = 0.20   # 20–40 % — partial coverage
# below 20 % — significant gap

_COLOURS = {
    "high":   "#27ae60",   # green
    "medium": "#e67e22",   # amber
    "low":    "#e74c3c",   # red
    "scr":    "#2980b9",   # blue  (used for SCR bars in grouped charts)
    "dw_scr": "#8e44ad",   # purple (DW-SCR)
}


def _tier_colour(scr: float) -> str:
    """Return a tier colour based on the SCR value."""
    if scr >= _TIER_HIGH:
        return _COLOURS["high"]
    if scr >= _TIER_MEDIUM:
        return _COLOURS["medium"]
    return _COLOURS["low"]


def _tier_legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=_COLOURS["high"],   label="≥ 40 % — well covered"),
        mpatches.Patch(color=_COLOURS["medium"], label="20–40 % — partial coverage"),
        mpatches.Patch(color=_COLOURS["low"],    label="< 20 % — significant gap"),
    ]


def plot_scr_by_category(df: pd.DataFrame, output_path: Path) -> None:
    """
    Side-by-side horizontal bar charts of SCR and DW-SCR per job category.

    Categories are sorted by SCR ascending so the highest-coverage category
    appears at the top.  Bars are colour-coded by coverage tier.  A dashed
    vertical line marks the macro-average for each metric.

    Intended audience: MOE officers — axis labels use plain percentages.
    """
    if df.empty:
        print("  [skip] No data for SCR-by-category chart.")
        return

    # Sort ascending so highest coverage appears at the top of the chart
    df_plot = df.sort_values("scr", ascending=True)
    n_cats  = len(df_plot)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, max(8, n_cats * 0.38)),
        sharey=True,
    )
    fig.suptitle(
        "NUS Curriculum Skill Coverage by Singapore Job Category\n"
        "(Baseline: exact string match after canonicalization)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, metric, label in [
        (axes[0], "scr",    "SCR — equal-weighted"),
        (axes[1], "dw_scr", "DW-SCR — demand-weighted"),
    ]:
        colours = [_tier_colour(v) for v in df_plot[metric]]
        bars    = ax.barh(
            df_plot["category"], df_plot[metric],
            color=colours, edgecolor="white", height=0.72,
        )

        # Percentage labels at end of each bar
        for bar, val in zip(bars, df_plot[metric]):
            ax.text(
                bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center", ha="left", fontsize=7.5, color="#2c3e50",
            )

        # Macro-average reference line
        macro = df_plot[metric].mean()
        ax.axvline(
            macro, color="#2c3e50", linestyle="--", linewidth=1.3,
            label=f"Macro avg: {macro:.1%}",
        )

        ax.set_xlabel(label, fontsize=10)
        ax.set_xlim(0, min(1.0, df_plot[metric].max() + 0.15))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

    # Shared colour-tier legend at the bottom
    fig.legend(
        handles=_tier_legend_patches(),
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9, frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_hard_vs_soft(
    df_all: pd.DataFrame,
    df_hard: pd.DataFrame,
    df_soft: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Grouped bar chart: overall / hard / soft SCR for the top-N categories.

    The top-N categories are chosen by total job-skill mention volume so we
    focus on the parts of the job market with the most data.

    This chart makes the soft-skill description artefact visible: course
    descriptions systematically under-mention soft skills, so soft-skill SCR
    is an artefact of how courses are written, not a curriculum gap.
    """
    if df_all.empty or df_hard.empty:
        print("  [skip] Insufficient data for hard-vs-soft chart.")
        return

    # Select top-N categories by mention volume in the all-skills dataframe
    top_cats = (
        df_all.nlargest(top_n, "n_job_mentions")["category"].tolist()
    )

    def _scr_for_cats(df: pd.DataFrame, cats: list[str]) -> list[float]:
        idx = df.set_index("category")["scr"] if not df.empty else pd.Series(dtype=float)
        return [float(idx.get(c, 0.0)) for c in cats]

    scr_all  = _scr_for_cats(df_all,  top_cats)
    scr_hard = _scr_for_cats(df_hard, top_cats)
    scr_soft = _scr_for_cats(df_soft, top_cats)

    x     = np.arange(len(top_cats))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, scr_all,  width, label="All skills",       color=_COLOURS["scr"],    alpha=0.88)
    ax.bar(x,         scr_hard, width, label="Hard skills only", color=_COLOURS["high"],   alpha=0.88)
    ax.bar(x + width, scr_soft, width, label="Soft skills only", color=_COLOURS["medium"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(top_cats, rotation=38, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Skill Coverage Rate (SCR)", fontsize=10)
    ax.set_ylim(0, min(1.0, max(scr_all + scr_hard + scr_soft) + 0.10))
    ax.set_title(
        f"Hard vs Soft Skill Coverage — Top {top_n} Job Categories by Demand Volume\n"
        "Note: lower soft-skill SCR reflects a course-description artefact, "
        "not a curriculum gap.",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate the artefact visually with a note
    ax.annotate(
        "Soft-skill gap ≠ curriculum gap\n"
        "(course descriptions rarely list\n"
        " soft skills even when taught)",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec="#f0c040", alpha=0.9),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    df_all:  pd.DataFrame,
    df_hard: pd.DataFrame,
    df_soft: pd.DataFrame,
    n_course_skills:    int,
    n_job_skills_total: int,
) -> dict[str, Any]:
    """Aggregate headline numbers into a summary dict (saved as JSON)."""

    def _agg(df: pd.DataFrame, tag: str) -> dict[str, Any]:
        if df.empty:
            return {}
        total_job   = df["n_unique_job_skills"].sum()
        total_cov   = df["n_covered"].sum()
        total_ment  = df["n_job_mentions"].sum()
        total_cment = df["n_covered_mentions"].sum()
        return {
            f"{tag}_macro_scr":    round(float(df["scr"].mean()),    4),
            f"{tag}_macro_dw_scr": round(float(df["dw_scr"].mean()), 4),
            f"{tag}_micro_scr":    round(total_cov  / total_job   if total_job   > 0 else 0.0, 4),
            f"{tag}_micro_dw_scr": round(total_cment / total_ment if total_ment  > 0 else 0.0, 4),
        }

    return {
        "matching_method":          "exact_string_match_after_canonicalization",
        "n_unique_course_skills":   n_course_skills,
        "n_unique_job_skills":      n_job_skills_total,
        "n_job_categories":         len(df_all),
        **_agg(df_all,  "all"),
        **_agg(df_hard, "hard"),
        **_agg(df_soft, "soft"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute baseline SCR (exact string match) for all job categories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-skills",    type=Path, default=None,
                   help="Path to job skills file (auto-selected from --job-source if omitted)")
    p.add_argument("--job-source",    type=str,  default="skills_list",
                   choices=["skillner", "skills_list"],
                   help="Job skill source: 'skillner' (NLP-extracted) or "
                        "'skills_list' (employer-provided)")
    p.add_argument("--course-skills", type=Path, default=DEFAULT_COURSE_SKILLS,
                   help="Path to module_skill_pairs.csv")
    p.add_argument("--output",        type=Path, default=None,
                   help="Output directory (auto-named from --job-source if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-select default paths based on source
    if args.job_skills is None:
        args.job_skills = (
            JOBS_PROCESSED_DIR / "job_skill_pair_skillner.csv"
            if args.job_source == "skillner"
            else JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"
        )
    if args.output is None:
        args.output = RESULTS_DIR / f"baseline_scr_{args.job_source}"

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading data  [job-source: {args.job_source}] …")
    courses = load_course_skills(args.course_skills)
    jobs    = load_job_skills_auto(args.job_skills, args.job_source)

    course_skill_set = set(courses["skill_canon"].unique())

    print(f"  Course skills  (unique, canonicalized) : {len(course_skill_set):>7,}")
    print(f"  Job skill rows                         : {len(jobs):>7,}")
    print(f"  Job skills     (unique, canonicalized) : {jobs['skill_canon'].nunique():>7,}")
    print(f"  Job categories                         : {jobs['category'].nunique():>7}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("\nComputing SCR …")
    df_all  = compute_scr(jobs, course_skill_set)
    df_hard = compute_scr(jobs, course_skill_set, skill_type_filter="hard")
    df_soft = compute_scr(jobs, course_skill_set, skill_type_filter="soft")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    print("\nSaving outputs …")
    for df, tag in [(df_all, "all"), (df_hard, "hard"), (df_soft, "soft")]:
        if df.empty:
            continue
        csv_path = out / f"per_category_{tag}.csv"
        # Drop the nested list column for the flat CSV; it is kept in df for charting
        df.drop(columns=["top_10_uncovered"]).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

    # ── Save gap lists (top uncovered skills per category) ────────────────────
    gap_path = out / "top_uncovered_skills.json"
    gap_data = {
        row["category"]: row["top_10_uncovered"]
        for _, row in df_all.iterrows()
    }
    with gap_path.open("w", encoding="utf-8") as f:
        json.dump(gap_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {gap_path.name}")

    # ── Save summary JSON ──────────────────────────────────────────────────────
    summary = build_summary(
        df_all, df_hard, df_soft,
        n_course_skills=len(course_skill_set),
        n_job_skills_total=jobs["skill_canon"].nunique(),
    )
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path.name}")

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_scr_by_category(df_all, out / "fig_scr_by_category.png")
    plot_hard_vs_soft(df_all, df_hard, df_soft, out / "fig_hard_vs_soft_scr.png")

    # ── Print headline numbers ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BASELINE SCR — HEADLINE RESULTS")
    print("=" * 62)
    print(f"  {'Metric':<30} {'All':>8}  {'Hard':>8}  {'Soft':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}")
    for key, label in [("macro_scr", "Macro SCR"), ("macro_dw_scr", "Macro DW-SCR"),
                        ("micro_scr", "Micro SCR"), ("micro_dw_scr", "Micro DW-SCR")]:
        a = summary.get(f"all_{key}",  0)
        h = summary.get(f"hard_{key}", 0)
        s = summary.get(f"soft_{key}", 0)
        print(f"  {label:<30} {a:>7.1%}  {h:>7.1%}  {s:>7.1%}")
    print("=" * 62)
    print(f"\n  Outputs written to: {out}\n")


if __name__ == "__main__":
    main()
