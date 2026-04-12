"""
Improved Skill Coverage Rate (SCR) — Step 3 of the coverage pipeline.

What this script does
---------------------
Addresses two systematic weaknesses of the baseline exact-match SCR:

  1. Vocabulary mismatch — exact matching misses semantically equivalent
     skills described differently in academic vs industry language.
     Fix: incorporate pre-computed embedding similarity scores so that
     near-equivalent skills contribute partial (soft) coverage credit.

  2. Generic skill dominance — raw frequency weighting inflates the
     contribution of ubiquitous soft skills ("communication", "teamwork")
     that appear in every job category and dilute the technical signal.
     Fix: IDF weighting down-weights skills that appear across many
     categories, amplifying category-specific technical skills.

The two fixes are combined into a single metric: IDF-weighted Soft-SCR.

Method
------
For each job skill j in category c:

  sim(j) = 1.0                    if j ∈ exact course skill set
         = nn_sim(j)              if nn_sim(j) ≥ θ  (θ = 0.70 by default)
         = 0.0                    otherwise

  idf(j) = log( N / df(j) ) + 1  where N = total job categories,
                                        df(j) = categories mentioning j

  Soft-IDF-SCR_c = Σ_{j∈c} sim(j) × idf(j)  /  Σ_{j∈c} idf(j)

Three coverage variants are reported for comparison:

  Variant A — Semantic-SCR (hard threshold, equal weights):
      sim(j) ∈ {0, 1}  based on threshold θ,  weight = 1
      Isolates the effect of semantic matching alone.

  Variant B — Soft-SCR (continuous similarity, equal weights):
      sim(j) ∈ [0, 1] continuously,  weight = 1
      Isolates the effect of soft/partial coverage credit.

  Variant C — Soft-IDF-SCR (continuous similarity, IDF weighted):
      sim(j) ∈ [0, 1] continuously,  weight = idf(j)
      The full improved metric combining both fixes.

IDF theoretical grounding
--------------------------
IDF (Inverse Document Frequency) originates in information retrieval
(Sparck Jones, 1972) and quantifies the informativeness of a term
relative to a corpus.  Applied here: a skill demanded in every job
category (df = 43) is generic and uninformative about curriculum gaps;
a skill demanded in only one category (df = 1) is highly specific and
informative.  log-smoothing (+1) prevents zero weights for ubiquitous
skills and handles the df = N edge case.

Semantic matching theoretical grounding
----------------------------------------
Skill coverage is treated as a soft set membership problem.  Rather than
a crisp boundary (in / not in), we use the continuous cosine similarity
from the embedding space as a graded membership function, truncated at θ
to exclude clearly unrelated skills.  This is equivalent to a fuzzy set
intersection (Zadeh, 1965) operationalised via dense vector retrieval.

Outputs
-------
results/improved_scr/
  ├── per_category_comparison.csv     — baseline vs all three variants
  ├── summary_comparison.json         — headline numbers across variants
  ├── fig_baseline_vs_improved.png    — side-by-side bar chart per category
  └── fig_improvement_breakdown.png  — improvement decomposition chart

Usage
-----
    python -m src.analysis.skill_metrics.improved_scr
    python -m src.analysis.skill_metrics.improved_scr --threshold 0.70  # exploratory
    python -m src.analysis.skill_metrics.improved_scr --threshold 0.85  # primary (default)
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
DEFAULT_NN_PATH       = RESULTS_DIR / "vocabulary_mismatch" / "all_uncovered_nearest_neighbour.csv"
DEFAULT_OUTPUT_DIR    = RESULTS_DIR / "improved_scr"
DEFAULT_THRESHOLD     = 0.72


# ─────────────────────────────────────────────────────────────────────────────
# IDF weighting
# ─────────────────────────────────────────────────────────────────────────────

def compute_idf(job_skills: pd.DataFrame) -> dict[str, float]:
    """
    Compute smoothed IDF weight for every job skill.

    IDF(j) = log( N / df(j) ) + 1

    where N  = number of distinct job categories,
          df(j) = number of categories in which skill j appears at least once.

    The +1 smoothing ensures skills present in all N categories still
    receive a positive (though small) weight, avoiding zero-division and
    preserving their contribution to the denominator.

    Returns
    -------
    dict mapping canonicalized skill label → IDF weight (float)
    """
    categories   = job_skills["category"].unique()
    N            = len(categories)

    # Document frequency: how many categories mention this skill?
    df_counts = (
        job_skills.groupby("skill_canon")["category"]
        .nunique()
    )

    idf_map: dict[str, float] = {
        skill: np.log(N / df) + 1.0
        for skill, df in df_counts.items()
    }
    return idf_map


# ─────────────────────────────────────────────────────────────────────────────
# Coverage score per skill
# ─────────────────────────────────────────────────────────────────────────────

def build_skill_coverage_scores(
    job_skills:       pd.DataFrame,
    course_skill_set: set[str],
    nn_similarities:  dict[str, float],
    threshold:        float,
) -> pd.DataFrame:
    """
    Assign a coverage score to every unique job skill.

    Coverage score:
        1.0              if skill is in exact course skill set
        nn_sim           if nn_sim ≥ threshold  (semantic near-match)
        0.0              otherwise  (genuine gap)

    This produces three binary/continuous variants simultaneously:
        - hard_covered  : 1 if sim ≥ threshold, else 0  (Variant A)
        - soft_score    : actual sim value              (Variant B/C)

    Parameters
    ----------
    job_skills       : DataFrame from load_job_skills()
    course_skill_set : set of canonicalized course skill labels
    nn_similarities  : dict mapping uncovered skill → max cosine similarity
                       (pre-computed by vocabulary_mismatch.py)
    threshold        : minimum similarity for a near-match to count

    Returns
    -------
    DataFrame with one row per unique job skill:
        skill_canon, exact_match, nn_sim, hard_covered, soft_score
    """
    unique_skills = job_skills["skill_canon"].unique()
    rows: list[dict[str, Any]] = []

    for skill in unique_skills:
        exact = skill in course_skill_set

        if exact:
            nn_sim       = 1.0
            hard_covered = True
            soft_score   = 1.0
        else:
            nn_sim = nn_similarities.get(skill, 0.0)
            hard_covered = nn_sim >= threshold
            soft_score   = nn_sim if nn_sim >= threshold else 0.0

        rows.append({
            "skill_canon":  skill,
            "exact_match":  exact,
            "nn_sim":       round(nn_sim,     4),
            "hard_covered": hard_covered,
            "soft_score":   round(soft_score, 4),
        })

    return pd.DataFrame(rows).set_index("skill_canon")


# ─────────────────────────────────────────────────────────────────────────────
# Per-category SCR computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_improved_scr(
    job_skills:      pd.DataFrame,
    coverage_scores: pd.DataFrame,
    idf_map:         dict[str, float],
) -> pd.DataFrame:
    """
    Compute all three improved SCR variants per job category.

    Variant A — Semantic-SCR (hard threshold, equal weights):
        Coverage is binary (0/1) at threshold θ; no IDF weighting.
        Isolates the effect of semantic matching vs exact matching.

    Variant B — Soft-SCR (continuous, equal weights):
        Coverage is the continuous similarity score ∈ [0,1].
        Isolates the effect of partial credit for near-matches.

    Variant C — Soft-IDF-SCR (continuous, IDF weighted):
        Full improved metric: soft coverage × IDF demand weight.
        This is the primary metric for Step 3.

    Returns
    -------
    DataFrame with one row per category, columns:
        category,
        baseline_scr,       ← exact match, equal weight (from Step 1)
        semantic_scr,       ← Variant A
        soft_scr,           ← Variant B
        soft_idf_scr,       ← Variant C  (primary)
        n_unique_job_skills,
        n_exact_covered,
        n_semantic_covered, ← exact + near-miss (hard threshold)
        top_10_gaps         ← highest-IDF uncovered skills
    """
    rows: list[dict[str, Any]] = []

    for category, grp in job_skills.groupby("category"):
        skills_in_cat = grp["skill_canon"].unique()

        # ── Per-skill metrics for this category ───────────────────────────────
        exact_covered    = 0
        semantic_covered = 0

        # Accumulators for weighted SCR variants
        sum_soft_eq   = 0.0   # Variant B numerator (soft, equal weight)
        sum_denom_eq  = 0.0   # Variant B denominator

        sum_soft_idf  = 0.0   # Variant C numerator (soft, IDF weight)
        sum_denom_idf = 0.0   # Variant C denominator

        gap_skills: list[dict] = []  # uncovered skills for gap list

        for skill in skills_in_cat:
            if skill not in coverage_scores.index:
                continue

            row      = coverage_scores.loc[skill]
            idf_w    = idf_map.get(skill, 1.0)
            soft     = float(row["soft_score"])
            is_exact = bool(row["exact_match"])
            is_hard  = bool(row["hard_covered"])

            if is_exact:
                exact_covered += 1
            if is_hard:
                semantic_covered += 1
            else:
                # Track uncovered skills with their IDF weight for gap list
                gap_skills.append({"skill": skill, "idf": round(idf_w, 3)})

            # Variant B: soft score, equal weight
            sum_soft_eq  += soft
            sum_denom_eq += 1.0

            # Variant C: soft score, IDF weight
            sum_soft_idf  += soft * idf_w
            sum_denom_idf += idf_w

        n_total = len(skills_in_cat)

        # Variant A: hard threshold, equal weight
        semantic_scr = semantic_covered / n_total if n_total > 0 else 0.0
        # Variant B
        soft_scr     = sum_soft_eq  / sum_denom_eq  if sum_denom_eq  > 0 else 0.0
        # Variant C (primary)
        soft_idf_scr = sum_soft_idf / sum_denom_idf if sum_denom_idf > 0 else 0.0
        # Baseline (exact, equal weight)
        baseline_scr = exact_covered / n_total if n_total > 0 else 0.0

        # Top gaps: uncovered skills sorted by IDF (most informative first)
        top_gaps = sorted(gap_skills, key=lambda x: x["idf"], reverse=True)[:10]

        rows.append({
            "category":             category,
            "baseline_scr":         round(baseline_scr,  4),
            "semantic_scr":         round(semantic_scr,  4),
            "soft_scr":             round(soft_scr,       4),
            "soft_idf_scr":         round(soft_idf_scr,  4),
            "n_unique_job_skills":  n_total,
            "n_exact_covered":      exact_covered,
            "n_semantic_covered":   semantic_covered,
            "top_10_gaps":          top_gaps,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("soft_idf_scr", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_vs_improved(df: pd.DataFrame, output_path: Path) -> None:
    """
    Side-by-side horizontal bar chart: baseline SCR vs Soft-IDF-SCR per category.

    Categories sorted by Soft-IDF-SCR descending (best coverage at top).
    The gap between the two bars in each row represents the coverage
    that exact matching was missing due to vocabulary mismatch.
    """
    df_plot = df.sort_values("soft_idf_scr", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(8, len(df_plot) * 0.42)))

    y      = np.arange(len(df_plot))
    height = 0.38

    bars_base = ax.barh(
        y + height / 2, df_plot["baseline_scr"],
        height=height, color="#e74c3c", alpha=0.85,
        label="Baseline SCR (exact match)",
    )
    bars_imp = ax.barh(
        y - height / 2, df_plot["soft_idf_scr"],
        height=height, color="#27ae60", alpha=0.85,
        label="Soft-IDF-SCR (semantic + IDF weighted)",
    )

    # Value labels
    for bar in bars_base:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1%}", va="center", ha="left",
                fontsize=7, color="#c0392b")
    for bar in bars_imp:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1%}", va="center", ha="left",
                fontsize=7, color="#1e8449")

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["category"], fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Skill Coverage Rate", fontsize=10)

    # Macro averages
    macro_base = df_plot["baseline_scr"].mean()
    macro_imp  = df_plot["soft_idf_scr"].mean()
    ax.axvline(macro_base, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Baseline macro avg: {macro_base:.1%}")
    ax.axvline(macro_imp,  color="#27ae60", linestyle="--", linewidth=1.2,
               label=f"Improved macro avg: {macro_imp:.1%}")

    ax.set_title(
        "Skill Coverage: Baseline (Exact Match) vs Improved (Semantic + IDF)\n"
        "Green bars show true coverage after accounting for vocabulary mismatch",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_improvement_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    """
    Stacked bar chart decomposing what drives the improvement for each category.

    Each bar shows:
      - Exact coverage     (baseline)
      - Semantic gain      (near-miss recovery, hard threshold)
      - IDF reweighting    (difference between soft-IDF and soft equal-weight)

    This makes the contribution of each methodological fix transparent.
    """
    df_plot = df.sort_values("soft_idf_scr", ascending=True)

    exact_cov  = df_plot["baseline_scr"].values
    sem_gain   = (df_plot["semantic_scr"] - df_plot["baseline_scr"]).clip(lower=0).values
    idf_effect = (df_plot["soft_idf_scr"] - df_plot["semantic_scr"]).values

    fig, ax = plt.subplots(figsize=(13, max(8, len(df_plot) * 0.38)))

    y = np.arange(len(df_plot))
    ax.barh(y, exact_cov,            color="#e74c3c", alpha=0.85, label="Exact coverage (baseline)")
    ax.barh(y, sem_gain,  left=exact_cov,
            color="#3498db", alpha=0.85, label="Semantic gain (near-miss recovery)")
    ax.barh(y, idf_effect, left=exact_cov + sem_gain,
            color="#f39c12", alpha=0.85, label="IDF reweighting effect")

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["category"], fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Skill Coverage Rate", fontsize=10)
    ax.set_title(
        "Coverage Improvement Decomposition by Category\n"
        "What fraction of the improvement comes from semantic matching vs IDF reweighting?",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc="lower right")
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
        description="Improved SCR: semantic matching + IDF demand weighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--job-source", type=str, default="skills_list",
        choices=["skillner", "skills_list"],
        help="Which job skill source to use: 'skillner' (NLP-extracted) or "
             "'skills_list' (employer-provided from 01b_jobs_cleaned.csv)",
    )
    p.add_argument("--job-skills",    type=Path,  default=None,
                   help="Override job skills file (default: auto-selected by --job-source)")
    p.add_argument("--course-skills", type=Path,  default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--nn-path",       type=Path,  default=None,
                   help="Path to all_uncovered_nearest_neighbour.csv from vocabulary_mismatch.py "
                        "(default: auto-selected from results/vocabulary_mismatch_{source}/)")
    p.add_argument("--output",        type=Path,  default=None,
                   help="Output directory (default: results/improved_scr_{source}/)")
    p.add_argument("--threshold",     type=float, default=DEFAULT_THRESHOLD,
                   help="Cosine similarity threshold for near-match (default: 0.72)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Auto-select paths based on --job-source ────────────────────────────────
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
        t_tag = f"{args.threshold:.2f}".replace(".", "")
        args.output = RESULTS_DIR / f"improved_scr_{args.job_source}_t{t_tag}"

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"Job source : {args.job_source}")
    print(f"Job skills : {args.job_skills}")
    print(f"NN path    : {args.nn_path}")
    print(f"Output dir : {out}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    courses = load_course_skills(args.course_skills)
    jobs    = load_job_skills_auto(args.job_skills, args.job_source)

    course_skill_set = set(courses["skill_canon"].unique())

    # Load pre-computed nearest-neighbour similarities (from vocabulary_mismatch.py)
    if not args.nn_path.exists():
        raise FileNotFoundError(
            f"Nearest-neighbour file not found: {args.nn_path}\n"
            "Run vocabulary_mismatch.py first."
        )
    nn_df = pd.read_csv(args.nn_path)
    nn_similarities: dict[str, float] = dict(
        zip(nn_df["job_skill"], nn_df["similarity"])
    )
    print(f"  Loaded {len(nn_similarities):,} pre-computed nearest-neighbour similarities")

    # ── IDF weights ────────────────────────────────────────────────────────────
    print("Computing IDF weights …")
    idf_map = compute_idf(jobs)
    idf_vals = np.array(list(idf_map.values()))
    print(f"  IDF range: [{idf_vals.min():.3f}, {idf_vals.max():.3f}]  "
          f"mean={idf_vals.mean():.3f}")

    # ── Coverage scores ────────────────────────────────────────────────────────
    print(f"Building coverage scores (threshold = {args.threshold}) …")
    coverage_scores = build_skill_coverage_scores(
        jobs, course_skill_set, nn_similarities, args.threshold
    )

    n_exact    = int(coverage_scores["exact_match"].sum())
    n_semantic = int(coverage_scores["hard_covered"].sum())
    n_total    = len(coverage_scores)
    print(f"  Exact matches          : {n_exact:,}  ({n_exact/n_total:.1%})")
    print(f"  Semantic matches (≥{args.threshold:.2f}): {n_semantic:,}  ({n_semantic/n_total:.1%})")
    print(f"  Genuine gaps           : {n_total - n_semantic:,}  ({(n_total-n_semantic)/n_total:.1%})")

    # ── Improved SCR per category ──────────────────────────────────────────────
    print("\nComputing improved SCR per category …")
    df = compute_improved_scr(jobs, coverage_scores, idf_map)

    # ── Save outputs ───────────────────────────────────────────────────────────
    print("\nSaving outputs …")

    df.drop(columns=["top_10_gaps"]).to_csv(
        out / "per_category_comparison.csv", index=False
    )
    print("  Saved: per_category_comparison.csv")

    # Gap list: IDF-ranked uncovered skills per category
    gap_data = {
        row["category"]: row["top_10_gaps"]
        for _, row in df.iterrows()
    }
    with (out / "top_gaps_idf_ranked.json").open("w", encoding="utf-8") as f:
        json.dump(gap_data, f, indent=2, ensure_ascii=False)
    print("  Saved: top_gaps_idf_ranked.json")

    # Summary
    def _macro(col: str) -> float:
        return round(float(df[col].mean()), 4)
    def _micro(col: str) -> float:
        total_cov = (df["n_unique_job_skills"] * df[col]).sum()
        total_job = df["n_unique_job_skills"].sum()
        return round(float(total_cov / total_job), 4) if total_job > 0 else 0.0

    summary: dict[str, Any] = {
        "threshold":               args.threshold,
        "n_categories":            len(df),
        "baseline_macro_scr":      _macro("baseline_scr"),
        "semantic_macro_scr":      _macro("semantic_scr"),
        "soft_macro_scr":          _macro("soft_scr"),
        "soft_idf_macro_scr":      _macro("soft_idf_scr"),
        "baseline_micro_scr":      _micro("baseline_scr"),
        "soft_idf_micro_scr":      _micro("soft_idf_scr"),
        "macro_improvement":       round(_macro("soft_idf_scr") - _macro("baseline_scr"), 4),
        "n_exact_covered":         n_exact,
        "n_semantic_covered":      n_semantic,
        "n_genuine_gaps":          n_total - n_semantic,
        "pct_genuine_gaps":        round((n_total - n_semantic) / n_total, 4),
    }
    with (out / "summary_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  Saved: summary_comparison.json")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_baseline_vs_improved(df, out / "fig_baseline_vs_improved.png")
    plot_improvement_breakdown(df, out / "fig_improvement_breakdown.png")

    # ── Headline ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  IMPROVED SCR — HEADLINE RESULTS")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Macro':>8}  {'Micro':>8}")
    print(f"  {'-'*28} {'-'*8}  {'-'*8}")
    for label, col in [
        ("Baseline (exact match)",  "baseline_scr"),
        ("Variant A: Semantic",     "semantic_scr"),
        ("Variant B: Soft",         "soft_scr"),
        ("Variant C: Soft-IDF",     "soft_idf_scr"),
    ]:
        macro = _macro(col)
        micro = _micro(col)
        print(f"  {label:<28} {macro:>7.1%}  {micro:>7.1%}")
    print(f"  {'─'*28} {'─'*8}  {'─'*8}")
    print(f"  {'Improvement (A→C)':<28} "
          f"{_macro('soft_idf_scr') - _macro('baseline_scr'):>+7.1%}")
    print("=" * 62)
    print(f"\n  Of {n_total:,} unique job skills:")
    print(f"    Exact matches     : {n_exact:,}  ({n_exact/n_total:.1%})")
    print(f"    Near-matches (≥{args.threshold}) : {n_semantic - n_exact:,}  "
          f"({(n_semantic - n_exact)/n_total:.1%})")
    print(f"    Genuine gaps      : {n_total - n_semantic:,}  "
          f"({(n_total - n_semantic)/n_total:.1%})")
    print(f"\n  Outputs written to: {out}\n")


if __name__ == "__main__":
    main()
