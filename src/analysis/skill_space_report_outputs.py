"""
skill_space_report_outputs.py — Report-ready figures for Skill-Space Alignment
-------------------------------------------------------------------------------
Generates 5 publication-quality charts matching the visual style of
report_outputs.py (same palette, background, DPI, font).

Charts produced
---------------
  1. fig_ss_chart1_ssoc_overview.png
       4 panels side by side:
         (a) Distribution (histogram) of Soft-IDF-SCR across all SSOC minor groups
         (b) Soft-IDF-SCR for every SSOC minor group (horizontal bar, sorted)
         (c) Top 10 SSOC minor groups by applied Soft-IDF-SCR
         (d) Top 10 SSOC minor groups by theoretical Soft-IDF-SCR

  2. fig_ss_chart2_major_coverage.png
       Horizontal bar — Soft-SCR for all 55 degree programmes, coloured by faculty.
       SSOC-aligned: each major scored against its target career track.

  3. fig_ss_chart3_degree_pgi.png
       Scatter — x=job demand weight, y=misalignment rate, size+colour=degree PGI.
       Top-right = high demand AND poorly aligned = highest MOE urgency.
       Annotated bar chart of top 10 by degree PGI alongside.

  4. fig_ss_chart4_degree_alignment.png
       3 panels:
         (a) Histogram distribution of Degree Alignment Scores across all 55 degrees
         (b) Top 10 highest alignment score (best-aligned degrees)
         (c) Bottom 10 lowest alignment score (most misaligned — MOE review priority)

  5. fig_ss_chart5_foundational_ratio.png
       Horizontal bar — Foundational Ratio per degree programme/major (share of theoretical course
       skills with no job-market demand), sorted ascending.

Usage
-----
    python -m src.analysis.skill_space_report_outputs
    python -m src.analysis.skill_space_report_outputs --job-source skills_list --threshold 0.72
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

FIG_OUT = REPO_ROOT / "outputs" / "report_figures"

# ── style — identical to report_outputs.py ─────────────────────────────────
ACCENT   = "#2563EB"
ACCENT2  = "#7C3AED"
MUTED    = "#CBD5E1"
GOOD     = "#10B981"
WARN     = "#F59E0B"
BAD      = "#EF4444"
BG       = "#F8FAFC"
TEXT_CLR = "#1E293B"

CONF_COLOURS = {"High": BAD, "Medium": WARN, "Low": GOOD}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "text.color":        TEXT_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
})


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_OUT / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


def _shorten(label: str, n: int = 40) -> str:
    return label if len(label) <= n else label[:n - 2] + ".."


# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — SSOC Minor Group Coverage: distribution + full ranking
# ══════════════════════════════════════════════════════════════════════════

def plot_chart1_ssoc_overview(ssoc: pd.DataFrame) -> None:
    """
    2 panels side by side:
      (a) Histogram distribution of Soft-IDF-SCR across all SSOC minor groups
      (b) Horizontal bar — Soft-IDF-SCR for every SSOC minor group, sorted
    """
    print("  Chart 1 — SSOC distribution + full ranking...")

    df = ssoc[ssoc["knowledge_type_filter"] == "all"].copy()
    df = df.sort_values("soft_idf_scr", ascending=True)
    n_groups = len(df)

    fig_h = max(14, n_groups * 0.30 + 3)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h),
                             gridspec_kw={"width_ratios": [1, 2.2]})

    # ── Panel (a): distribution histogram ─────────────────────────────────
    ax = axes[0]
    vals = df["soft_idf_scr"] * 100
    ax.hist(vals, bins=18, color=ACCENT, alpha=0.82, edgecolor="white", linewidth=0.6)
    mean_val = vals.mean()
    ax.axvline(mean_val, color=BAD, linewidth=1.5, linestyle="--")
    ax.text(mean_val + 0.5, ax.get_ylim()[1] * 0.95,
            f"Mean: {mean_val:.1f}%", fontsize=8, color=BAD, va="top")
    ax.set_xlabel("Soft-IDF-SCR (%)", fontsize=10, labelpad=6)
    ax.set_ylabel("Number of SSOC minor groups", fontsize=10, labelpad=6)
    ax.set_title("(a) Distribution of Soft-IDF-SCR\nacross all SSOC minor groups",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.25)

    # ── Panel (b): all SSOC minors ranked ─────────────────────────────────
    ax = axes[1]
    y = np.arange(n_groups)
    colours_b = [GOOD if v >= 0.7 else (WARN if v >= 0.5 else BAD)
                 for v in df["soft_idf_scr"]]
    ax.barh(y, df["soft_idf_scr"] * 100, height=0.75, color=colours_b, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(t, 50) for t in df["ssoc_minor_title"]], fontsize=5.5)
    ax.set_xlabel("Soft-IDF-SCR (%)", fontsize=10, labelpad=6)
    ax.set_title("(b) Soft-IDF-SCR by SSOC Minor Group\n(all groups, sorted ascending)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.axvline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)
    patches_b = [
        mpatches.Patch(color=GOOD, alpha=0.82, label="≥ 70%"),
        mpatches.Patch(color=WARN, alpha=0.82, label="50–70%"),
        mpatches.Patch(color=BAD,  alpha=0.82, label="< 50%"),
    ]
    ax.legend(handles=patches_b, fontsize=8, loc="lower right",
              framealpha=0.9, title="Coverage", title_fontsize=8)

    fig.suptitle("Soft-IDF-SCR: NUS Coverage of Job-Demanded Skills by SSOC Minor Group",
                 fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout()
    _save(fig, "fig_ss_chart1_ssoc_overview.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 1B — SSOC Minor Group Coverage: applied vs theoretical top 10
# ══════════════════════════════════════════════════════════════════════════

def plot_chart1b_ssoc_applied_theo(ssoc: pd.DataFrame) -> None:
    """
    2 panels side by side:
      (a) Top 10 SSOC minor groups by applied Soft-IDF-SCR
      (b) Top 10 SSOC minor groups by theoretical Soft-IDF-SCR
    """
    print("  Chart 1b — SSOC top 10 applied + theoretical...")

    df = ssoc[ssoc["knowledge_type_filter"] == "all"].copy()
    top10_applied = df.nlargest(10, "soft_idf_scr_applied").sort_values("soft_idf_scr_applied")
    top10_theo    = df.nlargest(10, "soft_idf_scr_theoretical").sort_values("soft_idf_scr_theoretical")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel (a): top 10 applied ─────────────────────────────────────────
    ax = axes[0]
    y = np.arange(len(top10_applied))
    bars = ax.barh(y, top10_applied["soft_idf_scr_applied"] * 100,
                   height=0.65, color=GOOD, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(t, 42) for t in top10_applied["ssoc_minor_title"]], fontsize=8)
    for bar, val in zip(bars, top10_applied["soft_idf_scr_applied"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=8)
    ax.set_xlabel("Applied Soft-IDF-SCR (%)", fontsize=10, labelpad=6)
    ax.set_title("(a) Top 10 SSOC Minor Groups\nby Applied Coverage",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, top10_applied["soft_idf_scr_applied"].max() * 100 * 1.22)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)

    # ── Panel (b): top 10 theoretical ────────────────────────────────────
    ax = axes[1]
    y2 = np.arange(len(top10_theo))
    bars2 = ax.barh(y2, top10_theo["soft_idf_scr_theoretical"] * 100,
                    height=0.65, color=WARN, alpha=0.82)
    ax.set_yticks(y2)
    ax.set_yticklabels([_shorten(t, 42) for t in top10_theo["ssoc_minor_title"]], fontsize=8)
    for bar, val in zip(bars2, top10_theo["soft_idf_scr_theoretical"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=8)
    ax.set_xlabel("Theoretical Soft-IDF-SCR (%)", fontsize=10, labelpad=6)
    ax.set_title("(b) Top 10 SSOC Minor Groups\nby Theoretical Coverage",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, top10_theo["soft_idf_scr_theoretical"].max() * 100 * 1.22)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)

    fig.suptitle("Soft-IDF-SCR: Applied vs Theoretical Coverage — Top 10 SSOC Minor Groups",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_ss_chart1b_ssoc_applied_theo.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — Soft-SCR per degree programme (all majors)
# ══════════════════════════════════════════════════════════════════════════

def plot_chart2_major_coverage(major: pd.DataFrame) -> None:
    """
    Horizontal bar — Soft-SCR (all) for every NUS degree programme.
    Sorted ascending (lowest coverage at top).
    Includes a note on Life Sciences data limitation.
    """
    print("  Chart 2 — Major Soft-SCR coverage...")

    df = major.sort_values("soft_scr_all", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(13, max(12, n * 0.32)))

    y = np.arange(n)
    bars = ax.barh(y, df["soft_scr_all"] * 100,
                   height=0.72, color=ACCENT, alpha=0.82)

    for bar, val in zip(bars, df["soft_scr_all"]):
        if val > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["major"], fontsize=7.5)
    ax.set_xlabel("Soft-SCR — Skill Coverage (%)", fontsize=11, labelpad=8)
    ax.set_title(
        "Skill Coverage by Degree Programme (Soft-SCR)\n"
        "SSOC-aligned: each major scored against its target career-track job postings",
        fontsize=12, fontweight="bold", pad=14,
    )

    med = df["soft_scr_all"].median() * 100
    ax.axvline(med, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.text(med + 0.2, n * 0.99, f"Median: {med:.1f}%",
            fontsize=7.5, color=TEXT_CLR, alpha=0.7, va="top")

    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, df["soft_scr_all"].max() * 100 * 1.18)

    ax.annotate(
        "* Life Sciences and Engineering Science have low skill-extraction\n"
        "  coverage — scores reflect data availability, not curriculum quality.",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=6.5, color=TEXT_CLR, alpha=0.6,
        ha="left", va="bottom",
    )

    fig.tight_layout()
    _save(fig, "fig_ss_chart2_major_coverage.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 3 — Degree-level PGI
# ══════════════════════════════════════════════════════════════════════════

def plot_chart3_degree_pgi(deg: pd.DataFrame) -> None:
    """
    Two panels:
      Left  — Scatter: x=job demand weight, y=misalignment rate.
               Bubble size and colour intensity = degree PGI.
               Top-right quadrant = high demand AND poorly aligned = most urgent.
      Right — Horizontal bar: top 15 degrees by degree PGI, coloured by faculty.

    Degree PGI = job_demand_weight × misalignment_rate
      job_demand_weight = share of total mapped jobs this degree targets
      misalignment_rate = 1 − (alignment_score / 100)
    """
    print("  Chart 3 — Degree PGI (scatter + bar)...")

    df = deg.copy()
    top15 = df.nlargest(15, "degree_pgi").sort_values("degree_pgi", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # ── Panel left: scatter ───────────────────────────────────────────────
    ax = axes[0]

    # Soft shading — top-right = most urgent
    ax.fill_between(
        [df["job_demand_weight"].quantile(0.75), df["job_demand_weight"].max() * 1.15],
        [df["misalignment_rate"].quantile(0.75)] * 2,
        [1.05, 1.05],
        alpha=0.05, color=BAD, zorder=0,
    )

    from matplotlib.colors import LinearSegmentedColormap
    cmap    = LinearSegmentedColormap.from_list("pgi", [ACCENT, BAD])
    pgi_max = df["degree_pgi"].max()
    for _, row in df.iterrows():
        col = cmap(row["degree_pgi"] / pgi_max)
        ax.scatter(
            row["job_demand_weight"] * 100,
            row["misalignment_rate"] * 100,
            s=80, color=col, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3,
        )
        is_top = row["pgi_rank"] <= 10
        label  = _shorten(row["major"], 22)
        ax.annotate(
            label,
            (row["job_demand_weight"] * 100, row["misalignment_rate"] * 100),
            fontsize=6 if not is_top else 7,
            fontweight="bold" if is_top else "normal",
            color=BAD if is_top else TEXT_CLR,
            xytext=(5, 3), textcoords="offset points",
        )

    # Reference lines at medians
    ax.axhline(df["misalignment_rate"].median() * 100,
               color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(df["job_demand_weight"].median() * 100,
               color=MUTED, linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel(
        "Job Demand Weight (%)\n← fewer jobs targeted          more jobs targeted →",
        fontsize=10, labelpad=8,
    )
    ax.set_ylabel(
        "Misalignment Rate (%)\n← better aligned          more misaligned →",
        fontsize=10, labelpad=8,
    )
    ax.set_title(
        "Degree PGI — Demand × Misalignment\n"
        "Top-right = large job-market footprint AND poorly aligned",
        fontsize=11, fontweight="bold", pad=12,
    )

    # Quadrant annotations
    ax.text(0.97, 0.97, "Review Priority", transform=ax.transAxes,
            fontsize=8, color=BAD, fontweight="bold", ha="right", va="top")
    ax.text(0.03, 0.03, "Low impact", transform=ax.transAxes,
            fontsize=8, color=GOOD, ha="left", va="bottom")
    ax.text(0.97, 0.03, "Well-aligned\n(high demand)", transform=ax.transAxes,
            fontsize=8, color=GOOD, ha="right", va="bottom")
    ax.text(0.03, 0.97, "Misaligned\n(niche degree)", transform=ax.transAxes,
            fontsize=8, color=WARN, ha="left", va="top")

    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, pgi_max * 1000))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], shrink=0.35, pad=0.02, aspect=18)
    cbar.set_label("Degree PGI (× 1000)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Panel right: top 15 bar ───────────────────────────────────────────
    ax = axes[1]
    y = np.arange(len(top15))
    bars_r = ax.barh(y, top15["degree_pgi"] * 1000,
                     height=0.70, color=ACCENT, alpha=0.85)

    # Annotate with demand % and misalignment %
    for bar, (_, row) in zip(bars_r, top15.iterrows()):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"demand {row['job_demand_weight']*100:.1f}%  gap {row['misalignment_rate']*100:.0f}%",
            va="center", fontsize=6.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(top15["major"], fontsize=8.5)
    ax.set_xlabel("Degree PGI (× 1000)", fontsize=10, labelpad=8)
    ax.set_title(
        "Top 15 Degrees by Degree PGI\n"
        "(highest demand × highest misalignment)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, top15["degree_pgi"].max() * 1000 * 1.55)

    fig.suptitle(
        "Degree-Level Priority Gap Index\n"
        "= Job Demand Weight × Misalignment Rate",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig_ss_chart3_degree_pgi.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 4 — Degree Alignment Score (3 panels)
# ══════════════════════════════════════════════════════════════════════════

def plot_chart4_degree_alignment(deg: pd.DataFrame) -> None:
    """
    3 panels:
      (a) Histogram distribution of Degree Alignment Scores across all 55 degrees
      (b) Top 10 highest alignment scores (best-aligned degrees — bar)
      (c) Bottom 10 lowest alignment scores (most misaligned — MOE review priority)
    """
    print("  Chart 4 — Degree alignment (3 panels)...")

    top10 = deg.nlargest(10, "alignment_score").sort_values("alignment_score")
    bot10 = deg.nsmallest(10, "alignment_score").sort_values("alignment_score")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                             gridspec_kw={"width_ratios": [1, 1.5, 1.5]})

    # ── Panel (a): distribution ───────────────────────────────────────────
    ax = axes[0]
    ax.hist(deg["alignment_score"], bins=15,
            color=ACCENT, alpha=0.82, edgecolor="white", linewidth=0.6)
    mean_s = deg["alignment_score"].mean()
    ax.axvline(mean_s, color=BAD, linewidth=1.5, linestyle="--")
    ax.text(mean_s + 0.5, ax.get_ylim()[1] * 0.97,
            f"Mean: {mean_s:.1f}",
            fontsize=8, color=BAD, va="top")
    ax.set_xlabel("Degree Alignment Score", fontsize=10, labelpad=6)
    ax.set_ylabel("Number of degree programmes", fontsize=10, labelpad=6)
    ax.set_title("(a) Distribution of\nDegree Alignment Scores\n(all 55 degrees)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.25)
    # Shade bottom-quartile region
    ax.axvspan(0, 25, alpha=0.05, color=BAD, zorder=0)
    ax.text(2, ax.get_ylim()[1] * 0.5, "Review\nPriority",
            fontsize=7.5, color=BAD, alpha=0.7, va="center")

    # ── Panel (b): top 10 (best aligned) ─────────────────────────────────
    ax = axes[1]
    y = np.arange(len(top10))
    bars_t = ax.barh(y, top10["alignment_score"],
                     height=0.65, color=GOOD, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(top10["major"], fontsize=8)
    for bar, val in zip(bars_t, top10["alignment_score"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Degree Alignment Score (0–100)", fontsize=10, labelpad=6)
    ax.set_title("(b) Top 10 — Best Aligned\n(highest score = strong coverage\non both skill + semantic dimensions)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 108)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)

    # ── Panel (c): bottom 10 (most misaligned) ───────────────────────────
    ax = axes[2]
    y2 = np.arange(len(bot10))
    bars_b = ax.barh(y2, bot10["alignment_score"],
                     height=0.65, color=BAD, alpha=0.82)
    ax.set_yticks(y2)
    ax.set_yticklabels(bot10["major"], fontsize=8)
    for bar, val in zip(bars_b, bot10["alignment_score"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Degree Alignment Score (0–100)", fontsize=10, labelpad=6)
    ax.set_title("(c) Bottom 10 — MOE Review Priority\n(lowest score = misaligned on\nboth skill + semantic dimensions)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 108)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)

    fig.suptitle(
        "Degree Curriculum Alignment Score\n"
        "= 0.5 × Semantic Alignment Percentile + 0.5 × Skill Coverage Percentile",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_ss_chart4_degree_alignment.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 5 — Foundational Ratio per faculty
# ══════════════════════════════════════════════════════════════════════════

def plot_chart5_foundational_ratio(ratio: pd.DataFrame) -> None:
    """
    Horizontal bar — Foundational Ratio per degree programme (major).
    Foundational Ratio = share of theoretical course skills with no job-market demand.
    Sorted ascending (lowest ratio at top — most industry-relevant degree first).
    """
    print("  Chart 5 — Foundational Ratio by degree...")

    required = {"major", "foundational_ratio"}
    missing = required - set(ratio.columns)
    if missing:
        raise ValueError(
            f"Chart 5 requires major-level foundational ratio columns {sorted(required)}; "
            f"missing {sorted(missing)}. "
            "Expected source file: foundational_ratio_by_major.csv."
        )

    df = ratio.sort_values("foundational_ratio", ascending=True).copy()
    n = len(df)

    fig, ax = plt.subplots(figsize=(12, max(12, n * 0.32)))
    y = np.arange(n)

    colours = [
        GOOD if v < 0.50 else (WARN if v < 0.70 else BAD)
        for v in df["foundational_ratio"]
    ]

    bars = ax.barh(y, df["foundational_ratio"] * 100,
                   height=0.70, color=colours, alpha=0.85)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{row['foundational_ratio']:.0%}",
                va="center", fontsize=7)

    global_mean = df["foundational_ratio"].mean() * 100
    ax.axvline(global_mean, color=MUTED, linewidth=1.2, linestyle="--", zorder=0)
    ax.text(global_mean + 0.4, n * 0.98, f"Mean: {global_mean:.0f}%",
            fontsize=7.5, color=TEXT_CLR, alpha=0.7, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(df["major"], fontsize=7.5)
    ax.set_xlabel("Foundational Ratio — % of theoretical skills with no job-market demand",
                  fontsize=10, labelpad=8)
    ax.set_title(
        "Foundational Ratio by Degree Programme\n"
        "Higher = more theoretical content with no direct job-market equivalent",
        fontsize=12, fontweight="bold", pad=14,
    )

    band_patches = [
        mpatches.Patch(color=GOOD, alpha=0.85, label="< 50%  — relatively industry-aligned"),
        mpatches.Patch(color=WARN, alpha=0.85, label="50–70% — moderately foundational"),
        mpatches.Patch(color=BAD,  alpha=0.85, label="≥ 70%  — highly foundational"),
    ]
    ax.legend(handles=band_patches, fontsize=8, loc="lower right",
              framealpha=0.9, title="Ratio band", title_fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 108)

    fig.tight_layout()
    _save(fig, "fig_ss_chart5_foundational_ratio.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--job-source", default="skills_list",
                   choices=["skills_list", "skillner"])
    p.add_argument("--threshold", type=float, default=0.72)
    return p.parse_args()


def _t_tag(v: float) -> str:
    return f"{v:.2f}".replace(".", "")


def main() -> None:
    args = parse_args()
    ttag = _t_tag(args.threshold)
    src  = args.job_source

    FIG_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SKILL-SPACE REPORT OUTPUTS — generating figures")
    print(f"  Job source : {src}   Threshold : {args.threshold}")
    print("=" * 70)
    print()

    # ── file paths ──────────────────────────────────────────────────────
    ssoc_minor_path = RESULTS_DIR / f"breakdown_analysis_{src}" / "ssoc_minor.csv"
    major_path      = RESULTS_DIR / f"breakdown_analysis_{src}" / "major.csv"
    deg_path        = RESULTS_DIR / "combined_degree_priority" / "combined_degree_priority.csv"
    ratio_path      = RESULTS_DIR / f"foundational_layer_{src}" / "foundational_ratio_by_major.csv"

    # ── load data ───────────────────────────────────────────────────────
    def _load(path: Path, label: str) -> pd.DataFrame | None:
        if not path.exists():
            print(f"  [skip] {label}: file not found at {path}")
            return None
        df = pd.read_csv(path)
        print(f"  Loaded {label}: {len(df)} rows")
        return df

    ssoc  = _load(ssoc_minor_path, "ssoc_minor.csv")
    major = _load(major_path,      "major.csv")
    deg   = _load(deg_path,        "combined_degree_priority.csv")
    ratio = _load(ratio_path,      "foundational_ratio_by_major.csv")

    # Degree PGI requires the new columns — recompute if missing (backwards compat)
    if deg is not None and "degree_pgi" not in deg.columns:
        print("  [info] degree_pgi not found in CSV — re-run combined_degree_priority.py")
        deg = None

    print()

    # ── generate charts ─────────────────────────────────────────────────
    if ssoc is not None:
        plot_chart1_ssoc_overview(ssoc)

    if major is not None:
        plot_chart2_major_coverage(major)

    if deg is not None:
        plot_chart3_degree_pgi(deg)

    if deg is not None:
        plot_chart4_degree_alignment(deg)

    if ratio is not None:
        plot_chart5_foundational_ratio(ratio)

    print(f"\n{'=' * 70}")
    print(f"DONE — all figures in {FIG_OUT}")
    print(f"{'=' * 70}")
    for f in sorted(FIG_OUT.glob("fig_ss_chart*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
