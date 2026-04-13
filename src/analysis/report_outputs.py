"""
report_outputs.py — Generate report-ready figures and summary tables
---------------------------------------------------------------------
Reads the CSV outputs from similarity_analysis.py and produces:
  1. Degree summary table (top 10, bottom 10, full — as CSV + Markdown)
  2. Scatter plot: alignment depth vs. breadth (degree level)
  3. Horizontal bar chart: coverage % by degree
  4. Targeted alignment ratio dot plot
  5. Similarity distribution histogram with threshold line

This is intentionally separate from similarity_analysis.py so we can
iterate on visualisations without recomputing the analysis.

Usage (from repo root):
    python -m src.analysis.report_outputs

Outputs land in: REPO_ROOT/outputs/report_figures/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── project config ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBEDDING_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_IN = REPO_ROOT / "outputs" / "similarity_analysis_outputs"
FIG_OUT = REPO_ROOT / "outputs" / "report_figures"


# ──────────────────────────────────────────────
# Chart style — clean, slide-ready, consistent
# with final_jobs_filtering.py palette
# ──────────────────────────────────────────────
ACCENT   = "#2563EB"   # blue — primary
ACCENT2  = "#7C3AED"   # purple — secondary
MUTED    = "#CBD5E1"   # light grey
GOOD     = "#10B981"   # green — for highlights
WARN     = "#F59E0B"   # amber — for warnings
BAD      = "#EF4444"   # red — for low performers
BG       = "#F8FAFC"   # off-white background
TEXT_CLR = "#1E293B"   # dark slate text

# Faculty colour map — consistent across all charts
FACULTY_COLOURS = {
    "School of Computing":               "#2563EB",
    "NUS Business School":               "#7C3AED",
    "College of Design and Engineering":  "#0891B2",
    "Faculty of Science":                 "#059669",
    "Faculty of Arts and Social Sciences":"#D97706",
    "Faculty of Law":                     "#DC2626",
    "Yong Loo Lin School of Medicine":    "#DB2777",
}
DEFAULT_COLOUR = "#64748B"

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


def _faculty_color(faculty):
    """Get consistent colour for a faculty, with fallback."""
    return FACULTY_COLOURS.get(faculty, DEFAULT_COLOUR)


# ══════════════════════════════════════════════
# 1. DEGREE SUMMARY TABLE
# ══════════════════════════════════════════════

def generate_degree_summary_table(degree_df, targeted_df):
    """
    Produce three versions of the degree summary table:
      - Full CSV (all degrees, for appendix)
      - Top 10 + bottom 10 CSV (for report body)
      - Markdown version of top/bottom 10 (for MkDocs rendering)

    Merges in targeted alignment ratio from targeted_ssoc_alignment.csv
    so that one table captures all key metrics.
    """
    print("  Generating degree summary tables...")

    # Merge in targeted ratio if available
    if targeted_df is not None:
        ratio_cols = targeted_df[["major", "targeted_vs_overall_ratio"]].copy()
        merged = degree_df.merge(ratio_cols, on="major", how="left")
    else:
        merged = degree_df.copy()
        merged["targeted_vs_overall_ratio"] = np.nan

    # Select and rename columns for the report table
    table = merged[[
        "major", "degree_faculty",
        "top_k_mean_sim", "rank_top_k_sim",
        "breadth_n_ssoc_groups", "rank_breadth",
        "coverage_pct", "rank_coverage",
        "targeted_vs_overall_ratio",
        "n_modules_matched",
    ]].copy()

    table = table.rename(columns={
        "major": "Degree",
        "degree_faculty": "Faculty",
        "top_k_mean_sim": "Alignment",
        "rank_top_k_sim": "Align Rank",
        "breadth_n_ssoc_groups": "Breadth",
        "rank_breadth": "Breadth Rank",
        "coverage_pct": "Coverage %",
        "rank_coverage": "Cov Rank",
        "targeted_vs_overall_ratio": "Target Ratio",
        "n_modules_matched": "Modules",
    })

    table = table.sort_values("Align Rank")

    # Full table (appendix)
    table.to_csv(FIG_OUT / "degree_summary_full.csv", index=False)

    # Top 10 + bottom 10 (report body)
    top10 = table.head(10)
    bottom10 = table.tail(10)
    highlight = pd.concat([top10, bottom10]).drop_duplicates()
    highlight.to_csv(FIG_OUT / "degree_summary_top_bottom.csv", index=False)

    # Markdown version
    md_lines = ["# Degree Alignment Summary (Top 10 & Bottom 10)\n"]
    md_lines.append("| Degree | Faculty | Alignment | Rank | Breadth | Coverage % | Target Ratio |")
    md_lines.append("|--------|---------|-----------|------|---------|------------|--------------|")
    for _, r in highlight.iterrows():
        ratio = f"{r['Target Ratio']:.3f}" if pd.notna(r['Target Ratio']) else "—"
        md_lines.append(
            f"| {r['Degree']} | {r['Faculty'][:25]} | "
            f"{r['Alignment']:.4f} | {int(r['Align Rank'])} | "
            f"{int(r['Breadth'])} | {r['Coverage %']:.1f} | {ratio} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (FIG_OUT / "degree_summary_table.md").write_text(md_text)

    print(f"    → {FIG_OUT / 'degree_summary_full.csv'}")
    print(f"    → {FIG_OUT / 'degree_summary_top_bottom.csv'}")
    print(f"    → {FIG_OUT / 'degree_summary_table.md'}")

    return table


# ══════════════════════════════════════════════
# 2. SCATTER: ALIGNMENT DEPTH VS. BREADTH
# ══════════════════════════════════════════════

def plot_depth_vs_breadth(degree_df):
    """
    Scatter plot of alignment depth (top-K mean sim) vs. breadth
    (distinct SSOC groups) at the degree level. Colour-coded by faculty.
    Labels each dot with degree name for readability.

    This is the "specialist vs. generalist" chart — arguably the single
    most informative visual for MOE.
    """
    print("  Generating depth vs. breadth scatter...")

    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot each degree as a dot
    for _, row in degree_df.iterrows():
        colour = _faculty_color(row["degree_faculty"])
        ax.scatter(
            row["top_k_mean_sim"], row["breadth_n_ssoc_groups"],
            c=colour, s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
            zorder=3,
        )
        # Label — offset slightly to avoid overlapping the dot
        ax.annotate(
            row["major"],
            (row["top_k_mean_sim"], row["breadth_n_ssoc_groups"]),
            fontsize=6.5, alpha=0.85,
            xytext=(5, 3), textcoords="offset points",
        )

    ax.set_xlabel("Alignment Depth (Top-K Mean Similarity)", fontsize=11, labelpad=8)
    ax.set_ylabel("Alignment Breadth (Distinct SSOC Minor Groups in Top-K)", fontsize=11, labelpad=8)
    ax.set_title("Degree Alignment: Depth vs. Breadth", fontsize=13, fontweight="bold", pad=14)

    # Faculty legend
    handles = []
    for fac, colour in FACULTY_COLOURS.items():
        # Only include faculties present in the data
        if fac in degree_df["degree_faculty"].values:
            handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=colour, markersize=8, label=fac))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "depth_vs_breadth_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {FIG_OUT / 'depth_vs_breadth_scatter.png'}")


# ══════════════════════════════════════════════
# 3. BAR CHART: COVERAGE % BY DEGREE
# ══════════════════════════════════════════════

def plot_coverage_bars(degree_df):
    """
    Horizontal bar chart of coverage % by degree, sorted descending.
    Coloured by faculty. Shows the share of "good" entry-level jobs
    each degree is meaningfully aligned with.
    """
    print("  Generating coverage bar chart...")

    df = degree_df.sort_values("coverage_pct", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 14))

    colours = [_faculty_color(f) for f in df["degree_faculty"]]
    bars = ax.barh(range(len(df)), df["coverage_pct"], color=colours, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["major"], fontsize=7.5)
    ax.set_xlabel("Job Market Coverage (%)", fontsize=11, labelpad=8)
    ax.set_title("Degree-Level Job Market Coverage", fontsize=13, fontweight="bold", pad=14)

    # Add value labels at the end of each bar
    for bar, val in zip(bars, df["coverage_pct"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=6.5)

    # Faculty legend
    handles = []
    for fac, colour in FACULTY_COLOURS.items():
        if fac in df["degree_faculty"].values:
            handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                       markerfacecolor=colour, markersize=8, label=fac))
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "coverage_by_degree.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {FIG_OUT / 'coverage_by_degree.png'}")


# ══════════════════════════════════════════════
# 4. TARGETED ALIGNMENT RATIO DOT PLOT
# ══════════════════════════════════════════════

def plot_targeted_ratio(targeted_df):
    """
    Dot plot showing targeted-vs-overall alignment ratio for each degree.
    A vertical line at ratio=1.0 shows the breakeven point.

    Ratio > 1: degree is more aligned with its intended career tracks
    Ratio < 1: strongest matches are in unexpected fields
    """
    if targeted_df is None or len(targeted_df) == 0:
        print("  Targeted alignment: no data — skipping chart.")
        return

    print("  Generating targeted alignment ratio chart...")

    df = targeted_df.sort_values("targeted_vs_overall_ratio", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 14))

    # Colour: green if ratio >= 1, amber if < 1
    colours = [GOOD if r >= 1.0 else WARN for r in df["targeted_vs_overall_ratio"]]

    ax.scatter(df["targeted_vs_overall_ratio"], range(len(df)),
               c=colours, s=50, zorder=3, edgecolors="white", linewidths=0.5)

    # Horizontal lines connecting dots to the axis for readability
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([1.0, row["targeted_vs_overall_ratio"]], [i, i],
                color=colours[i], alpha=0.3, linewidth=1.5)

    # Reference line at ratio = 1.0
    ax.axvline(x=1.0, color=TEXT_CLR, linewidth=1, linestyle="--", alpha=0.5, zorder=1)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["major"], fontsize=7.5)
    ax.set_xlabel("Targeted / Overall Alignment Ratio", fontsize=11, labelpad=8)
    ax.set_title("Targeted SSOC Alignment Ratio by Degree",
                 fontsize=13, fontweight="bold", pad=14)

    # Value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["targeted_vs_overall_ratio"] + 0.005, i,
                f"{row['targeted_vs_overall_ratio']:.3f}",
                va="center", fontsize=6.5)

    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "targeted_alignment_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {FIG_OUT / 'targeted_alignment_ratio.png'}")


# ══════════════════════════════════════════════
# 5. SIMILARITY DISTRIBUTION HISTOGRAM
# ══════════════════════════════════════════════

def plot_similarity_distribution(metadata):
    """
    Histogram of global similarity score distribution with threshold
    line marked. Uses pre-computed stats from analysis_metadata.json
    (we don't reload the full sim matrix here).

    Since we only have summary stats and not the raw distribution,
    we plot a normal approximation + annotated threshold line. If
    you want the exact histogram, run this on the sim matrix directly.
    """
    print("  Generating similarity distribution chart...")

    stats = metadata["distribution_stats"]
    mean = stats["global_mean"]
    std = stats["global_std"]
    threshold = stats["coverage_threshold"]
    pmin = stats["global_min"]
    pmax = stats["global_max"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Normal approximation of the distribution
    x = np.linspace(max(pmin, mean - 4*std), min(pmax, mean + 4*std), 500)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.fill_between(x, y, alpha=0.3, color=ACCENT)
    ax.plot(x, y, color=ACCENT, linewidth=1.5)

    # Threshold line
    ax.axvline(x=threshold, color=BAD, linewidth=1.5, linestyle="--", zorder=3)
    ax.text(threshold + 0.003, ax.get_ylim()[1] * 0.85,
            f"Threshold\n{threshold:.4f}\n(mean + 1 SD)",
            fontsize=8, color=BAD, fontweight="bold")

    # Mean line
    ax.axvline(x=mean, color=TEXT_CLR, linewidth=1, linestyle=":", alpha=0.5)
    ax.text(mean - 0.003, ax.get_ylim()[1] * 0.7, f"Mean\n{mean:.4f}",
            fontsize=7.5, ha="right", alpha=0.7)

    # Percentile annotations
    for p_label, p_val in stats["percentiles"].items():
        if int(p_label) in [25, 75, 95]:
            ax.axvline(x=p_val, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Cosine Similarity", fontsize=11, labelpad=8)
    ax.set_ylabel("Density (approx.)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Global Similarity Distribution "
        f"({stats['matrix_shape'][0]} modules × {stats['matrix_shape'][1]} jobs)",
        fontsize=12, fontweight="bold", pad=14,
    )

    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "similarity_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {FIG_OUT / 'similarity_distribution.png'}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("REPORT OUTPUTS — Generating figures & summary tables")
    print("=" * 70)

    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load pre-computed analysis CSVs ──
    degree_path = CSV_IN / "degree_summary.csv"
    targeted_path = CSV_IN / "targeted_ssoc_alignment.csv"
    metadata_path = CSV_IN / "analysis_metadata.json"

    if not degree_path.exists():
        print(f"ERROR: {degree_path} not found.")
        print("Run similarity_analysis.py first.")
        sys.exit(1)

    degree_df = pd.read_csv(degree_path)
    print(f"  Loaded degree_summary.csv: {len(degree_df)} degrees")

    targeted_df = None
    if targeted_path.exists():
        targeted_df = pd.read_csv(targeted_path)
        print(f"  Loaded targeted_ssoc_alignment.csv: {len(targeted_df)} degrees")

    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  Loaded analysis_metadata.json")

    # ── Generate outputs ──
    print()

    # 1. Summary table
    generate_degree_summary_table(degree_df, targeted_df)

    # 2. Depth vs. breadth scatter
    plot_depth_vs_breadth(degree_df)

    # 3. Coverage bar chart
    plot_coverage_bars(degree_df)

    # 4. Targeted alignment ratio
    plot_targeted_ratio(targeted_df)

    # 5. Similarity distribution (needs metadata)
    if metadata is not None:
        plot_similarity_distribution(metadata)
    else:
        print("  Similarity distribution: metadata not found — skipping.")

    print(f"\n{'=' * 70}")
    print(f"DONE — all outputs in {FIG_OUT}")
    print(f"{'=' * 70}")
    for f in sorted(FIG_OUT.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
