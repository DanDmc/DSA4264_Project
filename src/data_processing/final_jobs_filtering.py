"""
final_jobs_filtering.py
=======================
Step 3 of the jobs pipeline.

Filters SSOC-mapped jobs to entry-level roles based on
MAX_YEARS_EXPERIENCE (default: 2) and a minimum salary floor
MIN_SALARY_AVG (default: 3500). Both thresholds are configurable
in src/config.py so MOE officers can adjust them without editing
pipeline code.

Before each filter step a distribution chart is saved to
outputs/jobs_filtering_visualizations/ to justify the cutoff choices.

Usage (from repo root):
    python -m src.data_processing.final_jobs_filtering
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Import shared paths and parameters from project config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import JOBS_SSOC_MAPPED, JOBS_FILTERED, MAX_YEARS_EXPERIENCE, MIN_SALARY_AVG

# Output directory
VIZ_DIR = Path(__file__).resolve().parents[2] / "outputs" / "jobs_filtering_visualizations"

# Salary cap for cleaner plots
SALARY_CAP = 8000

# ── Chart style constants ──────────────────────────────────────────────────────
ACCENT   = "#2563EB"
MUTED    = "#CBD5E1"
CUT_CLR  = "#EF4444"
BG       = "#F8FAFC"
TEXT_CLR = "#1E293B"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
    "text.color":        TEXT_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
})


def _bar_labels(ax, bars, total, pct_only=False):
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        pct = 100 * h / total
        label = f"{pct:.1f}%" if pct_only else f"{int(h):,}\n({pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + total * 0.005,
            label,
            ha="center", va="bottom",
            fontsize=8.5, color=TEXT_CLR,
        )


def plot_experience_distribution(df: pd.DataFrame, cutoff: int, out_path: Path) -> None:
    col = "minimum_years_experience"
    series = df[col].dropna()
    total  = len(series)

    bin_edges  = [0, 2, 4, 6, 8, 10, series.max() + 1]
    bin_labels = ["0 – 2", "3 – 4", "5 – 6", "7 – 8", "9 – 10", "11+"]
    counts, _ = np.histogram(series, bins=bin_edges)

    # Retained vs excluded counts
    retained = (series <= cutoff).sum()
    excluded = total - retained

    colors = [ACCENT if right <= cutoff else MUTED for right in bin_edges[1:]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(bin_labels, counts, color=colors, width=0.6, zorder=2)

    _bar_labels(ax, bars, total)

    ax.axvline(x=0.5, color=CUT_CLR, linewidth=1.6, linestyle="--", zorder=3)

    ax.set_xlabel("Minimum Years of Experience Required", labelpad=8)
    ax.set_ylabel("Number of Jobs")
    ax.set_title(
        f"Distribution of Experience Requirements (n={total:,})",
        pad=14, fontsize=12, fontweight="bold",
    )

    # Legend with counts + %
    legend_labels = [
        f"Retained (≤{cutoff} yrs): {retained:,} ({retained/total*100:.1f}%)",
        f"Excluded: {excluded:,} ({excluded/total*100:.1f}%)"
    ]
    ax.legend(legend_labels, loc="upper right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Saved experience distribution → {out_path}")


def plot_salary_distribution(df: pd.DataFrame, cutoff: int, out_path: Path) -> None:
    col = "salary_avg"
    series = df[col].dropna()

    # Apply cap
    series = series.clip(upper=SALARY_CAP)
    total  = len(series)

    max_sal   = SALARY_CAP
    bin_edges = list(range(0, max_sal + 500, 500))
    counts, _ = np.histogram(series, bins=bin_edges)

    bin_mids = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))]

    # Custom labels with last bin as 8k+
    bin_labels = []
    for i in range(len(counts)):
        if i == len(counts) - 1:
            bin_labels.append(f"${SALARY_CAP:,}+")
        else:
            bin_labels.append(f"${bin_edges[i]:,}–\n${bin_edges[i+1]:,}")

    # Retained vs excluded counts (based on REAL salary, not capped)
    raw_series = df[col].dropna()
    retained = (raw_series >= cutoff).sum()
    excluded = total - retained

    colors = [MUTED if mid < cutoff else ACCENT for mid in bin_mids]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(counts)), counts, color=colors, width=0.7, zorder=2)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(
        [lbl if i % 2 == 0 else "" for i, lbl in enumerate(bin_labels)],
        fontsize=7.5,
    )

    _bar_labels(ax, bars, total)

    cutoff_x = (cutoff - bin_edges[0]) / 500 - 0.5
    ax.axvline(x=cutoff_x, color=CUT_CLR, linewidth=1.6, linestyle="--", zorder=3)

    ax.set_xlabel("Average Monthly Salary (SGD)", labelpad=8)
    ax.set_ylabel("Number of Jobs")
    ax.set_title(
        f"Distribution of Average Monthly Salaries (n={total:,})",
        pad=14, fontsize=12, fontweight="bold",
    )

    # Legend with counts + %
    legend_labels = [
        f"Retained (≥${cutoff:,}): {retained:,} ({retained/total*100:.1f}%)",
        f"Excluded: {excluded:,} ({excluded/total*100:.1f}%)"
    ]
    ax.legend(legend_labels, loc="upper right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Saved salary distribution    → {out_path}")


def main():
    print(f"Loading jobs from {JOBS_SSOC_MAPPED}")
    df = pd.read_csv(JOBS_SSOC_MAPPED)
    print(f"  → {len(df):,} jobs before filtering")

    plot_experience_distribution(
        df,
        cutoff=MAX_YEARS_EXPERIENCE,
        out_path=VIZ_DIR / "01_experience_distribution.png",
    )

    filtered_df = df[df["minimum_years_experience"] <= MAX_YEARS_EXPERIENCE]
    print(f"  → {len(filtered_df):,} jobs after experience filter")

    plot_salary_distribution(
        filtered_df,
        cutoff=MIN_SALARY_AVG,
        out_path=VIZ_DIR / "02_salary_distribution.png",
    )

    filtered_df = filtered_df[filtered_df["salary_avg"] >= MIN_SALARY_AVG]
    print(f"  → {len(filtered_df):,} jobs after salary filter")

    JOBS_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(JOBS_FILTERED, index=False)
    print(f"Saved to {JOBS_FILTERED}")


if __name__ == "__main__":
    main()