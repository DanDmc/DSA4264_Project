"""
Generate DSA Deep-Dive summary tables as PNG figures for the report.
Produces three figures:
  fig_dsa_summary_table.png       — headline metrics
  fig_dsa_top5_demanded.png       — top 5 demanded skills
  fig_dsa_top5_semantic.png       — top 5 DSA skills with semantic matches
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_COLOR = "#2c3e50"
HEADER_TEXT  = "white"
ROW_A        = "#f8f9fa"
ROW_B        = "white"
ACCENT       = "#eafaf1"


def style_table(tbl, n_cols, n_rows, header=True, accent_col=None):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dee2e6")
        cell.set_linewidth(0.5)
        if header and row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color=HEADER_TEXT, fontweight="bold")
        else:
            if accent_col is not None and col == accent_col and row > 0:
                cell.set_facecolor(ACCENT)
            elif row % 2 == 1:
                cell.set_facecolor(ROW_A)
            else:
                cell.set_facecolor(ROW_B)


# ── Figure 1: Summary metrics ─────────────────────────────────────────────────
def fig_summary():
    data = [
        ["DSA course skills (unique)",      "342"],
        ["Demanded job skills (unique)",     "1,192"],
        ["Exact match",                      "9 (0.8%)"],
        ["Semantic match (θ = 0.72)",        "6 (0.5%)"],
        ["Genuine gap",                      "1,177 (98.7%)"],
        ["Soft-SCR",                         "1.2%"],
    ]
    col_labels = ["Metric", "Value"]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.axis("off")
    fig.suptitle("DSA Deep-Dive: Coverage Summary", fontsize=12,
                 fontweight="bold", y=0.98)

    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_column_width([0, 1])
    style_table(tbl, 2, len(data))

    # right-align value column
    for (row, col), cell in tbl.get_celld().items():
        if col == 1:
            cell.set_text_props(ha="right")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_dsa_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ── Figure 2: Top 5 demanded skills ──────────────────────────────────────────
def fig_top5_demanded():
    data = [
        ["1", "python",                 "91",  "EXACT"],
        ["2", "information technology", "87",  "NON-EXACT"],
        ["3", "troubleshooting",        "84",  "NON-EXACT"],
        ["4", "sql",                    "74",  "NON-EXACT"],
        ["5", "software development",   "74",  "NON-EXACT"],
    ]
    col_labels = ["Rank", "Job Skill", "Postings", "Status"]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    fig.suptitle("Top 5 Demanded Skills (DSA-Mapped Jobs)", fontsize=12,
                 fontweight="bold", y=0.98)

    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_column_width([0, 1, 2, 3])
    style_table(tbl, 4, len(data))

    # colour EXACT rows green
    for (row, col), cell in tbl.get_celld().items():
        if row > 0 and data[row - 1][3] == "EXACT":
            cell.set_facecolor("#eafaf1")
        if col in (0, 2):
            cell.set_text_props(ha="center")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_dsa_top5_demanded.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ── Figure 3: Top 5 DSA skills with semantic matches ─────────────────────────
def fig_top5_semantic():
    data = [
        ["1", "assessing the precision of estimates", "3", "estimates"],
        ["2", "scalar valued functions",              "2", "scala"],
        ["3", "finance",                              "1", "asset finance"],
        ["4", "big data analysis",                    "1", "large scale data analysis"],
        ["5", "basic learning systems",               "1", "learning management systems"],
    ]
    col_labels = ["Rank", "DSA Skill", "Postings", "Matched Job Skill"]

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    fig.suptitle("Top 5 DSA Skills with Semantic Matches", fontsize=12,
                 fontweight="bold", y=0.98)

    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_column_width([0, 1, 2, 3])
    style_table(tbl, 4, len(data))

    for (row, col), cell in tbl.get_celld().items():
        if col in (0, 2):
            cell.set_text_props(ha="center")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_dsa_top5_semantic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    fig_summary()
    fig_top5_demanded()
    fig_top5_semantic()
    print("Done.")
