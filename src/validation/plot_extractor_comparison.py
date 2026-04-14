from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_plot(summary_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_csv)

    required = {"method", "macro_precision", "macro_recall", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")

    label_map = {
        "skillner_only": "Initial model\n(SkillNer only)",
        "llm_fewshot": "Final model\n(LLM few-shot)",
    }
    metric_labels = ["Precision", "Recall", "F1"]
    metric_cols = ["macro_precision", "macro_recall", "macro_f1"]
    colors = ["#5B8FF9", "#5AD8A6", "#F6BD16"]

    labels = [label_map.get(method, method.replace("_", " ").title()) for method in df["method"]]
    x = range(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (metric_col, metric_label, color) in enumerate(zip(metric_cols, metric_labels, colors)):
        offsets = [pos + (idx - 1) * width for pos in x]
        values = df[metric_col].tolist()
        bars = ax.bar(offsets, values, width=width, label=metric_label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Initial vs Final Skill Extractor Performance", pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot extractor comparison metrics from the summary CSV."
    )
    parser.add_argument(
        "--summary-csv",
        default="results/extractor_comparison_13_summary.csv",
        help="Path to extractor comparison summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/extractor_comparison_13_metrics.png",
        help="Path to save the output PNG.",
    )
    args = parser.parse_args()

    build_plot(Path(args.summary_csv), Path(args.output))


if __name__ == "__main__":
    main()
