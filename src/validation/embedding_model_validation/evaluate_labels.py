"""
Evaluate Human Labels vs Embedding Scores
-------------------------------------------
Reads the filled-in labelling_sheet.xlsx (with your 0/1/2 labels),
compares them against cosine scores, and produces:

1. Spearman rank correlation (the main number)
2. Score distributions per label group (do the groups separate?)
3. Per-category breakdown
4. A summary JSON for the report

The key question: does higher cosine score reliably correspond to
higher human-judged relevance? If Spearman rho > ~0.4–0.5, the
embedding is doing something meaningful. If it's near 0, the scores
are noise.

Usage:
  python evaluate_labels.py

Expects labelling_sheet.xlsx with the 'label' column filled in.
Produces:
  - labelling_evaluation.json (full results)
  - labelling_score_distributions.png (visual sanity check)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import warnings

SCRIPT_DIR = Path(__file__).parent
LABELLING_FILE = SCRIPT_DIR / "labelling_sheet.xlsx"
OUTPUT_JSON = SCRIPT_DIR / "labelling_evaluation.json"
OUTPUT_CHART = SCRIPT_DIR / "labelling_score_distributions.png"


def load_labels():
    """Load the labelling sheet and validate that labels are filled in."""
    df = pd.read_excel(LABELLING_FILE, sheet_name="labelling")

    # check that labels exist
    if "label" not in df.columns:
        raise ValueError("No 'label' column found in labelling sheet")

    # coerce to numeric, flag any non-numeric
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    n_missing = df["label"].isna().sum()

    if n_missing > 0:
        print(f"  Warning: {n_missing} pairs have no label — these will be excluded")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # validate label values
    invalid = df[~df["label"].isin([0, 1, 2])]
    if len(invalid) > 0:
        print(f"  Warning: {len(invalid)} labels outside {{0,1,2}} — excluding")
        df = df[df["label"].isin([0, 1, 2])]

    print(f"  Loaded {len(df)} labelled pairs")
    print(f"  Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    return df


def compute_spearman(df):
    """
    Spearman rank correlation between cosine score and human label.

    Why Spearman and not Pearson? We care about monotonic relationship
    (does higher score = higher label?), not linear relationship.
    Labels are ordinal (0 < 1 < 2) not interval — Spearman is the
    right test for this.

    Interpretation:
      rho > 0.5  → good alignment, embedding is meaningful
      rho 0.3–0.5 → moderate, embedding captures some signal
      rho < 0.3  → weak, embedding may not be reliable for this use case
    """
    rho, p_value = spearmanr(df["cosine_score"], df["label"])
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p_value), 6),
        "n_pairs": len(df),
        "interpretation": (
            "strong" if abs(rho) > 0.5
            else "moderate" if abs(rho) > 0.3
            else "weak"
        ),
    }


def compute_score_distributions(df):
    """
    Summary stats for cosine scores within each label group.

    The ideal outcome: label=2 scores are clearly higher than label=1,
    which are clearly higher than label=0, with minimal overlap.
    """
    distributions = {}
    for label in [0, 1, 2]:
        scores = df[df["label"] == label]["cosine_score"]
        if len(scores) == 0:
            continue
        distributions[str(label)] = {
            "count": int(len(scores)),
            "mean": round(float(scores.mean()), 4),
            "median": round(float(scores.median()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        }

    # overlap check: do any label=0 pairs score higher than any label=2 pairs?
    scores_0 = df[df["label"] == 0]["cosine_score"]
    scores_2 = df[df["label"] == 2]["cosine_score"]
    if len(scores_0) > 0 and len(scores_2) > 0:
        # fraction of label=0 scores above the label=2 median
        overlap = float((scores_0 > scores_2.median()).mean())
        distributions["overlap_0_above_2_median"] = round(overlap, 4)

    return distributions


def compute_per_category(df):
    """Per-category Spearman correlation (where sample size allows)."""
    per_cat = {}
    for cat in sorted(df["job_category"].unique()):
        cat_df = df[df["job_category"] == cat]
        if len(cat_df) < 4 or cat_df["label"].nunique() < 2:
            # too few points or no label variance — can't compute correlation
            per_cat[cat] = {
                "n_pairs": len(cat_df),
                "spearman_rho": None,
                "note": "insufficient data or label variance for correlation",
            }
            continue

        rho, p = spearmanr(cat_df["cosine_score"], cat_df["label"])
        per_cat[cat] = {
            "n_pairs": len(cat_df),
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 4),
        }
    return per_cat


def create_chart(df):
    """
    Box plot of cosine scores by label group.
    Simple visual: if the boxes don't overlap much, the embedding works.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))

        label_groups = [0, 1, 2]
        data = [df[df["label"] == l]["cosine_score"].values for l in label_groups]
        labels_str = [
            f"0 — No relevance\n(n={len(data[0])})",
            f"1 — Partial\n(n={len(data[1])})",
            f"2 — Strong\n(n={len(data[2])})",
        ]

        bp = ax.boxplot(
            data,
            labels=labels_str,
            patch_artist=True,
            widths=0.5,
        )

        colors = ["#fee2e2", "#fef3c7", "#d1fae5"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Cosine Similarity Score")
        ax.set_xlabel("Human Relevance Label")
        ax.set_title("Embedding Score vs Human Judgment")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_CHART, dpi=150)
        plt.close()
        print(f"  Saved chart: {OUTPUT_CHART}")

    except ImportError:
        print("  matplotlib not available — skipping chart")


def main():
    print("Evaluating human labels")
    print("=" * 50)

    df = load_labels()

    spearman = compute_spearman(df)
    distributions = compute_score_distributions(df)
    per_category = compute_per_category(df)

    output = {
        "spearman_correlation": spearman,
        "score_distributions_by_label": distributions,
        "per_category": per_category,
        "label_distribution": df["label"].value_counts().sort_index().to_dict(),
        "sample_type_distribution": df["sample_type"].value_counts().to_dict(),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved results: {OUTPUT_JSON}")

    create_chart(df)

    # print summary
    print(f"\n{'─' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Spearman rho: {spearman['spearman_rho']}")
    print(f"  p-value:      {spearman['p_value']}")
    print(f"  Strength:     {spearman['interpretation']}")
    print()
    print("  Score distributions by label:")
    for label, stats in distributions.items():
        if label.startswith("overlap"):
            continue
        print(f"    Label {label}: mean={stats['mean']:.4f}, median={stats['median']:.4f} (n={stats['count']})")
    if "overlap_0_above_2_median" in distributions:
        print(f"\n  Overlap: {distributions['overlap_0_above_2_median']:.1%} of label=0 scores exceed label=2 median")


if __name__ == "__main__":
    main()
