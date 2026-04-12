"""
Threshold Calibration — Step 2: Find optimal threshold from labelled pairs.

What this script does
---------------------
Loads the human-labelled (course_skill, job_skill) pairs and sweeps
cosine similarity thresholds to find the value that maximises F1
against human judgement of "same skill or not."

This empirically justifies the threshold used in improved_scr.py
rather than relying on a judgement call.

Outputs
-------
results/threshold_calibration/
    calibration_results.csv     — F1, precision, recall at each threshold
    calibration_summary.json    — optimal threshold and metrics
    fig_calibration_curve.png   — F1/precision/recall vs threshold plot

Usage
-----
    python -m src.validation.threshold_calibration.calibrate_threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import RESULTS_DIR

LABELLED_PATH = Path("/Users/mv/OneDrive/DSA4264_Project_Data/validation_skillmetrics/threshold_calibration_pairs.xlsx")
OUTPUT_DIR    = RESULTS_DIR / "threshold_calibration"


def load_labelled_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="pairs_to_label")
    # Drop rows where label wasn't filled in
    df = df[df["label"].isin([0, 1, "0", "1"])].copy()
    df["label"] = df["label"].astype(int)
    print(f"  Loaded {len(df)} labelled pairs")
    print(f"  Same (1): {df['label'].sum()}  |  Different (0): {(df['label'] == 0).sum()}")
    return df


def sweep_thresholds(
    df: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    For each threshold, classify pairs as same (pred=1) if similarity >= threshold.
    Compute precision, recall, F1 against human labels.
    """
    rows = []
    y_true = df["label"].values
    sims   = df["similarity"].values

    for t in thresholds:
        y_pred = (sims >= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 3),
            "f1":        round(f1,        4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "accuracy":  round(accuracy,  4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)


def plot_calibration_curve(results: pd.DataFrame, optimal_t: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results["threshold"], results["f1"],        marker="o", linewidth=2,
            label="F1", color="#27ae60")
    ax.plot(results["threshold"], results["precision"], marker="s", linewidth=2,
            label="Precision", color="#2980b9", linestyle="--")
    ax.plot(results["threshold"], results["recall"],    marker="^", linewidth=2,
            label="Recall", color="#e74c3c", linestyle="--")

    ax.axvline(optimal_t, color="#f39c12", linewidth=2, linestyle=":",
               label=f"Optimal θ = {optimal_t:.3f}")

    ax.set_xlabel("Cosine similarity threshold (θ)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Threshold Calibration: F1, Precision, Recall vs θ\n"
        "Optimal threshold selected by maximum F1 against human-labelled pairs",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find optimal similarity threshold from human-labelled pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--labelled",   type=Path,  default=LABELLED_PATH)
    p.add_argument("--output",     type=Path,  default=OUTPUT_DIR)
    p.add_argument("--thresh-min", type=float, default=0.60)
    p.add_argument("--thresh-max", type=float, default=0.95)
    p.add_argument("--thresh-step",type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading labelled pairs …")
    df = load_labelled_pairs(args.labelled)

    print("\nSweeping thresholds …")
    thresholds = np.round(
        np.arange(args.thresh_min, args.thresh_max + 1e-9, args.thresh_step), 3
    )
    results = sweep_thresholds(df, thresholds)

    # Optimal threshold = maximises F1
    best_idx = results["f1"].idxmax()
    best_row = results.loc[best_idx]
    optimal_t = float(best_row["threshold"])

    print(f"\n{'='*56}")
    print("  THRESHOLD CALIBRATION RESULTS")
    print(f"{'='*56}")
    print(f"  Optimal threshold : {optimal_t:.3f}")
    print(f"  F1                : {best_row['f1']:.4f}")
    print(f"  Precision         : {best_row['precision']:.4f}")
    print(f"  Recall            : {best_row['recall']:.4f}")
    print(f"  Accuracy          : {best_row['accuracy']:.4f}")
    print(f"  TP={int(best_row['tp'])}  FP={int(best_row['fp'])}  "
          f"FN={int(best_row['fn'])}  TN={int(best_row['tn'])}")
    print(f"{'='*56}")

    # Save outputs
    results.to_csv(args.output / "calibration_results.csv", index=False)
    print("\nSaved: calibration_results.csv")

    summary = {
        "n_labelled_pairs":    len(df),
        "n_same":              int(df["label"].sum()),
        "n_different":         int((df["label"] == 0).sum()),
        "optimal_threshold":   optimal_t,
        "f1_at_optimal":       float(best_row["f1"]),
        "precision_at_optimal":float(best_row["precision"]),
        "recall_at_optimal":   float(best_row["recall"]),
        "accuracy_at_optimal": float(best_row["accuracy"]),
        "threshold_range_swept": [args.thresh_min, args.thresh_max],
        "note": (
            f"Threshold calibrated against {len(df)} human-labelled "
            "course-skill / job-skill pairs. Optimal value maximises F1."
        ),
    }
    with (args.output / "calibration_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: calibration_summary.json")

    plot_calibration_curve(results, optimal_t,
                           args.output / "fig_calibration_curve.png")

    print(f"\nOutputs written to: {args.output}")
    print(f"\nNext step: update DEFAULT_THRESHOLD in improved_scr.py to {optimal_t:.3f}")


if __name__ == "__main__":
    main()
