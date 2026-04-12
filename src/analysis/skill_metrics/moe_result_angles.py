"""Generate MOE-oriented result angles from existing skill_metrics outputs.

This script does not rerun core methodology. It reframes existing outputs into
policy-facing tables for interpretation and action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import RESULTS_DIR


def _t_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build MOE-facing result angles from existing outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skills_list"])
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _load_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_action_matrix(cat: pd.DataFrame) -> pd.DataFrame:
    # Split by median priority index; confidence from existing High/Medium/Low tags
    med_priority = float(cat["priority_gap_index"].median())

    def decide_action(row: pd.Series) -> str:
        high_p = float(row["priority_gap_index"]) >= med_priority
        conf = str(row["confidence_flag"])

        if high_p and conf == "High":
            return "Act Now"
        if high_p and conf in {"Medium", "Low"}:
            return "Investigate First"
        if (not high_p) and conf == "High":
            return "Monitor"
        return "Low Priority"

    out = cat.copy()
    out["priority_band"] = np.where(out["priority_gap_index"] >= med_priority, "High", "Lower")
    out["recommended_action"] = out.apply(decide_action, axis=1)
    out["action_rationale"] = (
        "Priority=" + out["priority_band"] + "; Confidence=" + out["confidence_flag"].astype(str)
    )
    return out.sort_values(["recommended_action", "priority_gap_index"], ascending=[True, False]).reset_index(drop=True)


def build_lift_table(base: pd.DataFrame, imp: pd.DataFrame, depth: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    b = base[["category", "scr", "dw_scr"]].rename(columns={"scr": "baseline_scr_exact", "dw_scr": "baseline_dw_scr"})
    i = imp[["category", "soft_idf_scr", "baseline_scr"]].rename(
        columns={"soft_idf_scr": "improved_soft_idf_scr", "baseline_scr": "baseline_scr_recomputed"}
    )
    d = depth[["category", "mean_modules_per_covered_skill"]]
    c = conf[["category", "confidence_flag", "ci_width"]]

    out = b.merge(i, on="category", how="inner").merge(d, on="category", how="left").merge(c, on="category", how="left")
    out["absolute_lift_pp"] = (out["improved_soft_idf_scr"] - out["baseline_scr_exact"]) * 100.0
    out["relative_lift_pct"] = np.where(
        out["baseline_scr_exact"] > 0,
        (out["improved_soft_idf_scr"] / out["baseline_scr_exact"] - 1.0) * 100.0,
        np.nan,
    )
    return out.sort_values("absolute_lift_pp", ascending=False).reset_index(drop=True)


def build_threshold_policy_view(sweep: pd.DataFrame) -> pd.DataFrame:
    # Keep interpretable policy checkpoints
    checkpoints = [0.70, 0.85, 0.90]
    out = sweep[sweep["threshold"].round(2).isin(checkpoints)].copy()

    label_map = {
        0.70: "Exploratory / permissive",
        0.85: "Primary / validated",
        0.90: "Ultra-conservative",
    }
    out["policy_label"] = out["threshold"].round(2).map(label_map)
    return out[[
        "threshold",
        "policy_label",
        "macro_soft_idf_scr",
        "micro_soft_idf_scr",
        "pct_genuine_gaps",
        "n_genuine_gaps",
    ]].sort_values("threshold").reset_index(drop=True)


def build_exec_summary(
    improved_summary: dict,
    null_summary: dict,
    mismatch_summary: dict,
    foundational_summary: dict,
    action_matrix: pd.DataFrame,
) -> dict:
    act_now_n = int((action_matrix["recommended_action"] == "Act Now").sum())
    inv_first_n = int((action_matrix["recommended_action"] == "Investigate First").sum())

    return {
        "method_scope": {
            "job_source": "skills_list",
            "primary_threshold": float(improved_summary.get("threshold", 0.85)),
        },
        "coverage_primary": {
            "baseline_macro_scr": float(improved_summary["baseline_macro_scr"]),
            "improved_macro_soft_idf_scr": float(improved_summary["soft_idf_macro_scr"]),
            "genuine_gap_pct": float(improved_summary["pct_genuine_gaps"]),
        },
        "quality_signals": {
            "null_model_z_score": float(null_summary.get("macro_z_score", np.nan)),
            "null_model_p_value": float(null_summary.get("macro_p_value", np.nan)),
            "vocab_mismatch_pct_above_0_70": float(mismatch_summary.get("pct_above_0_70", np.nan)),
        },
        "foundational_context": {
            "pct_theoretical_without_explicit_job_demand": float(
                foundational_summary.get("pct_theoretical_skills_without_job_demand", np.nan)
            ),
            "mean_faculty_foundational_ratio": float(
                foundational_summary.get("mean_foundational_ratio_across_faculties", np.nan)
            ),
        },
        "policy_triage_counts": {
            "act_now": act_now_n,
            "investigate_first": inv_first_n,
            "other": int(len(action_matrix) - act_now_n - inv_first_n),
        },
    }


def main() -> None:
    args = parse_args()
    ttag = _t_tag(args.threshold)

    if args.output is None:
        args.output = RESULTS_DIR / f"moe_angles_{args.job_source}_t{ttag}"
    args.output.mkdir(parents=True, exist_ok=True)

    base_dir = RESULTS_DIR / f"baseline_scr_{args.job_source}"
    imp_dir = RESULTS_DIR / f"improved_scr_{args.job_source}_t{ttag}"
    rob_dir = RESULTS_DIR / f"robustness_{args.job_source}"
    pol_dir = RESULTS_DIR / f"policy_priority_{args.job_source}_t{ttag}"
    nul_dir = RESULTS_DIR / f"null_model_{args.job_source}"
    voc_dir = RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}"
    fnd_dir = RESULTS_DIR / f"foundational_layer_{args.job_source}"

    # Inputs
    base = _load_required(base_dir / "per_category_all.csv")
    imp = _load_required(imp_dir / "per_category_comparison.csv")
    depth = _load_required(rob_dir / "coverage_depth_by_category.csv")
    conf = _load_required(pol_dir / "confidence_flags_by_category.csv")
    sweep = _load_required(rob_dir / "threshold_sweep.csv")
    cat_priority = _load_required(pol_dir / "priority_gap_index_category_level.csv")
    skill_top = _load_required(pol_dir / "priority_gap_index_top10_per_category.csv")

    improved_summary = _load_required_json(imp_dir / "summary_comparison.json")
    null_summary = _load_required_json(nul_dir / "summary_null.json")
    mismatch_summary = _load_required_json(voc_dir / "mismatch_summary.json")
    foundational_summary = _load_required_json(fnd_dir / "summary_foundational_layer.json")

    # Angle 1: action matrix
    action_matrix = build_action_matrix(cat_priority)
    action_matrix.to_csv(args.output / "angle1_category_action_matrix.csv", index=False)

    # Angle 2: lift + depth + confidence
    lift_table = build_lift_table(base, imp, depth, conf)
    lift_table.to_csv(args.output / "angle2_coverage_lift_depth_confidence.csv", index=False)

    # Angle 3: policy threshold checkpoints
    threshold_view = build_threshold_policy_view(sweep)
    threshold_view.to_csv(args.output / "angle3_threshold_policy_view.csv", index=False)

    # Angle 4: top skill targets (carry forward ready table)
    skill_top.to_csv(args.output / "angle4_top_skill_targets_per_category.csv", index=False)

    # Exec summary
    exec_summary = build_exec_summary(
        improved_summary=improved_summary,
        null_summary=null_summary,
        mismatch_summary=mismatch_summary,
        foundational_summary=foundational_summary,
        action_matrix=action_matrix,
    )
    with (args.output / "moe_exec_summary.json").open("w", encoding="utf-8") as f:
        json.dump(exec_summary, f, indent=2)

    print("Saved: angle1_category_action_matrix.csv")
    print("Saved: angle2_coverage_lift_depth_confidence.csv")
    print("Saved: angle3_threshold_policy_view.csv")
    print("Saved: angle4_top_skill_targets_per_category.csv")
    print("Saved: moe_exec_summary.json")
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
