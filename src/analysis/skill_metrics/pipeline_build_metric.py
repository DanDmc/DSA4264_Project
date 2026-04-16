"""
Pipeline 1 — BUILD the metric.

Runs the three scripts that construct Soft-IDF-SCR:
  1. baseline_scr      — exact match coverage, establishes course/job skill sets
  2. vocabulary_mismatch — embedding nearest-neighbour for unmatched job skills
  3. improved_scr      — Soft-IDF-SCR combining soft matching + IDF weighting

Output: per-category Soft-IDF-SCR at the calibrated threshold (θ=0.72).

Usage
-----
    python -m src.analysis.skill_metrics.pipeline_build_metric
    python -m src.analysis.skill_metrics.pipeline_build_metric --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from src.config import RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import RESULTS_DIR


def _run(cmd: list[str], dry_run: bool) -> None:
    print("\n$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _t_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline 1: Build Soft-IDF-SCR metric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python",     type=str,   default=sys.executable)
    p.add_argument("--job-source", type=str,   default="skills_list",
                   choices=["skills_list", "skillner"])
    p.add_argument("--threshold",  type=float, default=0.72,
                   help="Calibrated similarity threshold (θ=0.72 by default, "
                        "empirically selected as equal-error-rate point)")
    p.add_argument("--dry-run",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ttag = _t_tag(args.threshold)
    improved_out = RESULTS_DIR / f"improved_scr_{args.job_source}_t{ttag}"

    cmds: list[list[str]] = [
        [args.python, "-m", "src.analysis.skill_metrics.baseline_scr",
         "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.vocabulary_mismatch",
         "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.improved_scr",
         "--job-source", args.job_source,
         "--threshold", str(args.threshold),
         "--output", str(improved_out)],
    ]

    print("=" * 60)
    print("PIPELINE 1 — BUILD METRIC")
    print(f"  Job source : {args.job_source}")
    print(f"  Threshold  : {args.threshold} (calibrated equal-error-rate point)")
    print("=" * 60)

    for cmd in cmds:
        _run(cmd, args.dry_run)

    manifest = {
        "pipeline": "build_metric",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_source": args.job_source,
        "threshold": args.threshold,
        "threshold_justification": (
            f"θ={args.threshold} selected as equal-error-rate point "
            "(precision = recall) from calibration against 357 human-labelled "
            "course-skill / job-skill pairs. "
            f"F1=0.685, Precision=0.660, Recall=0.711 at θ=0.72."
        ),
        "expected_outputs": {
            "baseline":           str(RESULTS_DIR / f"baseline_scr_{args.job_source}"),
            "vocabulary_mismatch":str(RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}"),
            "improved_scr":       str(improved_out),
        },
    }
    out_path = RESULTS_DIR / f"manifest_build_metric_{args.job_source}_t{ttag}.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\nPipeline 1 complete.")
    print(f"Manifest: {out_path}")


if __name__ == "__main__":
    main()
