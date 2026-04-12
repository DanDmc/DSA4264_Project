"""
Pipeline 2 — VALIDATE the metric.

Runs the two scripts that stress-test Soft-IDF-SCR:
  1. null_model   — permutation test: is coverage non-random? (z-score, p-value)
  2. robustness   — threshold sweep, coverage depth, bootstrap 95% CIs

Must be run AFTER pipeline_build_metric.py.

Usage
-----
    python -m src.analysis.skill_metrics.pipeline_validate_metric
    python -m src.analysis.skill_metrics.pipeline_validate_metric --dry-run
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline 2: Validate Soft-IDF-SCR metric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python",               type=str,   default=sys.executable)
    p.add_argument("--job-source",           type=str,   default="skills_list",
                   choices=["skills_list", "skillner"])
    p.add_argument("--threshold",            type=float, default=0.72)
    p.add_argument("--bootstrap-iterations", type=int,   default=1000)
    p.add_argument("--dry-run",              action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cmds: list[list[str]] = [
        [args.python, "-m", "src.analysis.skill_metrics.null_model",
         "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.robustness",
         "--job-source", args.job_source,
         "--depth-threshold",     str(args.threshold),
         "--bootstrap-threshold", str(args.threshold),
         "--bootstrap-iterations", str(args.bootstrap_iterations)],
    ]

    print("=" * 60)
    print("PIPELINE 2 — VALIDATE METRIC")
    print(f"  Job source : {args.job_source}")
    print(f"  Threshold  : {args.threshold}")
    print("=" * 60)

    for cmd in cmds:
        _run(cmd, args.dry_run)

    manifest = {
        "pipeline": "validate_metric",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_source": args.job_source,
        "threshold": args.threshold,
        "bootstrap_iterations": args.bootstrap_iterations,
        "expected_outputs": {
            "null_model": str(RESULTS_DIR / f"null_model_{args.job_source}"),
            "robustness": str(RESULTS_DIR / f"robustness_{args.job_source}"),
        },
    }
    ttag = f"{args.threshold:.2f}".replace(".", "")
    out_path = RESULTS_DIR / f"manifest_validate_metric_{args.job_source}_t{ttag}.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\nPipeline 2 complete.")
    print(f"Manifest: {out_path}")


if __name__ == "__main__":
    main()
