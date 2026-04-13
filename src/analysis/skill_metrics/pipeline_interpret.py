"""
Pipeline 3 — INTERPRET the metric.

Runs the scripts that generate policy-facing outputs:
  1. foundational_layer        — theoretical course skills with no job-market demand
  2. policy_priority           — Priority Gap Index + confidence flags per category
  3. breakdown_analysis        — coverage by job category / SSOC / faculty / major / department
  4. moe_result_angles         — action matrix, lift table, exec summary for MOE briefing
  5. combined_degree_priority  — degree-level alignment score combining skill-space +
                                 semantic analysis; ranked list of degrees for MOE review

Must be run AFTER pipeline_build_metric.py AND pipeline_validate_metric.py.
Step 5 also requires similarity_analysis.py to have been run (targeted_ssoc_alignment.csv).

Usage
-----
    python -m src.analysis.skill_metrics.pipeline_interpret
    python -m src.analysis.skill_metrics.pipeline_interpret --dry-run
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
        description="Pipeline 3: Interpret Soft-IDF-SCR outputs for MOE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python",     type=str,   default=sys.executable)
    p.add_argument("--job-source", type=str,   default="skills_list",
                   choices=["skills_list", "skillner"])
    p.add_argument("--threshold",  type=float, default=0.72,
                   help="Must match the threshold used in pipeline_build_metric.")
    p.add_argument("--dry-run",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ttag = _t_tag(args.threshold)

    improved_dir = RESULTS_DIR / f"improved_scr_{args.job_source}_t{ttag}"

    cmds: list[list[str]] = [
        [args.python, "-m", "src.analysis.skill_metrics.foundational_layer",
         "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.policy_priority",
         "--job-source", args.job_source,
         "--threshold",  str(args.threshold),
         "--improved-per-category", str(improved_dir / "per_category_comparison.csv")],
        [args.python, "-m", "src.analysis.skill_metrics.breakdown_analysis",
         "--job-source", args.job_source,
         "--threshold",  str(args.threshold)],
        [args.python, "-m", "src.analysis.skill_metrics.moe_result_angles",
         "--job-source", args.job_source,
         "--threshold",  str(args.threshold)],
        [args.python, "-m", "src.analysis.combined_degree_priority"],
    ]

    print("=" * 60)
    print("PIPELINE 3 — INTERPRET")
    print(f"  Job source : {args.job_source}")
    print(f"  Threshold  : {args.threshold}")
    print("=" * 60)

    # Step 5 depends on similarity_analysis.py having been run — skip gracefully if not
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sem_csv = REPO_ROOT / "outputs" / "similarity_analysis_outputs" / "targeted_ssoc_alignment.csv"
    if not sem_csv.exists():
        print(f"\n[skip] Step 5 (combined_degree_priority): semantic CSV not found at:")
        print(f"       {sem_csv}")
        print("       Run similarity_analysis.py first, then re-run pipeline_interpret.")
        cmds = cmds[:-1]

    for cmd in cmds:
        _run(cmd, args.dry_run)

    manifest = {
        "pipeline": "interpret",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_source": args.job_source,
        "threshold": args.threshold,
        "expected_outputs": {
            "foundational_layer":       str(RESULTS_DIR / f"foundational_layer_{args.job_source}"),
            "policy_priority":          str(RESULTS_DIR / f"policy_priority_{args.job_source}_t{ttag}"),
            "breakdown_analysis":       str(RESULTS_DIR / f"breakdown_analysis_{args.job_source}"),
            "moe_angles":               str(RESULTS_DIR / f"moe_angles_{args.job_source}_t{ttag}"),
            **({
                "combined_degree_priority": str(RESULTS_DIR / "combined_degree_priority"),
            } if sem_csv.exists() else {
                "combined_degree_priority": "skipped — run similarity_analysis.py first",
            }),
        },
    }
    out_path = RESULTS_DIR / f"manifest_interpret_{args.job_source}_t{ttag}.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\nPipeline 3 complete.")
    print(f"Manifest: {out_path}")


if __name__ == "__main__":
    main()
