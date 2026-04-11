"""Run the end-to-end skill_metrics methodology for jobs `skills_list` with one command.

This orchestrator reduces operational overhead and standardizes outputs for reporting.
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


def _threshold_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command pipeline for skill_metrics (skills_list-first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skills_list"], help="Locked to skills_list for current reporting scope")
    p.add_argument("--primary-threshold", type=float, default=0.85)
    p.add_argument("--exploratory-threshold", type=float, default=0.70)
    p.add_argument("--include-exploratory", action="store_true", help="Also run improved_scr at exploratory threshold for sensitivity appendix")
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    t_primary_tag = _threshold_tag(args.primary_threshold)
    t_expl_tag = _threshold_tag(args.exploratory_threshold)

    improved_primary_out = RESULTS_DIR / f"improved_scr_{args.job_source}_t{t_primary_tag}"
    improved_expl_out = RESULTS_DIR / f"improved_scr_{args.job_source}_t{t_expl_tag}"

    cmds: list[list[str]] = [
        [args.python, "-m", "src.analysis.skill_metrics.baseline_scr", "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.null_model", "--job-source", args.job_source],
        [args.python, "-m", "src.analysis.skill_metrics.vocabulary_mismatch", "--job-source", args.job_source],
        [
            args.python,
            "-m",
            "src.analysis.skill_metrics.improved_scr",
            "--job-source",
            args.job_source,
            "--threshold",
            str(args.primary_threshold),
            "--output",
            str(improved_primary_out),
        ],
        [
            args.python,
            "-m",
            "src.analysis.skill_metrics.robustness",
            "--job-source",
            args.job_source,
            "--depth-threshold",
            str(args.primary_threshold),
            "--bootstrap-threshold",
            str(args.primary_threshold),
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ],
        [args.python, "-m", "src.analysis.skill_metrics.foundational_layer", "--job-source", args.job_source],
        [
            args.python,
            "-m",
            "src.analysis.skill_metrics.policy_priority",
            "--job-source",
            args.job_source,
            "--threshold",
            str(args.primary_threshold),
            "--improved-per-category",
            str(improved_primary_out / "per_category_comparison.csv"),
        ],
    ]

    if args.include_exploratory:
        cmds.insert(
            4,
            [
                args.python,
                "-m",
                "src.analysis.skill_metrics.improved_scr",
                "--job-source",
                args.job_source,
                "--threshold",
                str(args.exploratory_threshold),
                "--output",
                str(improved_expl_out),
            ],
        )

    for cmd in cmds:
        _run(cmd, args.dry_run)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_source": args.job_source,
        "primary_threshold": args.primary_threshold,
        "exploratory_threshold": args.exploratory_threshold,
        "include_exploratory": bool(args.include_exploratory),
        "commands": cmds,
        "expected_outputs": {
            "baseline": str(RESULTS_DIR / f"baseline_scr_{args.job_source}"),
            "null_model": str(RESULTS_DIR / f"null_model_{args.job_source}"),
            "vocabulary_mismatch": str(RESULTS_DIR / f"vocabulary_mismatch_{args.job_source}"),
            "improved_primary": str(improved_primary_out),
            "robustness": str(RESULTS_DIR / f"robustness_{args.job_source}"),
            "foundational_layer": str(RESULTS_DIR / f"foundational_layer_{args.job_source}"),
            "policy_priority": str(RESULTS_DIR / f"policy_priority_{args.job_source}_t{t_primary_tag}"),
        },
    }

    local_manifest = Path("results") / f"pipeline_manifest_{args.job_source}_t{t_primary_tag}.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    with local_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPipeline complete.")
    print(f"Saved local manifest: {local_manifest}")


if __name__ == "__main__":
    main()
