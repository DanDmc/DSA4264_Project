"""Dedicated pipeline for MOE-facing result angles.

Separate from methodology pipeline. It assumes core result folders already
exist and only generates interpretation-ready MOE angle outputs.
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
        description="Build MOE-facing angle outputs from existing skill_metrics results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--job-source", type=str, default="skills_list", choices=["skills_list"])
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ttag = _t_tag(args.threshold)

    cmd = [
        args.python,
        "-m",
        "src.analysis.skill_metrics.moe_result_angles",
        "--job-source",
        args.job_source,
        "--threshold",
        str(args.threshold),
    ]
    _run(cmd, args.dry_run)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_source": args.job_source,
        "threshold": args.threshold,
        "commands": [cmd],
        "output_dir": str(RESULTS_DIR / f"moe_angles_{args.job_source}_t{ttag}"),
    }
    out = Path("results") / f"pipeline_manifest_moe_angles_{args.job_source}_t{ttag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nMOE angle pipeline complete.")
    print(f"Saved local manifest: {out}")


if __name__ == "__main__":
    main()
