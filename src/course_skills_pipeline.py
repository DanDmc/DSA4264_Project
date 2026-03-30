"""
course_skills_pipeline.py
=========================
End-to-end pipeline for NUS course skill extraction and validation.

All file paths live in one place (CONFIG below).
Each step can be run independently or as part of the full sequence.

  Step 1  fetch        Fetch NUS modules from NUSMods API
  Step 2  extract      Run SkillNer + taxonomy skill extraction
  Step 3  postprocess  Canonicalize and denoise extracted skills
  Step 4  validate     Compare extractor methods against manual annotations
                       (requires --ground-truth file)

Usage
-----
Run full pipeline (steps 1 → 2 → 3):
    python -m src.course_skills_pipeline

Run specific steps only:
    python -m src.course_skills_pipeline --steps 2 3

Re-run extraction after re-fetching modules:
    python -m src.course_skills_pipeline --steps 1 2 3

Run validation once annotation Excel is ready:
    python -m src.course_skills_pipeline --steps 4 \\
        --ground-truth data/validation/annotations.xlsx \\
        --validation-methods 1 2 3 4

Data flow
---------
  NUSMods API
      │  step 1
      ▼
  data/processed/modules_raw.csv          (all modules, no filter)
  data/processed/cleaned_modules.csv      (undergraduate, valid descriptions)
      │  step 2
      ▼
  data/processed/modules_with_skills.csv  (one row per module, skills as list)
  data/processed/module_skill_pairs.csv   (one row per module-skill pair)
      │  step 3
      ▼
  data/processed/modules_with_skills_clean.csv  (canonicalized + denoised)
      │  step 4 (optional, needs ground truth)
      ▼
  results/extractor_comparison.csv
  results/extractor_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# CONFIG — change paths here if needed, nowhere else
# ---------------------------------------------------------------------------

class Config:
    # NUSMods API
    NUSMODS_API_URL = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"

    # Step 1 outputs
    MODULES_RAW     = Path("data/processed/modules_raw.csv")
    MODULES_CLEANED = Path("data/processed/cleaned_modules.csv")

    # Step 2 outputs
    MODULES_WITH_SKILLS = Path("data/processed/modules_with_skills.csv")
    MODULE_SKILL_PAIRS  = Path("data/processed/module_skill_pairs.csv")

    # Step 3 output
    MODULES_WITH_SKILLS_CLEAN = Path("data/processed/modules_with_skills_clean.csv")

    # Step 4 outputs
    VALIDATION_OUTPUT  = Path("results/extractor_comparison.csv")
    VALIDATION_SUMMARY = Path("results/extractor_comparison_summary.csv")

    # Optional: path to a custom taxonomy JSON/CSV for step 2
    TAXONOMY_PATH: str | None = None


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _header(step: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {step}: {title}")
    print(f"{'─' * 60}")


def _done(elapsed: float) -> None:
    print(f"  Done  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Step 1 — Fetch NUSMods
# ---------------------------------------------------------------------------

def step_fetch(cfg: Config) -> None:
    _header(1, "Fetch NUS modules from NUSMods API")

    from src.data_processing.process_nusmods import run as fetch_run
    t = time.time()
    fetch_run(
        output_raw=str(cfg.MODULES_RAW),
        output_cleaned=str(cfg.MODULES_CLEANED),
    )
    _done(time.time() - t)


# ---------------------------------------------------------------------------
# Step 2 — Extract skills
# ---------------------------------------------------------------------------

def step_extract(cfg: Config) -> None:
    _header(2, "Extract skills with SkillNer + taxonomy")

    if not cfg.MODULES_CLEANED.exists():
        print(f"  [ERROR] Input not found: {cfg.MODULES_CLEANED}")
        print("  Run step 1 first (--steps 1) to fetch and clean modules.")
        sys.exit(1)

    from src.data_processing.extract_course_skills import (
        extract_course_skills,
        PipelineConfig,
    )
    t = time.time()
    cfg.MODULES_WITH_SKILLS.parent.mkdir(parents=True, exist_ok=True)
    extract_course_skills(
        input_csv=cfg.MODULES_CLEANED,
        output_rows_csv=cfg.MODULES_WITH_SKILLS,
        output_skills_csv=cfg.MODULE_SKILL_PAIRS,
        taxonomy_path=cfg.TAXONOMY_PATH,
        config=PipelineConfig(),
    )
    _done(time.time() - t)


# ---------------------------------------------------------------------------
# Step 3 — Post-process (canonicalize + denoise)
# ---------------------------------------------------------------------------

def step_postprocess(cfg: Config) -> None:
    _header(3, "Canonicalize and denoise extracted skills")

    if not cfg.MODULES_WITH_SKILLS.exists():
        print(f"  [ERROR] Input not found: {cfg.MODULES_WITH_SKILLS}")
        print("  Run step 2 first (--steps 2).")
        sys.exit(1)

    from src.data_processing.postprocess_skills import run as postprocess_run
    t = time.time()
    cfg.MODULES_WITH_SKILLS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    postprocess_run(cfg.MODULES_WITH_SKILLS, cfg.MODULES_WITH_SKILLS_CLEAN)
    _done(time.time() - t)


# ---------------------------------------------------------------------------
# Step 4 — Validate extractor methods
# ---------------------------------------------------------------------------

def step_validate(
    cfg: Config,
    ground_truth_path: str | None,
    methods: list[int],
    llm_provider: str | None,
) -> None:
    _header(4, "Compare skill extractor methods")

    if ground_truth_path is None:
        print("  [WARN] No --ground-truth file provided.")
        print("  Running with dummy ground truth (3 courses) as a smoke test.")
        print("  Pass --ground-truth <path> when your annotation Excel is ready.")

    if not cfg.MODULES_CLEANED.exists():
        print(f"  [ERROR] Input not found: {cfg.MODULES_CLEANED}")
        print("  Run step 1 first to generate cleaned_modules.csv.")
        sys.exit(1)

    from src.validation.compare_extractors import (
        load_ground_truth,
        run_method,
        evaluate,
        aggregate_metrics,
        print_summary,
        _detect_provider,
    )
    import pandas as pd

    ground_truth = load_ground_truth(ground_truth_path)
    df_courses   = pd.read_csv(cfg.MODULES_CLEANED)

    course_codes = list(ground_truth.keys())
    df_eval = df_courses[df_courses["module code"].isin(course_codes)].copy()

    if df_eval.empty:
        print("  [ERROR] None of the annotated course codes found in cleaned_modules.csv.")
        print(f"  Codes in ground truth: {course_codes[:10]}...")
        sys.exit(1)

    print(f"  Evaluating on {len(df_eval)} courses | Methods: {methods}")

    provider = llm_provider or _detect_provider()
    if any(m in (3, 4) for m in methods) and not provider:
        print("  [WARN] Methods 3/4 need an LLM key. Set GROQ_API_KEY (free at console.groq.com).")
        methods = [m for m in methods if m in (1, 2)]
        print(f"  Falling back to methods: {methods}")

    t = time.time()
    all_results = []
    for method_id in methods:
        method_names = {1: "skillner_only", 2: "skillner_taxonomy",
                        3: "llm_fewshot", 4: "hybrid"}
        print(f"  Running method {method_id}: {method_names[method_id]}...")
        for _, row in df_eval.iterrows():
            all_results.append(run_method(method_id, row, provider))

    per_course = evaluate(all_results, ground_truth)
    summary    = aggregate_metrics(per_course)

    print_summary(summary)
    print("\n  Per-course breakdown:")
    print(per_course[["module_code", "method", "n_predicted", "n_ground_truth",
                       "precision", "recall", "f1", "tp", "fp", "fn"]].to_string(index=False))

    cfg.VALIDATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    per_course.to_csv(cfg.VALIDATION_OUTPUT, index=False)
    summary.to_csv(cfg.VALIDATION_SUMMARY, index=False)
    print(f"\n  Saved → {cfg.VALIDATION_OUTPUT}")
    print(f"  Saved → {cfg.VALIDATION_SUMMARY}")
    _done(time.time() - t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NUS Course Skills Pipeline — end-to-end extraction and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.course_skills_pipeline                   # steps 1 2 3
  python -m src.course_skills_pipeline --steps 2 3       # skip fetch
  python -m src.course_skills_pipeline --steps 4 \\
      --ground-truth data/validation/annotations.xlsx \\
      --validation-methods 1 2 3 4
        """,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3],
        help="Steps to run (default: 1 2 3). Step 4 requires --ground-truth.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to annotation Excel for step 4 validation.",
    )
    parser.add_argument(
        "--validation-methods",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2],
        help="Extractor methods to compare in step 4 (default: 1 2).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "anthropic", "openai"],
        default=None,
        help="LLM provider for methods 3/4 (auto-detected from env vars if not set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config()

    steps = sorted(set(args.steps))
    print(f"\nCourse Skills Pipeline — running steps: {steps}")

    total_start = time.time()

    for step in steps:
        if step == 1:
            step_fetch(cfg)
        elif step == 2:
            step_extract(cfg)
        elif step == 3:
            step_postprocess(cfg)
        elif step == 4:
            step_validate(cfg, args.ground_truth, args.validation_methods, args.llm_provider)

    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete  (total: {time.time() - total_start:.1f}s)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
