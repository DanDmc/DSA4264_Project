"""
Threshold Calibration — Step 1: Generate labelling sheet.

What this script does
---------------------
Takes 306 manually annotated course skills and finds their nearest
job skill neighbours using sentence embeddings. Outputs an Excel
labelling sheet containing only the ambiguous pairs (similarity in
the 0.65–0.95 range) for human labelling.

Why only the ambiguous range?
    - Pairs with similarity < 0.65 are clearly different skills.
      No human needs to label "recursion" vs "customer service."
    - Pairs with similarity > 0.95 are clearly the same skill.
      No human needs to label "python" vs "python programming."
    - Only pairs in [0.65, 0.95] are where the threshold choice
      actually matters — these are the pairs to label.

Output
------
data/validation_skillmetrics/threshold_calibration_pairs.xlsx
    One row per (course_skill, job_skill) pair with columns:
        pair_id         — unique ID for each pair
        course_skill    — canonicalized course skill label
        job_skill       — canonicalized job skill label
        similarity      — cosine similarity from embeddings
        label           — BLANK — human fills in 1 (same) or 0 (different)
        notes           — BLANK — optional human notes

Usage
-----
    python -m src.validation.threshold_calibration.generate_pairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR

from src.analysis.skill_metrics.baseline_scr import canonicalize, load_job_skills_from_list

ANNOTATIONS_PATH = Path("/Users/mv/OneDrive/DSA4264_Project_Data/validation_skillmetrics/annotations.xlsx")
OUTPUT_PATH      = Path("/Users/mv/OneDrive/DSA4264_Project_Data/validation_skillmetrics/threshold_calibration_pairs.xlsx")
DEFAULT_JOB_PATH = JOBS_PROCESSED_DIR / "03_jobs_filtered.csv"

SIM_LOW  = 0.65   # below this — clearly different, skip
SIM_HIGH = 0.95   # above this — clearly same, skip
TOP_K    = 3      # top-k nearest job skills per course skill


def load_annotated_course_skills(path: Path) -> list[str]:
    df = pd.read_excel(path, sheet_name="courses_annotated")
    skills = df["skill_label"].dropna().map(canonicalize).unique().tolist()
    return [s for s in skills if s]


def embed_skills(skills: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(skills, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=128)
    return embeddings


def find_ambiguous_pairs(
    course_skills: list[str],
    job_skills:    list[str],
    course_embs:   np.ndarray,
    job_embs:      np.ndarray,
    top_k:         int,
    sim_low:       float,
    sim_high:      float,
) -> pd.DataFrame:
    """
    For each course skill, find top-k nearest job skills.
    Keep only pairs in the ambiguous similarity range [sim_low, sim_high].
    """
    # Full similarity matrix: (n_course, n_job)
    sim_matrix = course_embs @ job_embs.T   # L2-normalised → cosine sim

    rows = []
    for i, cs in enumerate(course_skills):
        sims = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:top_k]

        for j in top_idx:
            sim = float(sims[j])
            if sim_low <= sim <= sim_high:
                rows.append({
                    "course_skill": cs,
                    "job_skill":    job_skills[j],
                    "similarity":   round(sim, 4),
                    "label":        "",   # human fills: 1=same, 0=different
                    "notes":        "",
                })

    df = pd.DataFrame(rows)
    # Deduplicate same (course_skill, job_skill) pairs
    df = df.drop_duplicates(subset=["course_skill", "job_skill"])
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
    df.insert(0, "pair_id", [f"P{i+1:04d}" for i in range(len(df))])
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate threshold calibration labelling sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    p.add_argument("--job-skills",  type=Path, default=DEFAULT_JOB_PATH)
    p.add_argument("--output",      type=Path, default=OUTPUT_PATH)
    p.add_argument("--model",       type=str,
                   default="paraphrase-MiniLM-L6-v2",
                   help="Sentence-transformers model — must match vocabulary_mismatch.py")
    p.add_argument("--top-k",       type=int,  default=TOP_K)
    p.add_argument("--sim-low",     type=float, default=SIM_LOW)
    p.add_argument("--sim-high",    type=float, default=SIM_HIGH)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading annotated course skills …")
    course_skills = load_annotated_course_skills(args.annotations)
    print(f"  {len(course_skills)} unique canonicalized course skills")

    print("Loading job skills …")
    jobs = load_job_skills_from_list(args.job_path if hasattr(args, 'job_path') else args.job_skills)
    job_skills = jobs["skill_canon"].unique().tolist()
    print(f"  {len(job_skills)} unique job skills")

    print(f"\nEmbedding with {args.model} …")
    course_embs = embed_skills(course_skills, args.model)
    job_embs    = embed_skills(job_skills,    args.model)

    print(f"\nFinding ambiguous pairs (similarity {args.sim_low}–{args.sim_high}) …")
    pairs = find_ambiguous_pairs(
        course_skills, job_skills,
        course_embs,   job_embs,
        top_k=args.top_k,
        sim_low=args.sim_low,
        sim_high=args.sim_high,
    )
    print(f"  {len(pairs)} ambiguous pairs found")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        pairs.to_excel(writer, sheet_name="pairs_to_label", index=False)

        # Instructions sheet
        instructions = pd.DataFrame({
            "Instructions": [
                "For each row, fill in the 'label' column:",
                "  1 = these two skills represent the SAME concept",
                "  0 = these are DIFFERENT concepts",
                "",
                "Guidelines:",
                "  - Focus on meaning, not exact wording",
                "  - 'python' and 'python programming' → 1 (same)",
                "  - 'machine learning' and 'statistical modelling' → judgement call",
                "  - 'recursion' and 'customer service' → 0 (different)",
                "",
                "Add any notes in the 'notes' column for edge cases.",
                f"Total pairs to label: {len(pairs)}",
                f"Similarity range shown: {args.sim_low} – {args.sim_high}",
                "Pairs outside this range are excluded (clearly same or different).",
            ]
        })
        instructions.to_excel(writer, sheet_name="instructions", index=False)

    print(f"\nSaved labelling sheet: {args.output}")
    print(f"Next step: fill in the 'label' column (1=same, 0=different)")
    print(f"Then run: python -m src.validation.threshold_calibration.calibrate_threshold")


if __name__ == "__main__":
    main()
