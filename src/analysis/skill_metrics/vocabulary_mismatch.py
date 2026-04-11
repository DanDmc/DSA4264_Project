"""
Vocabulary Mismatch Analysis — Step 2b of the coverage pipeline.

What this script does
---------------------
Demonstrates and quantifies why exact string matching systematically
understates NUS's true skill coverage.

Motivation
----------
Step 1 found that NUS covers only 14–26 % of job-demanded skills under
exact matching.  Before accepting that as the true coverage gap, we need
to answer: are the "uncovered" job skills genuinely absent from NUS
curricula, or are they present but described using different vocabulary?

For example:
  Job posting  : "machine learning"
  NUS course   : "machine learning techniques"   ← same concept, different surface form
  Exact match  : FAIL

Method — embedding-based nearest-neighbour analysis
----------------------------------------------------
We embed every skill label (course and job) using a pre-trained
sentence-transformer model (paraphrase-MiniLM-L6-v2).  This model maps
short phrases into a dense vector space where semantically similar phrases
are geometrically close, grounded in the distributional hypothesis
(Harris, 1954) and learned via contrastive training on paraphrase pairs
(Reimers & Gurevych, 2019).

For each *uncovered* job skill we compute:
    max_sim(j) = max_{k ∈ course_skills} cosine_similarity(embed(j), embed(k))

This is the similarity to its *nearest neighbour* in the NUS course skill
set.  A high max_sim means the skill is conceptually present in NUS courses
even though it failed exact matching.

We then:
1. Plot the distribution of max_sim scores across all uncovered skills.
2. Identify near-misses: uncovered skills whose nearest course neighbour
   has similarity ≥ 0.75 (clearly the same concept, different wording).
3. Produce a threshold sensitivity curve: at each similarity threshold θ,
   what fraction of uncovered skills would be "recovered"?  This directly
   calibrates the improved metric in Step 3.

Theoretical foundation
----------------------
- Distributional semantics (Harris, 1954): meaning is determined by context.
- Sentence-BERT (Reimers & Gurevych, 2019): contrastive learning produces
  sentence embeddings where cosine similarity ≈ semantic similarity.
- Dense retrieval (Karpukhin et al., 2020): standard in IR and QA systems.

Model choice
------------
``paraphrase-MiniLM-L6-v2`` is chosen because:
  - It is optimised for *short phrase* similarity (our skill labels are
    2–5 words), unlike larger models optimised for full sentences.
  - Fast inference — embedding ~40 000 short phrases takes < 2 minutes
    on CPU.
  - Strong performance on semantic textual similarity benchmarks for
    paraphrase detection tasks.

Outputs
-------
results/vocabulary_mismatch/
  ├── near_miss_table.csv              — uncovered job skills + nearest course match
  ├── threshold_sensitivity.csv        — recovery rate at each similarity threshold
  ├── mismatch_summary.json           — headline statistics
  ├── fig_similarity_distribution.png — histogram of max similarity scores
  ├── fig_threshold_sensitivity.png   — recovery rate vs threshold curve
  └── fig_near_miss_examples.png      — annotated table of top near-misses

Usage
-----
    python -m src.analysis.skill_metrics.vocabulary_mismatch
    python -m src.analysis.skill_metrics.vocabulary_mismatch --threshold 0.80
    python -m src.analysis.skill_metrics.vocabulary_mismatch --top-examples 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Path resolution ───────────────────────────────────────────────────────────
try:
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import COURSES_PROCESSED_DIR, JOBS_PROCESSED_DIR, RESULTS_DIR

from src.analysis.skill_metrics.baseline_scr import load_course_skills, load_job_skills

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_JOB_SKILLS    = JOBS_PROCESSED_DIR    / "job_skill_pair_skillner.csv"
DEFAULT_COURSE_SKILLS = COURSES_PROCESSED_DIR / "module_skill_pairs.csv"
DEFAULT_OUTPUT_DIR    = RESULTS_DIR / "vocabulary_mismatch"
DEFAULT_MODEL         = "paraphrase-MiniLM-L6-v2"
DEFAULT_THRESHOLD     = 0.75     # near-miss threshold for the example table
DEFAULT_TOP_EXAMPLES  = 20       # rows in the near-miss table figure
DEFAULT_BATCH_SIZE    = 256


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_skills(
    skills: list[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode a list of skill labels into L2-normalised dense embeddings.

    L2 normalisation ensures that cosine similarity reduces to a dot
    product, which is faster to compute with numpy.

    Returns
    -------
    ndarray of shape (len(skills), embedding_dim), dtype float32.
    """
    embeddings = model.encode(
        skills,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise → cosine sim = dot product
    )
    return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-neighbour similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_nearest_neighbour(
    uncovered_skills:    list[str],
    course_skills_list:  list[str],
    uncovered_embs:      np.ndarray,
    course_embs:         np.ndarray,
    top_k:               int = 1,
) -> pd.DataFrame:
    """
    For each uncovered job skill, find its top-k nearest neighbours in the
    NUS course skill set by cosine similarity.

    Since embeddings are L2-normalised, cosine similarity = dot product,
    computed efficiently as a matrix multiplication.

    Parameters
    ----------
    uncovered_skills   : list of canonicalized uncovered job skill labels
    course_skills_list : list of canonicalized course skill labels
    uncovered_embs     : (n_uncovered, dim) float32 array
    course_embs        : (n_course, dim) float32 array
    top_k              : number of nearest neighbours to retrieve per skill

    Returns
    -------
    DataFrame with columns:
        job_skill, course_skill_match, similarity
    One row per uncovered skill (top-1 match only for top_k=1).
    """
    # Cosine similarity matrix: (n_uncovered, n_course)
    # Chunked to avoid OOM on large arrays
    chunk_size = 512
    n          = len(uncovered_skills)
    rows       = []

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = uncovered_embs[start:end]            # (chunk, dim)
        sims  = chunk @ course_embs.T                # (chunk, n_course)

        top_idx = np.argmax(sims, axis=1)            # best match per row
        top_sim = sims[np.arange(len(chunk)), top_idx]

        for i, (idx, sim) in enumerate(zip(top_idx, top_sim)):
            rows.append({
                "job_skill":          uncovered_skills[start + i],
                "course_skill_match": course_skills_list[idx],
                "similarity":         float(sim),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold_sensitivity(
    similarities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    For each similarity threshold θ, compute the fraction of uncovered
    skills that would be "recovered" (max_sim ≥ θ).

    This directly answers: "at what threshold does probabilistic matching
    in Step 3 stop recovering genuinely equivalent skills?"

    Returns DataFrame with columns: threshold, n_recovered, recovery_rate.
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 1.01, 0.05)

    n_total = len(similarities)
    rows = []
    for θ in thresholds:
        n_rec = int((similarities >= θ).sum())
        rows.append({
            "threshold":     round(float(θ), 2),
            "n_recovered":   n_rec,
            "recovery_rate": round(n_rec / n_total, 4) if n_total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_similarity_distribution(
    similarities: np.ndarray,
    threshold:    float,
    output_path:  Path,
) -> None:
    """
    Histogram of max cosine similarity scores for all uncovered job skills.

    The key story: skills to the right of the threshold line are
    "vocabulary mismatches" — conceptually present in NUS courses but
    missed by exact matching.  Skills to the left are genuine gaps.
    """
    n_total    = len(similarities)
    n_above    = int((similarities >= threshold).sum())
    pct_above  = n_above / n_total

    fig, ax = plt.subplots(figsize=(10, 5))

    counts, bins, patches = ax.hist(
        similarities, bins=60, edgecolor="white", linewidth=0.4,
    )

    # Colour bars by threshold
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor("#e74c3c" if left < threshold else "#27ae60")

    ax.axvline(threshold, color="#2c3e50", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel(
        "Max cosine similarity to nearest NUS course skill", fontsize=10
    )
    ax.set_ylabel("Number of uncovered job skills", fontsize=10)
    ax.set_title(
        "Vocabulary Mismatch Analysis: Semantic Similarity of Uncovered Job Skills\n"
        f"Red = genuine gap  |  Green = vocabulary mismatch (≥ {threshold:.2f})\n"
        f"{n_above:,} of {n_total:,} uncovered skills ({pct_above:.1%}) "
        f"have a near-equivalent in NUS courses",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation box
    ax.annotate(
        f"{pct_above:.1%} of 'uncovered' skills\n"
        f"ARE present in NUS courses\n"
        f"— just described differently",
        xy=(threshold + 0.01, counts.max() * 0.75),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", alpha=0.9),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_threshold_sensitivity(
    df_thresh:   pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Recovery rate vs similarity threshold curve.

    This is the calibration chart for Step 3: it shows what fraction of
    currently-uncovered job skills would be recovered at each threshold.
    A natural "elbow" in the curve suggests a principled threshold choice.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df_thresh["threshold"], df_thresh["recovery_rate"],
            color="#2980b9", linewidth=2.2, marker="o", markersize=5)

    # Mark common threshold choices
    for θ, label in [(0.70, "0.70"), (0.75, "0.75"), (0.80, "0.80"), (0.85, "0.85")]:
        row = df_thresh[df_thresh["threshold"] == θ]
        if not row.empty:
            rate = row["recovery_rate"].values[0]
            ax.axvline(θ, color="#bdc3c7", linewidth=1, linestyle=":")
            ax.text(θ + 0.004, rate + 0.012, f"{rate:.1%}",
                    fontsize=8, color="#2c3e50")

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Cosine similarity threshold (θ)", fontsize=10)
    ax.set_ylabel("Fraction of uncovered skills recovered", fontsize=10)
    ax.set_title(
        "Threshold Sensitivity: How Many 'Uncovered' Skills Are Recovered\n"
        "at Each Semantic Similarity Threshold?\n"
        "(Used to calibrate the improved SCR metric in Step 3)",
        fontsize=11, pad=10,
    )
    ax.set_xlim(0.48, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_near_miss_table(
    df_near_miss:  pd.DataFrame,
    top_n:         int,
    output_path:   Path,
) -> None:
    """
    Visual table of the top near-miss pairs sorted by job-skill demand frequency.

    Shows MOE officers concrete examples of vocabulary mismatch: the same
    concept expressed differently in academic vs industry language.
    """
    df_show = df_near_miss.head(top_n).copy()
    df_show["similarity"] = df_show["similarity"].map(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(13, max(4, top_n * 0.38 + 1.5)))
    ax.axis("off")

    col_labels = ["Job Skill\n(uncovered by exact match)",
                  "Nearest NUS Course Skill",
                  "Cosine\nSimilarity",
                  "Job Demand\n(freq)"]

    table_data = df_show[
        ["job_skill", "course_skill_match", "similarity", "job_freq"]
    ].values.tolist()

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours + highlight high-similarity rows
    for i in range(1, len(table_data) + 1):
        sim_val = float(df_show.iloc[i - 1]["similarity"])
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            if sim_val >= 0.85:
                cell.set_facecolor("#eafaf1")   # strong match — green tint
            elif sim_val >= 0.75:
                cell.set_facecolor("#fef9e7")   # moderate match — yellow tint
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("white")

    ax.set_title(
        f"Top {top_n} Near-Miss Pairs: Uncovered Job Skills with High-Similarity "
        "NUS Course Equivalents\n"
        "These skills are conceptually present in NUS courses "
        "but missed by exact string matching.",
        fontsize=10, pad=15, fontweight="bold",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify vocabulary mismatch between NUS course "
                    "and job posting skill descriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-skills",    type=Path,  default=DEFAULT_JOB_SKILLS)
    p.add_argument("--course-skills", type=Path,  default=DEFAULT_COURSE_SKILLS)
    p.add_argument("--output",        type=Path,  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model",         type=str,   default=DEFAULT_MODEL,
                   help="Sentence-transformer model name")
    p.add_argument("--threshold",     type=float, default=DEFAULT_THRESHOLD,
                   help="Cosine similarity threshold for near-miss classification")
    p.add_argument("--top-examples",  type=int,   default=DEFAULT_TOP_EXAMPLES,
                   help="Rows in the near-miss table figure")
    p.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = args.output
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    courses = load_course_skills(args.course_skills)
    jobs    = load_job_skills(args.job_skills)

    course_skill_set  = set(courses["skill_canon"].unique())
    all_job_skills    = set(jobs["skill_canon"].unique())
    uncovered_skills  = sorted(all_job_skills - course_skill_set)
    course_skills_lst = sorted(course_skill_set)

    # Demand frequency for each uncovered skill (total mentions across all jobs)
    skill_freq = jobs["skill_canon"].value_counts()

    print(f"  Course skills  (unique): {len(course_skill_set):>7,}")
    print(f"  Job skills     (unique): {len(all_job_skills):>7,}")
    print(f"  Uncovered skills       : {len(uncovered_skills):>7,}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading sentence-transformer model: {args.model} …")
    model = SentenceTransformer(args.model)

    # ── Embed skills ───────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(course_skills_lst):,} course skills …")
    course_embs = embed_skills(course_skills_lst, model, args.batch_size)

    print(f"Embedding {len(uncovered_skills):,} uncovered job skills …")
    uncov_embs = embed_skills(uncovered_skills, model, args.batch_size)

    # ── Nearest-neighbour search ───────────────────────────────────────────────
    print("\nComputing nearest-neighbour similarities …")
    df_nn = compute_nearest_neighbour(
        uncovered_skills, course_skills_lst,
        uncov_embs, course_embs,
    )

    # Attach demand frequency
    df_nn["job_freq"] = df_nn["job_skill"].map(
        lambda s: int(skill_freq.get(s, 0))
    )
    df_nn = df_nn.sort_values("similarity", ascending=False)

    # ── Threshold sensitivity ──────────────────────────────────────────────────
    similarities  = df_nn["similarity"].values
    df_thresh     = compute_threshold_sensitivity(similarities)

    # ── Near-miss table: high-similarity + high-demand ────────────────────────
    # Sort by similarity first to get the clearest near-miss examples,
    # then also provide a freq-sorted version in the CSV
    df_near_miss = df_nn[df_nn["similarity"] >= args.threshold].copy()
    df_near_miss_by_freq = df_near_miss.sort_values("job_freq", ascending=False)

    # ── Save outputs ───────────────────────────────────────────────────────────
    print("\nSaving outputs …")

    df_nn.to_csv(out / "all_uncovered_nearest_neighbour.csv", index=False)
    print("  Saved: all_uncovered_nearest_neighbour.csv")

    df_near_miss_by_freq.to_csv(out / "near_miss_table.csv", index=False)
    print("  Saved: near_miss_table.csv")

    df_thresh.to_csv(out / "threshold_sensitivity.csv", index=False)
    print("  Saved: threshold_sensitivity.csv")

    n_near_miss = len(df_near_miss)
    pct_near_miss = n_near_miss / len(uncovered_skills)

    summary = {
        "model":                  args.model,
        "threshold":              args.threshold,
        "n_course_skills":        len(course_skill_set),
        "n_job_skills":           len(all_job_skills),
        "n_uncovered_skills":     len(uncovered_skills),
        "n_near_misses":          n_near_miss,
        "pct_near_misses":        round(pct_near_miss, 4),
        "median_max_similarity":  round(float(np.median(similarities)), 4),
        "mean_max_similarity":    round(float(np.mean(similarities)),   4),
        "pct_above_0_70":  round(float((similarities >= 0.70).mean()), 4),
        "pct_above_0_75":  round(float((similarities >= 0.75).mean()), 4),
        "pct_above_0_80":  round(float((similarities >= 0.80).mean()), 4),
        "pct_above_0_85":  round(float((similarities >= 0.85).mean()), 4),
    }
    with (out / "mismatch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  Saved: mismatch_summary.json")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_similarity_distribution(
        similarities, args.threshold,
        out / "fig_similarity_distribution.png",
    )
    plot_threshold_sensitivity(df_thresh, out / "fig_threshold_sensitivity.png")
    plot_near_miss_table(
        df_near_miss_by_freq, args.top_examples,
        out / "fig_near_miss_examples.png",
    )

    # ── Headline ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  VOCABULARY MISMATCH — HEADLINE RESULTS")
    print("=" * 62)
    print(f"  Uncovered skills (exact match)  : {len(uncovered_skills):,}")
    print(f"  Near-misses (sim ≥ {args.threshold:.2f})         : "
          f"{n_near_miss:,}  ({pct_near_miss:.1%})")
    print()
    print(f"  Recovery rate at different thresholds:")
    for θ in [0.70, 0.75, 0.80, 0.85]:
        row = df_thresh[df_thresh["threshold"] == θ]
        if not row.empty:
            print(f"    θ = {θ:.2f}  →  "
                  f"{row['recovery_rate'].values[0]:.1%} of uncovered skills recovered")
    print()
    print(f"  Median max similarity : {np.median(similarities):.3f}")
    print(f"  Mean max similarity   : {np.mean(similarities):.3f}")
    print("=" * 62)
    print(f"\n  Outputs written to: {out}\n")


if __name__ == "__main__":
    main()
