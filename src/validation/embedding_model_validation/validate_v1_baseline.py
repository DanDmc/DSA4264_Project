"""
V1 — Baseline: Full-Text Embedding
-----------------------------------
Each job/course gets a single embedding from its full text.
Similarity = cosine between those two vectors.

This is the simplest approach and our reference point.
If the fancier methods can't beat this, they're not worth the complexity.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from validation_utils import (
    load_data, cosine, save_results, build_variant_summary,
)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DATA_FILE = Path(__file__).parent / "validation_dataset.xlsx"
OUTPUT_FILE = Path(__file__).parent / "results_v1_baseline.json"


def run():
    print("V1 — Baseline full-text embedding")
    print("=" * 50)

    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    jobs, courses = load_data(DATA_FILE)

    # precompute one embedding per document
    print("Embedding jobs...")
    job_vecs = {jid: model.encode(j["text"]) for jid, j in jobs.items()}

    print("Embedding courses...")
    course_vecs = {cid: model.encode(c["text"]) for cid, c in courses.items()}

    # score every (job, course) pair
    print("Computing similarities...")
    results = []
    for jid, j in tqdm(jobs.items()):
        for cid, c in courses.items():
            results.append({
                "job": jid,
                "course": cid,
                "job_category": j["category"],
                "course_category": c["category"],
                "score": cosine(job_vecs[jid], course_vecs[cid]),
            })

    config = {
        "variant": "v1_baseline",
        "embedding_model": EMBEDDING_MODEL,
        "method": "full_text_single_embedding",
        "timestamp": datetime.now().isoformat(),
    }

    output = build_variant_summary(results, "score", "v1_baseline", config)
    save_results(output, OUTPUT_FILE)

    mrr = output["aggregate"]["retrieval"]["MRR"]
    gap = output["aggregate"]["separation"]["separation_gap"]
    print(f"\n  MRR: {mrr:.4f}  |  Separation gap: {gap:.4f}")


if __name__ == "__main__":
    run()
