"""
V3 — Sentence Average: Split & Mean-Pool
------------------------------------------
Split text into sentences, embed each independently, then
mean-pool all sentence embeddings into one document vector.

No filtering, no weighting — just a different granularity
than full-text embedding. This isolates whether sentence-level
embedding itself helps, separate from any filtering/weighting
strategy.

Why this matters as a baseline:
Full-text models compress an entire document into one embedding
pass. For long job postings, that forces the model to "average"
internally. Explicit sentence-level embedding + mean-pooling
gives each sentence equal representation, which might help or
hurt depending on how much noise is in the text.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from validation_utils import (
    load_data, split_sentences, cosine, save_results, build_variant_summary,
)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DATA_FILE = Path(__file__).parent / "validation_dataset.xlsx"
OUTPUT_FILE = Path(__file__).parent / "results_v3_sentence_avg.json"


def sentence_average_embedding(text, model):
    """
    Plain sentence-level mean pooling:
    1. Split text into sentences
    2. Embed each sentence
    3. Mean-pool all embeddings

    No filtering or weighting — every sentence counts equally.
    """
    sentences = split_sentences(text)
    if not sentences:
        return np.zeros(model.get_sentence_embedding_dimension())

    sent_embs = model.encode(sentences)
    return np.mean(sent_embs, axis=0)


def run():
    print("V3 — Sentence average (mean-pool)")
    print("=" * 50)

    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    jobs, courses = load_data(DATA_FILE)

    # precompute embeddings
    print("Embedding jobs (sentence-level)...")
    job_vecs = {jid: sentence_average_embedding(j["text"], model) for jid, j in jobs.items()}

    print("Embedding courses (sentence-level)...")
    course_vecs = {cid: sentence_average_embedding(c["text"], model) for cid, c in courses.items()}

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
        "variant": "v3_sentence_avg",
        "embedding_model": EMBEDDING_MODEL,
        "method": "sentence_mean_pool",
        "timestamp": datetime.now().isoformat(),
    }

    output = build_variant_summary(results, "score", "v3_sentence_avg", config)
    save_results(output, OUTPUT_FILE)

    mrr = output["aggregate"]["retrieval"]["MRR"]
    gap = output["aggregate"]["separation"]["separation_gap"]
    print(f"\n  MRR: {mrr:.4f}  |  Separation gap: {gap:.4f}")


if __name__ == "__main__":
    run()
