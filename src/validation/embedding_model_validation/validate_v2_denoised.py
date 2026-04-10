"""
V2 — Sentence Denoised: Hard Threshold Filtering
--------------------------------------------------
Split text into sentences, embed each, compute centroid,
then DROP sentences whose similarity to the centroid falls
below a threshold. Mean-pool the survivors.

The idea: job postings contain boilerplate (legal disclaimers,
benefits lists, equal-opportunity statements) that dilutes the
semantic signal. Removing low-relevance sentences should sharpen
the embedding toward the role's actual content.

Threshold choice: 0.70
We use 0.70 rather than the original 0.75 because BGE-large produces
tighter similarity distributions than BGE-small — a higher threshold
would drop too many sentences. 0.70 is conservative enough to keep
substantive content while still filtering clear boilerplate.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from validation_utils import (
    load_data, split_sentences, cosine, save_results, build_variant_summary,
)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
THRESHOLD = 0.70
DATA_FILE = Path(__file__).parent / "validation_dataset.xlsx"
OUTPUT_FILE = Path(__file__).parent / "results_v2_denoised.json"


def sentence_denoised_embedding(text, model, threshold=THRESHOLD):
    """
    Hard-threshold sentence filtering:
    1. Embed each sentence
    2. Compute centroid of all sentence embeddings
    3. Drop sentences with cosine(sentence, centroid) < threshold
    4. Mean-pool the remaining embeddings

    Returns (embedding, kept_sentences, removed_sentences).
    """
    sentences = split_sentences(text)
    if not sentences:
        return np.zeros(model.get_sentence_embedding_dimension()), [], []

    sent_embs = model.encode(sentences)
    centroid = np.mean(sent_embs, axis=0, keepdims=True)
    sims = cosine_similarity(sent_embs, centroid).flatten()

    keep_idx = [i for i, s in enumerate(sims) if s >= threshold]

    # fallback: if everything gets filtered, keep the most central sentence
    if not keep_idx:
        keep_idx = [int(np.argmax(sims))]

    doc_emb = np.mean(sent_embs[keep_idx], axis=0)
    kept = [sentences[i] for i in keep_idx]
    removed = [sentences[i] for i in range(len(sentences)) if i not in keep_idx]

    return doc_emb, kept, removed


def run():
    print("V2 — Sentence denoised (hard threshold)")
    print("=" * 50)

    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    jobs, courses = load_data(DATA_FILE)

    print("Computing denoised embeddings & similarities...")
    results = []
    # save a few sentence-level examples for interpretability
    sentence_examples = []

    for jid, j in tqdm(jobs.items()):
        j_emb, j_kept, j_removed = sentence_denoised_embedding(j["text"], model)

        # save one example per job for inspection
        if len(sentence_examples) < 6:
            sentence_examples.append({
                "job": jid,
                "category": j["category"],
                "kept_sentences": j_kept[:5],
                "removed_sentences": j_removed[:5],
                "num_kept": len(j_kept),
                "num_removed": len(j_removed),
            })

        for cid, c in courses.items():
            c_emb, _, _ = sentence_denoised_embedding(c["text"], model)
            results.append({
                "job": jid,
                "course": cid,
                "job_category": j["category"],
                "course_category": c["category"],
                "score": cosine(j_emb, c_emb),
            })

    config = {
        "variant": "v2_denoised",
        "embedding_model": EMBEDDING_MODEL,
        "method": "sentence_hard_threshold",
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat(),
    }

    output = build_variant_summary(results, "score", "v2_denoised", config)
    output["sentence_examples"] = sentence_examples
    save_results(output, OUTPUT_FILE)

    mrr = output["aggregate"]["retrieval"]["MRR"]
    gap = output["aggregate"]["separation"]["separation_gap"]
    print(f"\n  MRR: {mrr:.4f}  |  Separation gap: {gap:.4f}")


if __name__ == "__main__":
    run()
