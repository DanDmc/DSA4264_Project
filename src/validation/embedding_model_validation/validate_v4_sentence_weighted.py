"""
V4 — Sentence Weighted: Soft Centroid Weighting
-------------------------------------------------
Split text into sentences, embed each, then compute a weighted
mean where each sentence's weight = its cosine similarity to
the document centroid.

This is the "best of both worlds" hypothesis:
- Unlike v2 (hard threshold), no information is destroyed
- Unlike v3 (equal weights), boilerplate gets downweighted
- Sentences that are central to the document's meaning
  contribute more to the final embedding

The weighting is just raw cosine similarity to centroid (no
softmax, no temperature). Keeping it simple because:
1. Cosine sims to centroid are already in [0, 1] for normalized vecs
2. Adding softmax/temperature would introduce another hyperparameter
   we'd need to tune — not worth it for a validation experiment
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
DATA_FILE = Path(__file__).parent / "validation_dataset.xlsx"
OUTPUT_FILE = Path(__file__).parent / "results_v4_sentence_weighted.json"


def sentence_weighted_embedding(text, model):
    """
    Centroid-weighted sentence mean pooling:
    1. Split text into sentences, embed each
    2. Compute centroid (unweighted mean)
    3. Weight each sentence by its cosine similarity to centroid
    4. Compute weighted mean of sentence embeddings

    Returns (embedding, sentence_weights) where sentence_weights
    is a list of (sentence_text, weight) for interpretability.
    """
    sentences = split_sentences(text)
    if not sentences:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim), []

    sent_embs = model.encode(sentences)
    centroid = np.mean(sent_embs, axis=0, keepdims=True)

    # cosine similarity of each sentence to the centroid
    weights = cosine_similarity(sent_embs, centroid).flatten()

    # weighted mean: each sentence contributes proportionally
    # to how "central" it is to the document
    weights_norm = weights / weights.sum()
    doc_emb = np.average(sent_embs, axis=0, weights=weights_norm)

    sentence_weights = [
        (sentences[i], round(float(weights[i]), 4))
        for i in range(len(sentences))
    ]

    return doc_emb, sentence_weights


def run():
    print("V4 — Sentence weighted (centroid similarity)")
    print("=" * 50)

    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    jobs, courses = load_data(DATA_FILE)

    print("Computing weighted embeddings & similarities...")
    results = []
    # save sentence-weight examples for a few jobs (interpretability)
    weight_examples = []

    for jid, j in tqdm(jobs.items()):
        j_emb, j_weights = sentence_weighted_embedding(j["text"], model)

        # save top/bottom weighted sentences for first job per category
        seen_cats = {ex["category"] for ex in weight_examples}
        if j["category"] not in seen_cats:
            sorted_w = sorted(j_weights, key=lambda x: x[1], reverse=True)
            weight_examples.append({
                "job": jid,
                "category": j["category"],
                "highest_weight_sentences": [
                    {"text": s[:150], "weight": w} for s, w in sorted_w[:3]
                ],
                "lowest_weight_sentences": [
                    {"text": s[:150], "weight": w} for s, w in sorted_w[-3:]
                ],
                "num_sentences": len(j_weights),
            })

        for cid, c in courses.items():
            c_emb, _ = sentence_weighted_embedding(c["text"], model)
            results.append({
                "job": jid,
                "course": cid,
                "job_category": j["category"],
                "course_category": c["category"],
                "score": cosine(j_emb, c_emb),
            })

    config = {
        "variant": "v4_sentence_weighted",
        "embedding_model": EMBEDDING_MODEL,
        "method": "sentence_centroid_weighted_mean",
        "timestamp": datetime.now().isoformat(),
    }

    output = build_variant_summary(results, "score", "v4_sentence_weighted", config)
    output["weight_examples"] = weight_examples
    save_results(output, OUTPUT_FILE)

    mrr = output["aggregate"]["retrieval"]["MRR"]
    gap = output["aggregate"]["separation"]["separation_gap"]
    print(f"\n  MRR: {mrr:.4f}  |  Separation gap: {gap:.4f}")


if __name__ == "__main__":
    run()
