import json
import argparse
from pathlib import Path

import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_data(split: str):
    base_dir = Path.cwd()
    data_path = base_dir / "data" / "processed" / "pubmed" / f"pubmed_{split}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_sentences(document):
    sentences = []
    for chunk in document["chunks"]:
        sentences.extend(chunk["sentences"])
    return sentences


def textrank_summary(sentences, top_k=3):
    if len(sentences) == 0:
        return ""

    if len(sentences) <= top_k:
        return " ".join(sentences)

    # TF-IDF sentence embeddings
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    # Similarity matrix
    sim_matrix = cosine_similarity(tfidf)

    # Build graph
    graph = nx.from_numpy_array(sim_matrix)

    # PageRank
    scores = nx.pagerank(graph)

    # Rank sentences
    ranked = sorted(
        ((scores[i], i, s) for i, s in enumerate(sentences)),
        reverse=True
    )

    # Select top-k sentences
    selected = sorted(ranked[:top_k], key=lambda x: x[1])

    return " ".join([s for (_, _, s) in selected])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):
    data = load_data(args.split)
    predictions = []

    for doc in data:
        sentences = flatten_sentences(doc)
        summary = textrank_summary(sentences, top_k=3)

        predictions.append({
            "doc_id": doc["doc_id"],
            "summary": summary
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"TextRank summaries generated: {len(predictions)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextRank baseline for PubMed summarization")

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True
    )

    args = parser.parse_args()
    main(args)