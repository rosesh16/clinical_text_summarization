import json
from pathlib import Path

INPUT_PATH = Path("data/processed/pubmed/pubmed_sentences.json")
OUTPUT_PATH = Path("data/processed/pubmed/pubmed_chunks.json")

CHUNK_SIZE = 8
STRIDE = 6  # overlap of 2 sentences


def chunk_sentences(sentences, chunk_size, stride):
    """
    Create overlapping sentence chunks.
    """
    chunks = []
    start = 0
    idx = 0

    while start < len(sentences):
        chunk = sentences[start:start + chunk_size]
        if not chunk:
            break

        chunks.append({
            "chunk_index": idx,
            "sentences": chunk
        })

        idx += 1
        start += stride

    return chunks


def chunk_documents():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunked_docs = []

    for doc in documents:
        sentence_chunks = chunk_sentences(
            doc["sentences"],
            CHUNK_SIZE,
            STRIDE
        )

        chunked_docs.append({
            "doc_id": doc["doc_id"],
            "source": doc.get("source"),
            "chunks": [
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{c['chunk_index']}",
                    "sentences": c["sentences"]
                }
                for c in sentence_chunks
            ],
            "meta": doc.get("meta", {})
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, indent=2, ensure_ascii=False)

    print(f"[Chunking] Documents processed: {len(chunked_docs)}")
    print(f"[Chunking] Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_documents()