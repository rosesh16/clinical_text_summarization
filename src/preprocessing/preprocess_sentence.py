import json
from pathlib import Path
import spacy

# Load spaCy English model (scientific text friendly)
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])

INPUT_PATH = Path("data/processed/pubmed/pubmed_preprocessed.json")
OUTPUT_PATH = Path("data/processed/pubmed/pubmed_sentences.json")


def split_sentences(text: str):
    """
    Split text into sentences using spaCy.
    """
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def preprocess_sentences():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    processed_docs = []

    for doc in documents:
        sentences = split_sentences(doc["text"])
        summary_sentences = split_sentences(doc["summary"])

        processed_docs.append({
            "doc_id": doc["doc_id"],
            "source": doc.get("source"),
            "sentences": sentences,
            "summary_sentences": summary_sentences,
            "meta": doc.get("meta", {})
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)

    print(f"[Sentence Split] Documents processed: {len(processed_docs)}")
    print(f"[Sentence Split] Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess_sentences()