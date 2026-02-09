import json
from pathlib import Path
from typing import List, Dict

RAW_PUBMED_PATH = Path("data/raw/pubmed/pubmed_articles.json")
PROCESSED_PUBMED_PATH = Path("data/processed/pubmed/pubmed_clean.json")


def load_raw_pubmed(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_pubmed_article(article):
    pmid = article.get("pmid")
    title = article.get("title", "").strip()
    abstract = article.get("abstract", "").strip()

    # Accept abstract-only documents
    if not abstract:
        return None

    return {
        "doc_id": f"PMID_{pmid}",
        "source": "pubmed",
        "title": title,
        "text": abstract,        # abstract as document text
        "summary": abstract,     # placeholder summary
        "meta": {
            "abstract_only": True
        }
    }

def ingest_pubmed():
    raw_articles = load_raw_pubmed(RAW_PUBMED_PATH)

    processed_docs = []
    skipped = 0

    for article in raw_articles:
        doc = normalize_pubmed_article(article)
        if doc is None:
            skipped += 1
            continue
        processed_docs.append(doc)

    PROCESSED_PUBMED_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_PUBMED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2)

    print(f"[Ingestion] Total raw articles: {len(raw_articles)}")
    print(f"[Ingestion] Processed articles: {len(processed_docs)}")
    print(f"[Ingestion] Skipped articles: {skipped}")


if __name__ == "__main__":
    ingest_pubmed()