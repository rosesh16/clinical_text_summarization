"""
scripts/prepare_dataset.py
--------------------------
Step 3 of the fine-tuning pipeline.

Downloads the `ccdv/pubmed-summarization` dataset from HuggingFace Hub,
filters low-quality samples, and saves clean JSON splits to:
    data/processed/pubmed_hf/train.json
    data/processed/pubmed_hf/val.json
    data/processed/pubmed_hf/test.json

Each record:
    { "article": "<full paper text>", "abstract": "<reference summary>" }

Run from the project root:
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME   = "ccdv/pubmed-summarization"
TRAIN_SAMPLES  = 20_000     # increase to 50K or 119K for more accuracy
VAL_SAMPLES    = 2_000
TEST_SAMPLES   = 2_000

MIN_ARTICLE_WORDS  = 100    # skip articles shorter than this
MIN_ABSTRACT_WORDS = 30     # skip abstracts shorter than this

OUTPUT_DIR = Path("data/processed/pubmed_hf")

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Basic cleanup shared between article and abstract text."""
    if not text:
        return ""
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove citation markers like [1], [1,2], (Author et al., 2020)
    text = re.sub(r"\[[0-9,\s]+\]", "", text)
    text = re.sub(r"\([A-Z][a-z]+ et al\.,?\s*\d{4}\)", "", text)
    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())
    return text.strip()


def is_valid(article: str, abstract: str) -> bool:
    """Return True only if both fields meet minimum quality thresholds."""
    if not article or not abstract:
        return False
    if len(article.split()) < MIN_ARTICLE_WORDS:
        return False
    if len(abstract.split()) < MIN_ABSTRACT_WORDS:
        return False
    return True


def process_split(raw_split, max_samples: int) -> list[dict]:
    """Clean and filter a HuggingFace dataset split."""
    records = []
    for row in raw_split:
        article  = clean(row.get("article", ""))
        abstract = clean(row.get("abstract", ""))
        if not is_valid(article, abstract):
            continue
        records.append({"article": article, "abstract": abstract})
        if len(records) >= max_samples:
            break
    return records


def save(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(records):,} records -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 3 -- Downloading & preparing ccdv/pubmed-summarization")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "\n[ERROR] `datasets` package not found.\n"
            "Run:  .venv\\Scripts\\python.exe -m pip install 'datasets>=2.14'\n"
        )

    print("\n[1/4] Loading dataset from HuggingFace Hub (may download ~2GB)...")
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    print(f"      Raw sizes -- train: {len(ds['train']):,} | "
          f"val: {len(ds['validation']):,} | test: {len(ds['test']):,}")

    print(f"\n[2/4] Processing train split  (target: {TRAIN_SAMPLES:,} samples)...")
    train = process_split(ds["train"], TRAIN_SAMPLES)

    print(f"\n[3/4] Processing val split    (target: {VAL_SAMPLES:,} samples)...")
    val = process_split(ds["validation"], VAL_SAMPLES)

    print(f"\n[4/4] Processing test split   (target: {TEST_SAMPLES:,} samples)...")
    test = process_split(ds["test"], TEST_SAMPLES)

    print("\nSaving splits...")
    save(train, OUTPUT_DIR / "train.json")
    save(val,   OUTPUT_DIR / "val.json")
    save(test,  OUTPUT_DIR / "test.json")

    print("\nDataset preparation complete!")
    print(f"   Train : {len(train):,} samples")
    print(f"   Val   : {len(val):,} samples")
    print(f"   Test  : {len(test):,} samples")
    print(f"   Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
