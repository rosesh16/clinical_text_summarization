# Suvidha Internship — PubMed Summarization Pipeline

This project builds a small extractive summarization pipeline over PubMed abstracts. It includes scripts for downloading and preprocessing data, two baseline summarizers (Lead-3 and TextRank), and an evaluation script using ROUGE-L, BERTScore, and a redundancy metric.

## Project Layout

- `src/preprocessing/`: data download + cleaning + sentence splitting + chunking + dataset split
- `src/baselines/`: extractive baselines (`lead3.py`, `textrank.py`)
- `src/evaluation/`: evaluation metrics and `evaluate.py`
- `data/`: raw and processed datasets (created by scripts)

## Quick Start

1. Create a virtual environment and install dependencies.

2. Download and prepare data:
```bash
python src/preprocessing/download_dataset.py
python src/preprocessing/data_ingestion.py
python src/preprocessing/preprocess_raw.py
python src/preprocessing/preprocess_sentence.py
python src/preprocessing/chunk_long_docs.py
python src/preprocessing/split_pubmed_dataset.py
```

3. Generate baseline summaries:
```bash
python src/baselines/lead3.py --split train --output data/preds/lead3_train.json
python src/baselines/textrank.py --split train --output data/preds/textrank_train.json
```

4. Evaluate predictions:
```bash
python src/evaluation/evaluate.py --predictions data/preds/lead3_train.json --split train
```

## Notes and Assumptions

- `src/evaluation/evaluate.py` currently uses a **temporary reference**: the first 5 sentences of each document as a proxy summary. This is intended to validate the metric pipeline, not to reflect true model quality.
- `src/preprocessing/download_dataset.py` requires setting a real email for NCBI API access. Update `EMAIL` in that file.
- Some runtime dependencies used by the code are not listed in `requirements.txt` (e.g., `requests`, `nltk`, `scikit-learn`, `spacy`, `networkx`, `rouge_score`, `bert_score`). Install them as needed.

## Expected Data Paths

Scripts read/write under `data/`:
- Raw: `data/raw/pubmed/pubmed_articles.json`
- Processed: `data/processed/pubmed/`
- Splits: `data/processed/pubmed/pubmed_{train|val|test}.json`

