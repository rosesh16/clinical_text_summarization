git s# Clinical Text Summarization

A FastAPI-based web application that accepts medical PDF reports or scanned images, extracts their text, and returns a structured, patient-friendly summary powered by HuggingFace transformers.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture & Pipeline](#architecture--pipeline)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [PDF Extraction: Smart OCR Fallback](#pdf-extraction-smart-ocr-fallback)
- [HuggingFace Summarization](#huggingface-summarization)
- [Research Pipeline (PubMed Baselines)](#research-pipeline-pubmed-baselines)
- [Known Issues & Fixes Applied](#known-issues--fixes-applied)

---

## Overview

This project was built as part of a medical NLP internship. It has two distinct components:

| Component | Description |
|---|---|
| **Web App** (`app/`) | FastAPI backend + HTML frontend — upload a medical report, get a structured summary |
| **Research Pipeline** (`src/`) | Extractive summarization baselines (Lead-3, TextRank) on PubMed abstracts |

---

## Project Structure

```
clinical_text_summarization/
│
├── app/                          # FastAPI web application
│   ├── main.py                   # API entry point (routes, CORS, file validation)
│   ├── postprocessor.py          # Structures summary into labelled sections + flags
│   ├── ingestion/
│   │   ├── pdf_extractor.py      # PDF text extraction (pdfplumber + OCR fallback)
│   │   └── ocr_extractor.py      # Image OCR (OpenCV pre-processing + Tesseract)
│   └── summarizer/
│       ├── engine.py             # Unified pipeline: file → text → summary
│       └── hf_summarizer.py      # HuggingFace abstractive summarizer
│
├── frontend/
│   └── index.html                # Patient-facing upload UI
│
├── src/                          # Research / baseline pipeline
│   ├── baselines/
│   │   ├── lead3.py              # Lead-3 extractive baseline
│   │   └── textrank.py           # TextRank extractive baseline
│   ├── evaluation/
│   │   ├── evaluate.py           # ROUGE-L + BERTScore + redundancy metrics
│   │   ├── bertscore.py
│   │   ├── rouge_1.py
│   │   └── redundancy.py
│   ├── preprocessing/
│   │   ├── download_dataset.py   # Fetch PubMed abstracts via NCBI API
│   │   ├── data_ingestion.py
│   │   ├── preprocess_raw.py
│   │   ├── preprocess_sentence.py
│   │   ├── chunk_long_docs.py
│   │   └── split_pubmed_dataset.py
│   ├── exception.py
│   └── logger.py
│
├── data/                         # Created by preprocessing scripts
├── experiments/
├── requirements.txt
├── setup.py
├── utils.py
└── test_summarizer.py            # Standalone smoke test for hf_summarizer
```

---

## Architecture & Pipeline

```
User uploads PDF or Image
         │
         ▼
┌─────────────────────────────┐
│     FastAPI  POST /summarize │
│         app/main.py          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│            engine.py                    │
│   Detect file type → route to extractor │
└────────┬──────────────┬─────────────────┘
         │ PDF           │ Image
         ▼               ▼
┌────────────────┐  ┌──────────────────────┐
│ pdf_extractor  │  │   ocr_extractor      │
│                │  │                      │
│ 1. pdfplumber  │  │ OpenCV pre-process:  │
│    (fast path) │  │  - upscale           │
│                │  │  - grayscale         │
│ 2. Garble      │  │  - denoise           │
│    detect?     │  │  - adaptive thresh   │
│    (cid:N)/gNN │  │  - deskew            │
│                │  │                      │
│ 3. OCR fallback│  │ Tesseract OCR        │
│  pdf2image     │  │ (--oem 3 --psm 6)    │
│  300 DPI       │  └──────────────────────┘
│  Tesseract     │
└────────┬───────┘
         │
         ▼
┌──────────────────────────────┐
│       hf_summarizer.py        │
│                               │
│  Short doc → single pass      │
│  Long doc  → chunk (675 words)│
│             → summarise each  │
│             → merge → re-sum  │
│                               │
│  Model: Falconsai/medical_sum │
│  Fallback: facebook/bart-cnn  │
└──────────────┬────────────────┘
               │
               ▼
┌──────────────────────────────┐
│       postprocessor.py        │
│                               │
│  Classify sentences into:     │
│   • Key Findings              │
│   • Diagnosis / Impression    │
│   • Medications / Treatment   │
│   • Next Steps / Follow-up    │
│   • General Notes             │
│                               │
│  Detect flags:                │
│   ⚠ Elevated / Low values    │
│   ⚠ Critical values          │
│   ℹ Medication mentioned      │
│   ℹ Follow-up recommended    │
└──────────────────────────────┘
               │
               ▼
        JSON Response → UI
```

---

## Key Features

- **Smart PDF extraction** — automatically detects garbled text from broken font encodings (`(cid:N)`, `/gNN`) and falls back to OCR without any manual intervention
- **Image OCR pipeline** — full pre-processing (upscale, denoise, adaptive threshold, deskew) before Tesseract for high accuracy on scanned documents
- **Long document chunking** — handles reports longer than 675 words via two-pass chunk-and-merge summarization
- **Structured output** — summary is parsed and labelled into medical sections (findings, diagnosis, medications, next steps)
- **Abnormal value flags** — automatically highlights elevated/critical/low values and medication/follow-up mentions
- **Extraction method transparency** — API response includes `extraction_method: "pdfplumber"` or `"ocr"` so you always know which path was taken

---

## Prerequisites

### System dependencies

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.9 – 3.11** | Runtime (3.10.x recommended) | [python.org](https://www.python.org/downloads/) |
| **Tesseract OCR** | Text recognition from images/scanned PDFs | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) |
| **Poppler** | PDF-to-image rendering (used by `pdf2image`) | [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases) |

#### Tesseract (Windows)
1. Download and run the UB-Mannheim installer.
2. Add the install path (e.g. `C:\Program Files\Tesseract-OCR`) to your system `PATH`.

#### Poppler (Windows)
1. Download the latest zip from the releases page above.
2. Extract to e.g. `C:\poppler`.
3. Add `C:\poppler\Library\bin` to your system `PATH`.
4. Restart your terminal after updating `PATH`.

---

## Installation

```powershell
# 1. Clone the repo (or open the project folder)
cd c:\Users\ASUS\clinical_text_summarization

# 2. Create a fresh virtual environment (Python 3.10 recommended)
python -m venv .venv
.venv\Scripts\activate

# 3. Upgrade bootstrap tools
pip install --upgrade pip setuptools wheel

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `torch` is a large download (~2 GB for CPU). The first `pip install` will take a few minutes. Subsequent runs use the cache.

---

## Running the Application

```powershell
# Activate the venv if not already active
.venv\Scripts\activate

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser:

| URL | Description |
|---|---|
| `http://localhost:8000` | Patient-facing upload UI |
| `http://localhost:8000/docs` | Interactive Swagger API docs |
| `http://localhost:8000/health` | Liveness check |

> The first request that triggers summarization will download the HuggingFace model (~500 MB for BART). Subsequent requests use the cached model.

### Quick smoke test (no server needed)

```powershell
python test_summarizer.py
```

---

## API Reference

### `POST /summarize`

Upload a medical report file and receive a structured summary.

**Request:** `multipart/form-data` with a `file` field.

**Supported formats:** `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`

**Response:**
```json
{
  "summary":           "Concise abstractive summary of the report...",
  "key_findings":      ["Troponin I elevated at 12.4 ng/mL", "..."],
  "diagnosis":         ["Inferior STEMI", "..."],
  "medications":       ["Metformin 1000mg BD", "..."],
  "next_steps":        ["Cardiac rehabilitation referral", "..."],
  "general_notes":     ["..."],
  "flags": [
    "⚠ Elevated values detected",
    "⚠ Critical values mentioned",
    "ℹ Medication mentioned",
    "ℹ Follow-up recommended"
  ],
  "word_count":         312,
  "source_type":        "pdf",
  "extraction_method":  "ocr",
  "filename":           "patient_report.pdf"
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## PDF Extraction: Smart OCR Fallback

Some PDFs — particularly those generated by legal redaction tools, older medical document software, or certain scanners — have a **corrupt font encoding map (cmap)**. This causes text-extraction libraries like `pdfplumber` to produce garbled output such as:

```
Case:/g3/g3MD/g88209/g882...Physician:/g3MD/g3
Case:(cid:2)(cid:2)MD(cid:3)09(cid:3)...
```

The PDF renders correctly visually because glyph *shapes* are correct — but the *character codes* mapped to those glyphs are wrong. No amount of switching text libraries fixes this; the fault is in the font data.

### How the fallback works

```
pdfplumber extract
      │
      ├─ >15% tokens match (cid:N) or /gNN? ──YES──→ OCR fallback
      │                                                    │
      └─ NO → return pdfplumber text ✓         pdf2image (300 DPI PNG)
                                                    │
                                               Tesseract OCR per page
                                                    │
                                               return OCR text ✓
```

This is implemented in [`app/ingestion/pdf_extractor.py`](app/ingestion/pdf_extractor.py). The caller (and the frontend) always receives clean text — the method is reported in `extraction_method` in the API response.

---

## HuggingFace Summarization

Implemented in [`app/summarizer/hf_summarizer.py`](app/summarizer/hf_summarizer.py).

| Setting | Value |
|---|---|
| Primary model | `Falconsai/medical_summarization` (T5-based, biomedical fine-tuned) |
| Fallback model | `facebook/bart-large-cnn` (general, high quality) |
| Max chunk size | 675 words (~900 tokens) |
| Summary length | 60 – 200 tokens |
| Long-doc strategy | Chunk → summarise each → merge → final summarise |

> **Implementation note:** The code uses `AutoModelForSeq2SeqLM` + `AutoTokenizer` directly (not `pipeline("summarization")`), which makes it compatible with all versions of `transformers` including 5.x, which removed the `"summarization"` task from its pipeline registry.

---

## Research Pipeline (PubMed Baselines)

The original internship pipeline for extractive summarization over PubMed abstracts. Run in order:

```powershell
# 1. Set your NCBI email in download_dataset.py, then:
python src/preprocessing/download_dataset.py
python src/preprocessing/data_ingestion.py
python src/preprocessing/preprocess_raw.py
python src/preprocessing/preprocess_sentence.py
python src/preprocessing/chunk_long_docs.py
python src/preprocessing/split_pubmed_dataset.py

# 2. Generate baseline summaries
python src/baselines/lead3.py --split train --output data/preds/lead3_train.json
python src/baselines/textrank.py --split train --output data/preds/textrank_train.json

# 3. Evaluate
python src/evaluation/evaluate.py --predictions data/preds/lead3_train.json --split train
```

Data is read/written under `data/`:
- Raw: `data/raw/pubmed/pubmed_articles.json`
- Processed: `data/processed/pubmed/`
- Splits: `data/processed/pubmed/pubmed_{train|val|test}.json`

---

## Known Issues & Fixes Applied

| Issue | Root Cause | Fix |
|---|---|---|
| `No module named 'pkg_resources'` on `pip install` | pip 26+ no longer pre-installs `setuptools` in venvs | Added `setuptools>=68.0` + `wheel` to top of `requirements.txt` |
| `pandas==2.0.3` build failure | No prebuilt wheel for newer Python; needs MSVC to compile | Relaxed to `pandas>=2.1` |
| `Unknown task summarization` in transformers 5.x | transformers 5.x removed the `"summarization"` pipeline task | Rewrote `hf_summarizer.py` to use `AutoModelForSeq2SeqLM` directly |
| `tokenizers` build failure on Python 3.13 | No prebuilt wheel; building from Rust source requires `link.exe` (MSVC) | Switched to Python 3.10 which has prebuilt wheels for all packages |
| Garbled PDF text (`(cid:N)`, `/gNN`) | Broken font cmap in the PDF — not fixable by switching extraction libraries | Added automatic OCR fallback in `pdf_extractor.py` using `pdf2image` + Tesseract |
