"""
hf_summarizer.py
----------------
HuggingFace-based abstractive summarizer using a biomedical-fine-tuned model.

Model priority (auto-selected at startup):
  1. Local fine-tuned model : models/finetuned-medical-t5/final  (if present)
  2. Remote HF model        : Falconsai/medical_summarization     (fallback)
  3. General fallback       : facebook/bart-large-cnn

To update the local model, run:  python scripts/finetune.py

Long documents are handled by chunking: each chunk is summarised individually,
then the chunk-summaries are combined and summarised again (two-pass).
"""

from __future__ import annotations

import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Optional

# transformers / torch are imported lazily inside _load_pipeline()
# so that the FastAPI process can start even when torch DLLs are unavailable.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the locally fine-tuned model (relative to the project root)
_LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "finetuned-medical-t5" / "final"
_FALCONSAI_MODEL  = "Falconsai/medical_summarization"
FALLBACK_MODEL    = "facebook/bart-large-cnn"


def _resolve_primary_model() -> str:
    """
    Return the best available model identifier at startup.

    Priority:
      1. Local fine-tuned model (models/finetuned-medical-t5/final) — if present
      2. Falconsai/medical_summarization                             — remote HF
    """
    if _LOCAL_MODEL_PATH.exists() and any(_LOCAL_MODEL_PATH.iterdir()):
        print(f"[HF] Local fine-tuned model found: {_LOCAL_MODEL_PATH}")
        return str(_LOCAL_MODEL_PATH)
    print(f"[HF] No local model found at {_LOCAL_MODEL_PATH}.")
    print(f"[HF] Using remote model: {_FALCONSAI_MODEL}")
    print(f"[HF] Tip: run `python scripts/finetune.py` to create a local fine-tuned model.")
    return _FALCONSAI_MODEL


PRIMARY_MODEL = _resolve_primary_model()

# Safe token limits (conservative — actual model limits are higher)
MAX_INPUT_TOKENS = 900       # tokens per chunk fed to the model
WORDS_PER_TOKEN  = 0.75      # rough approximation: 1 token ≈ 0.75 words
CHUNK_WORD_LIMIT = int(MAX_INPUT_TOKENS * WORDS_PER_TOKEN)   # ~675 words per chunk

SUMMARY_MIN_LEN  = 60
SUMMARY_MAX_LEN  = 200


# ---------------------------------------------------------------------------
# Model loader (singleton — loaded once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    """Load and cache the summarization pipeline (lazy torch import)."""
    import torch
    from transformers import pipeline  # deferred — avoids torch DLL crash at startup

    device = 0 if torch.cuda.is_available() else -1   # 0 = first GPU, -1 = CPU
    device_label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"[HF] Loading model: {model_name} (first call only)...")
    print(f"[HF] Inference device: {device_label}")
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        device=device,
    )


def get_pipeline(model_name: Optional[str] = None):
    """Return the cached pipeline, falling back gracefully if model unavailable."""
    target = model_name or PRIMARY_MODEL
    try:
        return _load_pipeline(target)
    except Exception as exc:
        print(f"[HF] Could not load '{target}': {exc}\n     Falling back to {FALLBACK_MODEL}.")
        return _load_pipeline(FALLBACK_MODEL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize(text: str, model_name: Optional[str] = None) -> str:
    """
    Summarise arbitrary-length medical text using HuggingFace.

    Strategy:
      - Short text (≤ CHUNK_WORD_LIMIT words) → single-pass summarisation
      - Long text                              → chunk → summarise → merge → summarise

    Args:
        text:       Raw extracted text from the medical report.
        model_name: Override the default model (optional).

    Returns:
        A concise abstractive summary string.
    """
    text = text.strip()
    if not text:
        return "No text provided for summarisation."

    words = text.split()

    if len(words) <= CHUNK_WORD_LIMIT:
        return _summarise_chunk(text, model_name)

    # ── Long document: chunk → summarise each → summarise merged ────────────
    chunks   = _split_into_chunks(words, CHUNK_WORD_LIMIT)
    partials = [_summarise_chunk(c, model_name) for c in chunks]
    merged   = " ".join(partials)

    # Final pass — summarise the combined partial summaries
    if len(merged.split()) > CHUNK_WORD_LIMIT:
        merged = " ".join(merged.split()[:CHUNK_WORD_LIMIT])

    return _summarise_chunk(merged, model_name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarise_chunk(text: str, model_name: Optional[str]) -> str:
    """Run the model on a single chunk and return the summary string."""
    nlp = get_pipeline(model_name)
    result = nlp(
        text,
        max_length=SUMMARY_MAX_LEN,
        min_length=SUMMARY_MIN_LEN,
        do_sample=False,
    )
    return result[0]["summary_text"].strip()


def _split_into_chunks(words: list[str], chunk_size: int) -> list[str]:
    """
    Split a word list into overlapping chunks.
    Overlap of 50 words preserves sentence context across chunk boundaries.
    """
    overlap = 50
    chunks  = []
    start   = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap   # slide with overlap

    return chunks
