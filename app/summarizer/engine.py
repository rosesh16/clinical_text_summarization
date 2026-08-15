"""
engine.py
---------
Unified summarisation entry point.
Handles file-type detection, text extraction, and summarisation in one call.
"""

from __future__ import annotations

from typing import Optional

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.ocr_extractor import extract_text_from_image
from app.summarizer.hf_summarizer import summarize as hf_summarize


# Supported MIME types
PDF_TYPES   = {"application/pdf", "pdf"}
IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/tiff", "image/bmp", "image/webp",
    "jpg", "jpeg", "png", "tiff", "bmp", "webp",
}


def run(
    file_bytes: bytes,
    content_type: str,
    model_name: Optional[str] = None,
) -> dict:
    """
    Full pipeline: bytes → extract text → summarise → return structured result.

    Args:
        file_bytes:   Raw bytes of the uploaded file.
        content_type: MIME type or file extension (e.g. "application/pdf", "image/jpeg").
        model_name:   Optional HuggingFace model override.

    Returns:
        dict with keys:
            - ``extracted_text``     : raw text pulled from the document
            - ``summary``            : abstractive summary from HuggingFace
            - ``source_type``        : "pdf" or "image"
            - ``word_count``         : word count of extracted text
            - ``extraction_method``  : "pdfplumber", "ocr" (PDFs only)
    """
    ct = content_type.lower().strip()

    # ── Detect file type and extract text ────────────────────────────────────
    if any(t in ct for t in PDF_TYPES):
        extracted_text, extraction_method = extract_text_from_pdf(file_bytes)
        source_type = "pdf"

    elif any(t in ct for t in IMAGE_TYPES):
        extracted_text     = extract_text_from_image(file_bytes)
        extraction_method  = "ocr"
        source_type        = "image"

    else:
        raise ValueError(
            f"Unsupported file type: '{content_type}'. "
            "Please upload a PDF or an image (JPG, PNG, TIFF, BMP)."
        )

    word_count = len(extracted_text.split())

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary = hf_summarize(extracted_text, model_name=model_name)

    return {
        "extracted_text"    : extracted_text,
        "summary"           : summary,
        "source_type"       : source_type,
        "word_count"        : word_count,
        "extraction_method" : extraction_method,
    }
