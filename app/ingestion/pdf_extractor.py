"""
pdf_extractor.py
----------------
Extracts raw text from PDF medical reports.

Strategy (automatic, transparent to callers):
  1. Try pdfplumber (fast, perfect for most PDFs).
  2. If the extracted text is garbled (broken font cmap -> (cid:N) / /gNN artefacts),
     fall back to OCR: render each page as a 300-DPI image via pdf2image and run
     Tesseract on each page image.  This reliably handles redacted, legally-processed,
     or older scanner-generated PDFs where the font encoding table is corrupt.

The caller receives clean text regardless of which path was taken.
`extraction_method` is returned as metadata so the API can report it.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


# ---------------------------------------------------------------------------
# Garbled-text detection thresholds
# ---------------------------------------------------------------------------

# Fraction of "words" that look like encoding artefacts before we give up
_GARBLE_RATIO_THRESHOLD = 0.15   # >15 % artefact tokens -> treat as garbled

# Patterns that indicate a broken font cmap
_GARBLE_RE = re.compile(
    r"""
      \(cid:\d+\)       # (cid:123)  -- pdfplumber / pypdf CID escapes
    | /g\d+             # /g3, /g88  -- raw glyph references
    | \ufffd            # Unicode replacement character
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    source: Union[str, Path, bytes],
) -> tuple[str, str]:
    """
    Extract and clean text from a PDF file.

    Args:
        source: File path (str | Path) OR raw bytes from an uploaded file.

    Returns:
        Tuple of (cleaned_text, extraction_method) where extraction_method
        is one of ``"pdfplumber"`` or ``"ocr"``.

    Raises:
        ValueError: If no text can be extracted by either method.
    """
    # Normalise to bytes so both paths share the same input
    if isinstance(source, (str, Path)):
        pdf_bytes = Path(source).read_bytes()
    else:
        pdf_bytes = source

    # -- Pass 1: pdfplumber --------------------------------------------------
    raw = _extract_with_pdfplumber(pdf_bytes)
    cleaned = _clean(raw)

    if cleaned.strip() and not _is_garbled(cleaned):
        return cleaned, "pdfplumber"

    reason = "empty" if not cleaned.strip() else "garbled font encoding"
    print(
        f"[PDF] pdfplumber output is {reason}. "
        "Falling back to OCR (pdf2image + Tesseract)."
    )

    # -- Pass 2: OCR fallback ------------------------------------------------
    ocr_text = _extract_with_ocr(pdf_bytes)
    cleaned_ocr = _clean(ocr_text)

    if not cleaned_ocr.strip():
        raise ValueError(
            "Neither pdfplumber nor OCR could extract text from this PDF. "
            "Ensure the file is not password-protected or completely blank."
        )

    return cleaned_ocr, "ocr"


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Use pdfplumber to pull text from all pages."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """
    Render each PDF page at 300 DPI and run Tesseract OCR on it.
    Uses pdf2image (Poppler wrapper) -- no subprocess calls needed.
    """
    images: list[Image.Image] = convert_from_bytes(
        pdf_bytes,
        dpi=300,
        fmt="png",
    )

    page_texts: list[str] = []
    custom_cfg = r"--oem 3 --psm 6"   # LSTM engine, uniform block layout

    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, config=custom_cfg)
        if text.strip():
            page_texts.append(f"--- Page {i} ---\n{text}")

    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# Garbled-text detector
# ---------------------------------------------------------------------------

def _is_garbled(text: str) -> bool:
    """
    Return True if a significant fraction of tokens look like encoding artefacts.

    Checks for (cid:N), /gNN, and Unicode replacement characters -- the classic
    symptoms of a PDF with a broken or non-standard font encoding map (cmap).
    """
    tokens = text.split()
    if not tokens:
        return False

    artefact_count = sum(1 for t in tokens if _GARBLE_RE.search(t))
    ratio = artefact_count / len(tokens)

    if ratio > _GARBLE_RATIO_THRESHOLD:
        print(
            f"[PDF] Garbled text detected: {artefact_count}/{len(tokens)} tokens "
            f"({ratio:.1%}) are encoding artefacts."
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Text cleaner (shared by both backends)
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Remove common PDF/OCR artefacts from medical reports:
    - Page numbers (e.g. "Page 1 of 4", "- 2 -", lone digits on a line)
    - Excessive whitespace / blank lines
    - Soft hyphens and ligatures
    - Residual (cid:N) tokens not caught by the garble detector
    """
    # Strip residual CID tokens (in case a few slip through under the threshold)
    text = _GARBLE_RE.sub(" ", text)

    # Remove page-number patterns
    text = re.sub(r"(?im)^\s*[Pp]age\s+\d+\s*(of\s*\d+)?\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-]\s*\d+\s*[-]\s*$", "", text)   # "- 2 -"
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)                # lone digit

    # Replace soft hyphens / non-breaking spaces
    text = text.replace("\u00ad", "").replace("\u00a0", " ")

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
