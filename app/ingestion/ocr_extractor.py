"""
ocr_extractor.py
----------------
Extracts text from scanned/photographed medical reports using Tesseract OCR.
Applies image pre-processing (deskew, denoise, thresholding) to improve accuracy.
"""

from __future__ import annotations

import io
import re
from typing import Union

import cv2
import numpy as np
import pytesseract
from PIL import Image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_image(source: Union[str, bytes]) -> str:
    """
    Extract and clean text from an image (JPG, PNG, TIFF, BMP, etc.).

    Args:
        source: File path string OR raw bytes from an uploaded image.

    Returns:
        Cleaned plain-text string.

    Raises:
        ValueError: If Tesseract finds no text in the image.
    """
    # ── Load image ──────────────────────────────────────────────────────────
    if isinstance(source, (str,)):
        img = cv2.imread(source)
    else:
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Check the file format.")

    # ── Pre-process ──────────────────────────────────────────────────────────
    processed = _preprocess(img)

    # ── OCR ──────────────────────────────────────────────────────────────────
    pil_img = Image.fromarray(processed)
    custom_config = r"--oem 3 --psm 6"          # LSTM engine, assume uniform block
    raw_text = pytesseract.image_to_string(pil_img, config=custom_config)

    cleaned = _clean(raw_text)

    if not cleaned.strip():
        raise ValueError(
            "Tesseract could not extract any text from the image. "
            "Ensure the image is clear, well-lit, and not rotated."
        )

    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(img: np.ndarray) -> np.ndarray:
    """
    Image pre-processing pipeline optimised for document OCR:
    1. Upscale small images (improves OCR accuracy significantly)
    2. Convert to grayscale
    3. Denoise
    4. Adaptive thresholding (handles uneven lighting / shadows)
    5. Deskew
    """
    # 1. Upscale if the image is small
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Adaptive threshold (Gaussian method)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # 5. Deskew
    deskewed = _deskew(thresh)

    return deskewed


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in a binarised image using the Hough-line method.
    Skips correction if the detected angle is negligible (< 0.5°).
    """
    coords = np.column_stack(np.where(binary_img < 127))  # foreground pixels
    if len(coords) == 0:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in (-90, 0]; normalise to (-45, 45]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:          # negligible skew
        return binary_img

    (h, w) = binary_img.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _clean(text: str) -> str:
    """Remove OCR artefacts: stray characters, double spaces, etc."""
    # Remove lines that are just punctuation or noise characters
    text = re.sub(r"(?m)^[\|\-_=~`#@\*\^]{2,}\s*$", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
