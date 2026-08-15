"""
main.py
-------
FastAPI application entry point for the Medical Report Summarizer.

Endpoints:
  POST /summarize   — accepts PDF or image upload, returns structured summary
  GET  /health      — liveness check
  GET  /            — serves the frontend HTML

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.summarizer.engine import run as run_pipeline
from app.postprocessor import structure


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Medical Report Summarizer",
    description = "Upload a PDF or image of a medical report and receive a patient-friendly summary.",
    version     = "1.0.0",
)

# Allow frontend served from a different origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Path to the bundled frontend
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the patient-facing upload UI."""
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend.</h1>", status_code=404)
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/summarize")
async def summarize_report(file: UploadFile = File(...)):
    """
    Accept a PDF or image upload of a medical report.

    Returns a JSON object with:
      - ``summary``        : full abstractive summary
      - ``key_findings``   : list of finding sentences
      - ``diagnosis``      : list of diagnosis/impression sentences
      - ``medications``    : list of medication sentences
      - ``next_steps``     : list of follow-up/recommendation sentences
      - ``general_notes``  : other sentences
      - ``flags``          : detected abnormal-value indicators
      - ``word_count``     : word count of the extracted text
      - ``source_type``    : "pdf" or "image"
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # ── Validate file type ────────────────────────────────────────────────────
    ext = Path(file.filename).suffix.lower().lstrip(".")
    allowed_extensions = {"pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"}

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    # ── Read file bytes ───────────────────────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        content_type = file.content_type or ext
        result = run_pipeline(file_bytes, content_type=content_type)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Summarisation failed. Check server logs for details."
        )

    # ── Structure output ──────────────────────────────────────────────────────
    structured = structure(result["summary"], result["extracted_text"])
    structured["word_count"]         = result["word_count"]
    structured["source_type"]        = result["source_type"]
    structured["extraction_method"]  = result.get("extraction_method", "pdfplumber")
    structured["filename"]           = file.filename

    return JSONResponse(content=structured)
