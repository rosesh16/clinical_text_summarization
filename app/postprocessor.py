"""
postprocessor.py
----------------
Structures the raw HuggingFace summary into patient-friendly sections
by detecting common medical report keywords.

Sections attempted (shown only when relevant content is found):
  • Key Findings
  • Diagnosis / Impression
  • Medications / Treatment
  • Next Steps / Follow-up
  • General Summary (always present as fallback)
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Section keyword patterns
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = {
    "diagnosis": re.compile(
        r"\b(diagnos\w*|impression|assessment|condition|disorder|disease|syndrome)\b",
        re.IGNORECASE,
    ),
    "findings": re.compile(
        r"\b(finding\w*|result\w*|level\w*|value\w*|test\w*|report\w*|show\w*|reveal\w*|detect\w*|abnormal\w*|normal\w*)\b",
        re.IGNORECASE,
    ),
    "medications": re.compile(
        r"\b(medic\w*|drug\w*|prescri\w*|tablet\w*|capsule\w*|dose\w*|dosage\w*|mg\b|treatment\w*|therap\w*)\b",
        re.IGNORECASE,
    ),
    "next_steps": re.compile(
        r"\b(follow.?up|recommend\w*|refer\w*|schedule\w*|return\w*|monitor\w*|repeat\w*|consult\w*|advised?\w*)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def structure(summary: str, extracted_text: str) -> dict:
    """
    Parse the summary and extracted text to produce a structured patient card.

    Args:
        summary:        HuggingFace abstractive summary.
        extracted_text: Full extracted text (used for section detection).

    Returns:
        dict with structured sections for the frontend.
    """
    sentences = _split_sentences(summary)

    sections: dict[str, list[str]] = {
        "findings"    : [],
        "diagnosis"   : [],
        "medications" : [],
        "next_steps"  : [],
        "general"     : [],
    }

    for sent in sentences:
        placed = False
        for section, pattern in _SECTION_PATTERNS.items():
            if pattern.search(sent):
                sections[section].append(sent.strip())
                placed = True
                break
        if not placed:
            sections["general"].append(sent.strip())

    # ── Build readable labels ─────────────────────────────────────────────────
    structured = {
        "summary"            : summary,
        "key_findings"       : sections["findings"]   or [],
        "diagnosis"          : sections["diagnosis"]  or [],
        "medications"        : sections["medications"] or [],
        "next_steps"         : sections["next_steps"] or [],
        "general_notes"      : sections["general"]    or [],
    }

    # ── Flag any detected abnormal values ────────────────────────────────────
    structured["flags"] = _detect_flags(extracted_text)

    return structured


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter (handles '. ', '! ', '? ' boundaries)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _detect_flags(text: str) -> list[str]:
    """
    Detect common abnormal-value indicators in the raw report text.
    Returns a list of short warning strings shown in the UI.
    """
    flags: list[str] = []

    patterns = {
        "⚠ Elevated values detected"   : re.compile(r"\b(high|elevated|increased|above normal|abnormal)\b", re.IGNORECASE),
        "⚠ Low values detected"         : re.compile(r"\b(low|decreased|reduced|below normal|deficient)\b", re.IGNORECASE),
        "⚠ Critical values mentioned"   : re.compile(r"\b(critical|urgent|emergency|immediate|severe)\b", re.IGNORECASE),
        "ℹ Medication mentioned"        : re.compile(r"\b(\d+\s*mg|\d+\s*mcg|tablet|capsule|injection)\b", re.IGNORECASE),
        "ℹ Follow-up recommended"       : re.compile(r"\b(follow.?up|revisit|return in)\b", re.IGNORECASE),
    }

    for label, pattern in patterns.items():
        if pattern.search(text):
            flags.append(label)

    return flags
