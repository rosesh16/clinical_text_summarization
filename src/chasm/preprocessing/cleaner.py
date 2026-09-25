"""Conservative text cleaning for biomedical research-paper summarization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# Matches common web URLs while leaving biomedical punctuation and identifiers alone.
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def sha256_text(text: str) -> str:
    """Create a deterministic SHA-256 identifier from the supplied original text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BiomedicalTextCleaner:
    """Apply only loss-minimising transformations required by CHASM Phase 1.

    No case conversion, stemming, lemmatisation, citation stripping, or medical
    term filtering is performed. This preserves abbreviations, negations, units,
    dosages, laboratory values, numbers, and section headings.
    """

    remove_urls: bool = True
    remove_non_printable: bool = True
    normalize_whitespace: bool = True

    def clean(self, text: str | None) -> str:
        """Return cleaned text, or an empty string for a null/non-string value."""
        if not isinstance(text, str):
            return ""

        cleaned = text
        if self.remove_urls:
            cleaned = URL_PATTERN.sub(" ", cleaned)
        if self.remove_non_printable:
            # Newlines and tabs are whitespace, so retain them until whitespace
            # normalisation can turn them into separators rather than deleting words.
            cleaned = "".join(
                character
                for character in cleaned
                if character.isprintable() or character.isspace()
            )
        if self.normalize_whitespace:
            cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
        return cleaned.strip()

    @staticmethod
    def word_count(text: str) -> int:
        """Count whitespace-delimited tokens without changing the text."""
        return len(text.split())
