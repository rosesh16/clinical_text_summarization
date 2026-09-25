"""Read the existing processed ccdv/pubmed-summarization JSON splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping


class PubMedJsonAdapter:
    """Adapter for JSON arrays with ``article`` and ``abstract`` fields.

    This adapter is deliberately read-only: it never changes the baseline
    processed PubMed files from which CHASM derives its JSONL files.
    """

    def __init__(self, article_field: str = "article", abstract_field: str = "abstract"):
        self.article_field = article_field
        self.abstract_field = abstract_field

    def iter_records(self, path: Path) -> Iterator[Mapping[str, Any]]:
        """Yield records from a processed split, validating its expected shape."""
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {path}, found {type(records).__name__}.")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at index {index} in {path}, found {type(record).__name__}."
                )
            yield record

    def get_document(self, record: Mapping[str, Any]) -> str | None:
        """Return the original article text without transforming it."""
        value = record.get(self.article_field)
        return value if isinstance(value, str) else None

    def get_summary(self, record: Mapping[str, Any]) -> str | None:
        """Return the original reference abstract without transforming it."""
        value = record.get(self.abstract_field)
        return value if isinstance(value, str) else None
