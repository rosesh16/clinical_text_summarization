"""Statistics and leakage checks for CHASM Phase 1 output."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass
class SplitStatistics:
    """Incrementally collected statistics for one output split."""

    input_records: int = 0
    output_records: int = 0
    skipped_empty_documents: int = 0
    skipped_empty_summaries: int = 0
    duplicate_documents_removed: int = 0
    document_word_counts: list[int] = field(default_factory=list)
    summary_word_counts: list[int] = field(default_factory=list)
    compression_ratios: list[float] = field(default_factory=list)

    @staticmethod
    def _distribution(values: list[int] | list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {"min": min(values), "max": max(values), "mean": mean(values)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "skipped_empty_documents": self.skipped_empty_documents,
            "skipped_empty_summaries": self.skipped_empty_summaries,
            "duplicate_documents_removed": self.duplicate_documents_removed,
            "document_word_count": self._distribution(self.document_word_counts),
            "summary_word_count": self._distribution(self.summary_word_counts),
            "compression_ratio": self._distribution(self.compression_ratios),
        }


class DatasetInspector:
    """Build reports and verify no SHA-256 document-ID leakage across splits."""

    def __init__(self, split_names: Iterable[str]):
        self.statistics = {name: SplitStatistics() for name in split_names}
        self.document_ids = {name: set() for name in split_names}

    def add_record(
        self,
        split: str,
        document_id: str,
        document_word_count: int,
        summary_word_count: int,
        compression_ratio: float,
    ) -> None:
        stats = self.statistics[split]
        stats.output_records += 1
        stats.document_word_counts.append(document_word_count)
        stats.summary_word_counts.append(summary_word_count)
        stats.compression_ratios.append(compression_ratio)
        self.document_ids[split].add(document_id)

    def overlap_counts(self) -> dict[str, int]:
        names = list(self.document_ids)
        return {
            f"{left}_{right}": len(self.document_ids[left] & self.document_ids[right])
            for position, left in enumerate(names)
            for right in names[position + 1 :]
        }

    def assert_no_document_overlap(self) -> dict[str, int]:
        overlaps = self.overlap_counts()
        leaked = {pair: count for pair, count in overlaps.items() if count}
        if leaked:
            raise ValueError(f"Document overlap detected between output splits: {leaked}")
        return overlaps

    def report(
        self,
        source_dataset: str,
        input_files: Mapping[str, str],
        processing: Mapping[str, Any],
    ) -> dict[str, Any]:
        split_report = {name: stats.as_dict() for name, stats in self.statistics.items()}
        return {
            "framework": "CHASM",
            "dataset_description": "PubMed biomedical research-paper summarization dataset",
            "source_dataset": source_dataset,
            "input_files": dict(input_files),
            "processing": dict(processing),
            "splits": split_report,
            "cross_split_document_id_overlap": self.assert_no_document_overlap(),
            "totals": {
                "input_records": sum(item["input_records"] for item in split_report.values()),
                "output_records": sum(item["output_records"] for item in split_report.values()),
                "skipped_empty_documents": sum(
                    item["skipped_empty_documents"] for item in split_report.values()
                ),
                "skipped_empty_summaries": sum(
                    item["skipped_empty_summaries"] for item in split_report.values()
                ),
                "duplicate_documents_removed": sum(
                    item["duplicate_documents_removed"] for item in split_report.values()
                ),
            },
        }
