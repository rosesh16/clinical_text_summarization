"""Build CHASM Phase 1 JSONL data from existing processed PubMed HF files.

The input is the already-downloaded ``ccdv/pubmed-summarization`` corpus with
``article`` as the source document and ``abstract`` as the reference summary.
This script is read-only with respect to the baseline dataset and writes only
under ``data/chasm/processed`` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chasm.preprocessing import (  # noqa: E402
    BiomedicalTextCleaner,
    DatasetInspector,
    PubMedJsonAdapter,
    sha256_text,
)

LOGGER = logging.getLogger("chasm.phase1")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate a CHASM YAML configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {config_path} must be a YAML mapping.")
    for section in ("dataset", "input", "output", "cleaning"):
        if section not in config:
            raise ValueError(f"Configuration is missing required '{section}' section.")
    return config


def resolve_project_path(value: str | Path) -> Path:
    """Resolve configured paths from the project root unless already absolute."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    """Render project-relative paths when possible, otherwise keep absolute paths."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_logging(level: str) -> None:
    """Configure concise console logging without changing project-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def process_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Process configured splits and return the calculated preprocessing report."""
    input_config = config["input"]
    output_config = config["output"]
    cleaning_config = config["cleaning"]
    split_files = input_config["splits"]
    if not isinstance(split_files, dict) or not split_files:
        raise ValueError("input.splits must be a non-empty mapping of split names to files.")

    input_directory = resolve_project_path(input_config["directory"])
    output_directory = resolve_project_path(output_config["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    cleaner = BiomedicalTextCleaner(
        remove_urls=bool(cleaning_config.get("remove_urls", True)),
        remove_non_printable=bool(cleaning_config.get("remove_non_printable", True)),
        normalize_whitespace=bool(cleaning_config.get("normalize_whitespace", True)),
    )
    adapter = PubMedJsonAdapter(
        article_field=input_config.get("article_field", "article"),
        abstract_field=input_config.get("abstract_field", "abstract"),
    )
    inspector = DatasetInspector(split_files.keys())
    seen_document_fingerprints: set[str] = set()
    input_files: dict[str, str] = {}

    for split, filename in split_files.items():
        input_path = input_directory / filename
        if not input_path.is_file():
            raise FileNotFoundError(f"Configured input split does not exist: {input_path}")
        input_files[split] = display_path(input_path)
        output_path = output_directory / f"{split}.jsonl"
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        stats = inspector.statistics[split]

        LOGGER.info("Processing %s from %s", split, input_path)
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_handle:
            for row in adapter.iter_records(input_path):
                stats.input_records += 1
                original_document = adapter.get_document(row)
                document_id = sha256_text(original_document) if original_document is not None else None
                document = cleaner.clean(original_document)
                summary = cleaner.clean(adapter.get_summary(row))

                if not document:
                    stats.skipped_empty_documents += 1
                    continue
                if not summary and cleaning_config.get("drop_empty_summaries", False):
                    stats.skipped_empty_summaries += 1
                    continue

                # Deduplicate cleaned source text globally so records cannot leak
                # from train into validation/test due to harmless raw formatting noise.
                fingerprint = sha256_text(document)
                if cleaning_config.get("deduplicate_documents", True) and fingerprint in seen_document_fingerprints:
                    stats.duplicate_documents_removed += 1
                    continue
                seen_document_fingerprints.add(fingerprint)

                document_word_count = cleaner.word_count(document)
                summary_word_count = cleaner.word_count(summary)
                compression_ratio = (
                    document_word_count / summary_word_count if summary_word_count else 0.0
                )
                record = {
                    "document_id": document_id,
                    "document": document,
                    "summary": summary,
                    "metadata": {
                        "source_dataset": config["dataset"]["name"],
                        "split": split,
                        "doc_word_count": document_word_count,
                        "summary_word_count": summary_word_count,
                        "compression_ratio": compression_ratio,
                    },
                }
                _write_jsonl_record(output_handle, record)
                inspector.add_record(
                    split,
                    document_id,
                    document_word_count,
                    summary_word_count,
                    compression_ratio,
                )

        temporary_path.replace(output_path)
        LOGGER.info(
            "Finished %s: %s input, %s output, %s duplicates removed",
            split,
            stats.input_records,
            stats.output_records,
            stats.duplicate_documents_removed,
        )

    report = inspector.report(
        source_dataset=config["dataset"]["name"],
        input_files=input_files,
        processing={
            "document_id": "SHA-256 of original article text encoded as UTF-8",
            "deduplication_basis": "SHA-256 of cleaned document text; global train-validation-test order",
            "cleaning": dict(cleaning_config),
        },
    )
    report_path = output_directory / output_config["report_filename"]
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Wrote preprocessing report to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CHASM Phase 1 biomedical research-paper JSONL splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "chasm.yaml",
        help="Path to the CHASM YAML configuration (default: configs/chasm.yaml).",
    )
    arguments = parser.parse_args()
    config_path = resolve_project_path(arguments.config)
    config = load_config(config_path)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    report = process_dataset(config)
    LOGGER.info(
        "CHASM Phase 1 complete: %s output records; %s cross-split overlaps.",
        report["totals"]["output_records"],
        sum(report["cross_split_document_id_overlap"].values()),
    )


if __name__ == "__main__":
    main()
