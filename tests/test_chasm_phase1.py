"""Focused tests for CHASM Phase 1 cleaning, deduplication, and leakage checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase1_preprocess import process_dataset
from src.chasm.preprocessing.cleaner import BiomedicalTextCleaner, sha256_text


class ChasmPhase1Tests(unittest.TestCase):
    def test_cleaner_preserves_medical_content_and_removes_url(self) -> None:
        cleaner = BiomedicalTextCleaner()
        text = "RESULTS:\tNo AKI; HbA1c 7.2% at 5 mg/day. https://example.org\x00"
        self.assertEqual(
            cleaner.clean(text), "RESULTS: No AKI; HbA1c 7.2% at 5 mg/day."
        )

    def test_processing_removes_duplicates_and_validates_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            train_article = "METHODS: No sepsis. Dose 5 mg/day. https://example.org"
            duplicate_article = "METHODS: No sepsis. Dose 5 mg/day. https://other.example"
            files = {
                "train": [{"article": train_article, "abstract": "No sepsis occurred."}],
                "validation": [{"article": duplicate_article, "abstract": "Duplicate."}],
                "test": [{"article": None, "abstract": "Missing article."}],
            }
            filenames = {"train": "train.json", "validation": "val.json", "test": "test.json"}
            for split, rows in files.items():
                (input_dir / filenames[split]).write_text(json.dumps(rows), encoding="utf-8")

            config = {
                "dataset": {"name": "ccdv/pubmed-summarization"},
                "input": {
                    "directory": str(input_dir),
                    "article_field": "article",
                    "abstract_field": "abstract",
                    "splits": filenames,
                },
                "output": {"directory": str(output_dir), "report_filename": "report.json"},
                "cleaning": {
                    "remove_urls": True,
                    "remove_non_printable": True,
                    "normalize_whitespace": True,
                    "deduplicate_documents": True,
                    "drop_empty_summaries": False,
                },
            }
            report = process_dataset(config)

            records = [
                json.loads(line) for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["document_id"], sha256_text(train_article))
            self.assertEqual(records[0]["metadata"]["doc_word_count"], 6)
            self.assertEqual(report["splits"]["validation"]["duplicate_documents_removed"], 1)
            self.assertEqual(report["splits"]["test"]["skipped_empty_documents"], 1)
            self.assertTrue(all(value == 0 for value in report["cross_split_document_id_overlap"].values()))


if __name__ == "__main__":
    unittest.main()
