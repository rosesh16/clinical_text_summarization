import json
import argparse
from pathlib import Path

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_data(split: str):
    """
    Load PubMed data for the given split.
    split: train | val | test
    """
    base_dir = Path(__file__).resolve().parents[2]
    print("BASE_DIR =", base_dir)
    data_path = base_dir / "data" / "processed" / "pubmed" / f"pubmed_{split}.json"
    print("Trying to load:", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def lead3_summary(document):
    """
    Generate Lead-3 summary for a single document.
    """
    sentences = []

    for chunk in document["chunks"]:
        sentences.extend(chunk["sentences"])

    # Take first 3 sentences
    summary_sentences = sentences[:3]

    return " ".join(summary_sentences)


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main(args):
    data = load_data(args.split)

    predictions = []

    for doc in data:
        summary = lead3_summary(doc)

        predictions.append({
            "doc_id": doc["doc_id"],
            "summary": summary
        })

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Lead-3 summaries generated: {len(predictions)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lead-3 baseline for PubMed summarization")

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        required=True,
        help="Dataset split to run Lead-3 on"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save generated summaries (JSON)"
    )

    args = parser.parse_args()
    main(args)