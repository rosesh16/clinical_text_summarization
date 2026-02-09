import sys
import json
import argparse
from pathlib import Path

# --------------------------------------------------
# Force Python to see this directory
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from rouge_1 import compute_rouge_l
from bertscore import compute_bertscore
from redundancy import compute_redundancy

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_references(split):
    """
    TEMPORARY reference:
    first 5 sentences of document
    (used only to validate metric pipeline)
    """
    base_dir = Path.cwd()
    data_path = base_dir / "data" / "processed" / "pubmed" / f"pubmed_{split}.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs = []
    for doc in data:
        sentences = []
        for chunk in doc["chunks"]:
            sentences.extend(chunk["sentences"])
        refs.append(" ".join(sentences[:5]))

    return refs


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):
    preds_data = load_predictions(args.predictions)
    preds = [d["summary"] for d in preds_data]

    refs = load_references(args.split)

    rouge = compute_rouge_l(preds, refs)
    bert  = compute_bertscore(preds, refs)
    red   = sum(compute_redundancy(p) for p in preds) / len(preds)

    print("\nEvaluation Results")
    print("------------------")
    print(f"ROUGE-L     : {rouge:.4f}")
    print(f"BERTScore   : {bert:.4f}")
    print(f"Redundancy  : {red:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)

    args = parser.parse_args()
    main(args)