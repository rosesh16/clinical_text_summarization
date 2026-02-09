import json
import random
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

random.seed(SEED)

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "pubmed" / "pubmed_chunks.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pubmed"

# -------------------------
# Load data
# -------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total documents found: {len(data)}")

# -------------------------
# Shuffle documents
# -------------------------
random.shuffle(data)

# -------------------------
# Split
# -------------------------
n = len(data)
train_end = int(TRAIN_RATIO * n)
val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# -------------------------
# Save splits
# -------------------------
with open(OUTPUT_DIR / "pubmed_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open(OUTPUT_DIR / "pubmed_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

with open(OUTPUT_DIR / "pubmed_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

print("Split completed:")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")


# quick_check_splits.py
train_ids = set(d["doc_id"] for d in json.load(open("data/processed/pubmed/pubmed_train.json")))
val_ids   = set(d["doc_id"] for d in json.load(open("data/processed/pubmed/pubmed_val.json")))
test_ids  = set(d["doc_id"] for d in json.load(open("data/processed/pubmed/pubmed_test.json")))

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("No leakage detected.")