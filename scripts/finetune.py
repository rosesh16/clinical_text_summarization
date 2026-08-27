"""
scripts/finetune.py
-------------------
Step 5 of the fine-tuning pipeline.

Fine-tunes `Falconsai/medical_summarization` (T5-based) on the
processed PubMed summarization dataset prepared by prepare_dataset.py.

Saves the best checkpoint to:
    models/finetuned-medical-t5/

Run from the project root:
    python scripts/finetune.py

Expected runtime on a single NVIDIA GPU:
    ~30-60 min for 20K training samples x 3 epochs (fp16 enabled)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


# ── GPU check (fail fast with helpful message) ────────────────────────────────
def _check_cuda() -> None:
    """Abort early with actionable instructions if CUDA is not available."""
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU detected : {gpu}  ({vram:.1f} GB VRAM)")
        return

    print(
        "\n[ERROR] torch.cuda.is_available() returned False.\n"
        "Your PyTorch was installed WITHOUT CUDA support (CPU-only build).\n\n"
        "Fix — uninstall and reinstall PyTorch with CUDA:\n"
        "  (CUDA 12.1)  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  (CUDA 11.8)  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
        "  (CUDA 12.4)  pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
        "Run `nvidia-smi` to check your CUDA version, then pick the right URL above.\n"
        "After reinstalling, rerun this script.\n"
    )
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "Falconsai/medical_summarization"
OUTPUT_DIR      = Path("models/finetuned-medical-t5")
DATA_DIR        = Path("data/processed/pubmed_hf")

MAX_INPUT_LEN   = 512    # T5 encoder max tokens
MAX_TARGET_LEN  = 128    # decoder max tokens (abstract length)

TRAIN_BATCH     = 8      # per-device batch size  (reduce to 4 if OOM)
GRAD_ACCUM      = 4      # effective batch = TRAIN_BATCH * GRAD_ACCUM = 32
EVAL_BATCH      = 16
LEARNING_RATE   = 5e-5
EPOCHS          = 3
WARMUP_STEPS    = 200
WEIGHT_DECAY    = 0.01
SAVE_TOTAL      = 2      # keep only the 2 best checkpoints

USE_FP16        = torch.cuda.is_available()   # auto-detect GPU


# ── Load data ─────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── HuggingFace Dataset wrapper ───────────────────────────────────────────────

class PubMedDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset that tokenizes on-the-fly."""

    def __init__(self, records: list[dict], tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]

        # Prefix required by T5 models
        input_text  = "summarize: " + row["article"]
        target_text = row["abstract"]

        model_inputs = self.tokenizer(
            input_text,
            max_length    = MAX_INPUT_LEN,
            padding       = "max_length",
            truncation    = True,
            return_tensors= "pt",
        )
        labels = self.tokenizer(
            target_text,
            max_length    = MAX_TARGET_LEN,
            padding       = "max_length",
            truncation    = True,
            return_tensors= "pt",
        )

        # Replace padding token id in labels with -100 (ignored by loss)
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"      : model_inputs["input_ids"].squeeze(),
            "attention_mask" : model_inputs["attention_mask"].squeeze(),
            "labels"         : label_ids,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    print("=" * 60)
    print("Step 5 -- Fine-tuning Falconsai/medical_summarization")
    print("=" * 60)
    _check_cuda()   # abort immediately if no CUDA-enabled PyTorch
    print(f"  Device      : {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  FP16        : {USE_FP16}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {TRAIN_BATCH} x {GRAD_ACCUM} accum = {TRAIN_BATCH * GRAD_ACCUM} effective")
    print()

    # -- Load tokenizer and model ---------------------------------------------
    print("[1/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    print(f"      Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- Load datasets --------------------------------------------------------
    print("[2/5] Loading processed datasets...")
    train_data = load_json(DATA_DIR / "train.json")
    val_data   = load_json(DATA_DIR / "val.json")
    print(f"      Train: {len(train_data):,} | Val: {len(val_data):,}")

    train_ds = PubMedDataset(train_data, tokenizer)
    val_ds   = PubMedDataset(val_data,   tokenizer)

    # -- Data collator --------------------------------------------------------
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model             = model,
        label_pad_token_id= -100,
        pad_to_multiple_of= 8 if USE_FP16 else None,
    )

    # -- Training arguments ---------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir              = str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH,
        per_device_eval_batch_size  = EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate           = LEARNING_RATE,
        warmup_steps            = WARMUP_STEPS,
        weight_decay            = WEIGHT_DECAY,
        fp16                    = USE_FP16,
        predict_with_generate   = True,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        save_total_limit        = SAVE_TOTAL,
        logging_dir             = str(OUTPUT_DIR / "logs"),
        logging_steps           = 100,
        report_to               = "none",   # disable wandb/tensorboard
        dataloader_num_workers  = 0,        # safer on Windows
    )

    # -- Trainer --------------------------------------------------------------
    print("[3/5] Setting up Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        tokenizer     = tokenizer,
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # -- Train ----------------------------------------------------------------
    print("[4/5] Starting training...")
    print("      (Loss will be printed every 100 steps)")
    print()
    trainer.train()

    # -- Save final model to clean output path --------------------------------
    print("\n[5/5] Saving fine-tuned model...")
    final_path = OUTPUT_DIR / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"      Model saved -> {final_path.resolve()}")

    print("\nFine-tuning complete!")
    print(f"  Load with: AutoModelForSeq2SeqLM.from_pretrained('{final_path}')")
    print(f"  Or update PRIMARY_MODEL in hf_summarizer.py to: '{final_path}'")


if __name__ == "__main__":
    main()
