"""
scripts/evaluate_finetuned.py
-----------------------------
Step 7 of the fine-tuning pipeline.

Evaluates and compares two models on the PubMed test set:
  1. Baseline  : Falconsai/medical_summarization  (original)
  2. Fine-tuned: models/finetuned-medical-t5/final (ours)

Metrics computed:
  - ROUGE-1, ROUGE-2, ROUGE-L  (lexical overlap)
  - BERTScore F1                (semantic similarity)

Run from the project root:
    python scripts/evaluate_finetuned.py

Optional flags:
    --samples 200          # evaluate on first N test samples (default: all)
    --skip-baseline        # skip baseline model (saves time)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# ── Config ────────────────────────────────────────────────────────────────────
BASELINE_MODEL   = "Falconsai/medical_summarization"
FINETUNED_MODEL  = str(Path("models/finetuned-medical-t5/final"))
TEST_DATA_PATH   = Path("data/processed/pubmed_hf/test.json")

MAX_INPUT_LEN    = 512
SUMMARY_MAX_LEN  = 128
SUMMARY_MIN_LEN  = 30
BATCH_SIZE       = 8


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_summaries(model_path: str, articles: list[str]) -> list[str]:
    """Run batch inference and return list of generated summaries."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"  Loading: {model_path}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    summaries = []
    for i in range(0, len(articles), BATCH_SIZE):
        batch_texts = ["summarize: " + a for a in articles[i:i + BATCH_SIZE]]
        inputs = tokenizer(
            batch_texts,
            max_length    = MAX_INPUT_LEN,
            truncation    = True,
            padding       = True,
            return_tensors= "pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = SUMMARY_MAX_LEN,
                min_length     = SUMMARY_MIN_LEN,
                num_beams      = 4,
                early_stopping = True,
                no_repeat_ngram_size = 3,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(decoded)

        done = min(i + BATCH_SIZE, len(articles))
        print(f"    [{done}/{len(articles)}] samples processed...", end="\r")

    print()

    # Free GPU memory before loading next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summaries


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals  = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
    n = len(predictions)
    return {k: v / n for k, v in totals.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    from bert_score import score as bert_score
    _, _, F1 = bert_score(
        predictions, references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return F1.mean().item()


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_results_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Model':<35} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERTScore':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, m in results.items():
        tag = " <-- BETTER" if name != "Baseline (Falconsai)" else ""
        print(
            f"{name:<35} "
            f"{m['rouge1']:>8.4f} "
            f"{m['rouge2']:>8.4f} "
            f"{m['rougeL']:>8.4f} "
            f"{m['bertscore']:>10.4f}"
            f"{tag}"
        )
    print(sep + "\n")


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Step 7/8 -- Evaluating models on PubMed test set")
    print("=" * 60)

    # Load test data
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.samples:
        test_data = test_data[:args.samples]

    articles   = [d["article"]  for d in test_data]
    references = [d["abstract"] for d in test_data]
    print(f"  Evaluating on {len(articles):,} test samples\n")

    all_results: dict[str, dict] = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    if not args.skip_baseline:
        print("[Baseline] Falconsai/medical_summarization")
        base_preds  = generate_summaries(BASELINE_MODEL, articles)
        base_rouge  = compute_rouge(base_preds, references)
        print("  Computing BERTScore (this takes ~1-2 min)...")
        base_bert   = compute_bertscore(base_preds, references)
        all_results["Baseline (Falconsai)"] = {**base_rouge, "bertscore": base_bert}
        print(f"  ROUGE-L: {base_rouge['rougeL']:.4f}  BERTScore: {base_bert:.4f}\n")

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    ft_path = Path(FINETUNED_MODEL)
    if not ft_path.exists():
        print(f"[WARN] Fine-tuned model not found at {ft_path}.")
        print("       Run `python scripts/finetune.py` first.")
    else:
        print("[Fine-tuned] Our model")
        ft_preds  = generate_summaries(FINETUNED_MODEL, articles)
        ft_rouge  = compute_rouge(ft_preds, references)
        print("  Computing BERTScore...")
        ft_bert   = compute_bertscore(ft_preds, references)
        all_results["Fine-tuned (ours)"] = {**ft_rouge, "bertscore": ft_bert}
        print(f"  ROUGE-L: {ft_rouge['rougeL']:.4f}  BERTScore: {ft_bert:.4f}\n")

    # ── Print & save ──────────────────────────────────────────────────────────
    print_results_table(all_results)
    save_results(
        all_results,
        Path("experiments/baseline_results/finetuned_vs_baseline.json")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned model")
    parser.add_argument("--samples",       type=int,  default=None,  help="Number of test samples to evaluate")
    parser.add_argument("--skip-baseline", action="store_true",      help="Skip baseline model evaluation")
    args = parser.parse_args()
    main(args)
