"""
Evaluation Script – compute accuracy, precision, recall, F1 on test set.

Usage:
    python training/evaluate.py --test-file data/processed/test_set.jsonl
"""
import argparse
import json
import logging
import os
import sys

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from config import MODEL_BASE, LORA_WEIGHTS, MAX_NEW_TOKENS, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    if os.path.isdir(LORA_WEIGHTS):
        model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
    else:
        logger.warning("LoRA weights not found – evaluating base model.")
        model = base_model
    model.eval()
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
    )


def keyword_label(text: str, expected: str) -> int:
    """Binary: 1 if expected keyword is present in generated text."""
    return int(expected.lower() in text.lower())


def evaluate(test_file: str):
    records = load_test_data(test_file)
    logger.info("Evaluating on %d samples …", len(records))

    gen = build_pipeline()
    y_true, y_pred = [], []

    for rec in records:
        prompt   = rec.get("prompt", rec.get("text", ""))
        expected = rec.get("expected", "")
        output   = gen(prompt, return_full_text=False)[0]["generated_text"]
        y_true.append(1)
        y_pred.append(keyword_label(output, expected))

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("\nFull report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    args = parser.parse_args()
    evaluate(args.test_file)
