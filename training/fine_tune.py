"""
Fine-tuning Script – TinyLlama + LoRA on medicine dataset.
Optimised for 6 GB VRAM via 4-bit quantisation + small batch.

Usage (from pharma-llm root):
    python training/fine_tune.py \
        --train-file data/processed/medicines_train.jsonl \
        --val-file   data/processed/medicines_train_val.jsonl
"""
import argparse
import logging
import os
import sys

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_lora import get_lora_config, TRAIN_CONFIG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from config import MODEL_BASE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 256   # Halved to reduce activation memory on 6 GB GPU

# 4-bit quantisation config – cuts model from ~2.2 GB → ~0.7 GB VRAM
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # nested quantisation saves ~0.4 GB extra
)


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def main(train_file: str, val_file: str):
    logger.info("Loading tokenizer: %s", MODEL_BASE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", MODEL_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        quantization_config=BNB_CONFIG if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset …")
    data_files = {"train": train_file, "validation": val_file}
    raw_datasets = load_dataset("json", data_files=data_files)
    tokenized = raw_datasets.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(**{
        **TRAIN_CONFIG,
        "dataloader_num_workers": 0,  # Windows-safe
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    logger.info("Starting training …")
    trainer.train()

    output_dir = TRAIN_CONFIG["output_dir"]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuned model saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file",   required=True)
    args = parser.parse_args()
    main(args.train_file, args.val_file)
