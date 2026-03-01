"""
ModelLoader – loads TinyLlama with optional LoRA adapters.
"""
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_BASE, LORA_WEIGHTS

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._load()

    # ── Internal ───────────────────────────────────────────────────────────────
    def _load(self):
        logger.info("Loading tokenizer from %s", MODEL_BASE)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_BASE, use_fast=True
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("Loading base model …")
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )

            # Apply LoRA adapters if available
            if os.path.isdir(LORA_WEIGHTS):
                logger.info("Applying LoRA adapters from %s", LORA_WEIGHTS)
                self._model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
            else:
                logger.warning(
                    "LoRA weights not found at %s – using base model.", LORA_WEIGHTS
                )
                self._model = base_model

            self._model.eval()
            self._loaded = True
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    # ── Public API ─────────────────────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
