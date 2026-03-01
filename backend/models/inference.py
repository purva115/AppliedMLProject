"""
Inference helpers – generate responses from the loaded model.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY


def build_prompt(user_message: str) -> str:
    """Wraps the user message in TinyLlama chat template."""
    return (
        "<|system|>\n"
        "You are PharmaLLM, a knowledgeable pharmacy assistant. "
        "Provide accurate, helpful information about medicines including their uses, "
        "composition, dosage, and side effects. Keep your responses concise and "
        "limit them to a maximum of 50 words. Always remind users to consult a "
        "healthcare professional for medical advice.\n"
        f"<|user|>\n{user_message}\n"
        "<|assistant|>\n"
    )


def generate_response(model_loader, query: str) -> str:
    """Generates a text response for the given query."""
    if not model_loader.is_loaded():
        return "Model not loaded."

    tokenizer = model_loader.tokenizer
    model = model_loader.model

    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
