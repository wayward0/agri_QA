"""Perplexity scorer using a local language model.

Uses transformers library for local PPL computation.
No LLM API calls — runs locally.
"""

import math
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_model_cache = {}


def _load_model(model_name: str = "gpt2"):
    """Load and cache model + tokenizer."""
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


def compute_ppl(text: str, model_name: str = "gpt2", max_length: int = 512) -> float:
    """Compute perplexity of text using a local language model.

    Args:
        text: Text to evaluate.
        model_name: HuggingFace model name (default: gpt2).
        max_length: Maximum token length for evaluation.

    Returns:
        Perplexity score (lower = more fluent).
    """
    if not text.strip():
        return float("inf")

    model, tokenizer = _load_model(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())
