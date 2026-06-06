"""Perplexity scorer using DeepSeek API logprobs.

Computes average negative log-likelihood from generated token logprobs
as a proxy for text fluency. Lower values = more fluent.

Replaces the previous GPT-2 local model approach, which had severe issues:
- Vocabulary misalignment with Chinese text (byte-level BPE)
- 1024 token context limit truncating longer chains
- Capability gap between GPT-2 (2019) and modern LLMs
"""

import math
import os
from typing import Optional

from openai import OpenAI


def compute_ppl(
    text: str,
    api_key: str = "",
    base_url: str = "https://ai.centos.hk/v1",
    model: str = "deepseek-v4-flash",
) -> Optional[float]:
    """Compute perplexity proxy via DeepSeek API logprobs.

    Sends the text as a prompt and asks the model to briefly summarize it.
    The generated tokens' logprobs reflect how confidently the model
    processes the input — a meaningful fluency signal.

    Args:
        text: Text to evaluate (typically a reasoning chain).
        api_key: DeepSeek API key. Falls back to AI_CENTOS_API_KEY env var.
        base_url: API base URL.
        model: Model to use (default: deepseek-v4-flash for cost efficiency).

    Returns:
        Perplexity score (lower = more fluent), or None if API call fails.
    """
    if not text.strip():
        return float("inf")

    api_key = api_key or os.environ.get("AI_CENTOS_API_KEY", "")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Read the following reasoning chain and provide a brief "
                        "(1-2 sentence) summary of its main conclusion.\n\n"
                        f"{text}"
                    ),
                }
            ],
            max_tokens=150,
            temperature=0.0,
            logprobs=True,
        )

        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            return None

        logprobs = [tok.logprob for tok in choice.logprobs.content]
        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)

    except Exception:
        return None
