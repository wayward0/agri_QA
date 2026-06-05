"""LLM call abstraction.

The ONLY place that creates an OpenAI client.
Returns a callable matching the LLMCallFn protocol.
Supports per-call model override for multi-tier cost optimization.
"""

import time
from openai import OpenAI


def create_llm_caller(
    api_key: str,
    base_url: str = "https://ai.centos.hk/v1",
    model: str = "deepseek-v4-pro",
    max_tokens: int = 4096,
):
    """Create an LLM call function with a default model.

    Returns a callable with signature:
        call(prompt, system="", temperature=0.2, max_tokens=4096, model=None) -> str

    The `model` parameter allows per-call override for cost optimization.
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    def call(
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = max_tokens,
        model: str = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model or call._default_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    return ""
                return content.strip()
            except Exception as e:
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise

    call._default_model = model
    return call


def create_stage_caller(base_caller, stage_model: str):
    """Create a stage-specific caller that binds a default model.

    Args:
        base_caller: The base LLM call function from create_llm_caller.
        stage_model: Model name to use for this stage.

    Returns:
        A callable with the same signature, but defaulting to stage_model.
    """
    def stage_call(
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: str = None,
    ) -> str:
        return base_caller(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model or stage_model,
        )
    return stage_call
