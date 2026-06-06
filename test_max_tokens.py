#!/usr/bin/env python3
"""Test DeepSeek max_tokens limits."""
import os, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("AI_CENTOS_API_KEY", ""),
    base_url="https://ai.centos.hk/v1",
    timeout=180.0,
)

prompt = """You are an agricultural reasoning expert. Given a question and answer, generate a structured reasoning chain.

Question: How do I apply phosphorus fertilizer to save my bell peppers on a small farm in Massachusetts?
Answer: Apply phosphorus fertilizer by side-dressing each plant with 1/2 cup of 0-20-0 triple superphosphate when fruiting begins. Also ensure soil pH is 6.0-6.5 for optimal uptake.

Generate a JSON reasoning chain with 5-7 steps covering soil analysis, deficiency diagnosis, application method, timing, and precautions."""

for mt in [4096, 8192, 16384]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=mt,
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content or ""
        fr = resp.choices[0].finish_reason
        usage = resp.usage
        print(f"max_tokens={mt}: {elapsed:.1f}s, finish={fr}, content_len={len(content)}, completion_tokens={usage.completion_tokens}")
        if len(content) > 0:
            print(f"  preview: {content[:200]}")
        else:
            print(f"  EMPTY CONTENT")
    except Exception as e:
        print(f"max_tokens={mt}: ERROR: {e}")
    print()
