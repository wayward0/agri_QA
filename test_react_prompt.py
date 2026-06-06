#!/usr/bin/env python3
"""Test actual ReAct prompt with DeepSeek to diagnose empty responses."""
import json, os, sys, time
sys.path.insert(0, ".")
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("AI_CENTOS_API_KEY", ""),
    base_url="https://ai.centos.hk/v1",
    timeout=180.0,
)

SYSTEM = """You are an agricultural reasoning expert using the ReAct framework.

Given a question and its answer, generate a structured reasoning chain that explains
how to arrive at the answer. Use the Thought-Action-Observation loop:

1. THOUGHT: Analyze what you know and what you need to find out.
2. ACTION: Either:
   - retrieve: <search query>  (to get more evidence)
   - FINISH  (when you have enough information to build the reasoning chain)

After each ACTION, you will receive an OBSERVATION with retrieved evidence.

When you choose FINISH, output a JSON reasoning chain:
```json
{"steps": [
  {"step": 1, "type": "context_setup", "content": "...", "evidence": "...", "confidence": "high"},
  {"step": 2, "type": "knowledge_application", "content": "...", "evidence": "...", "confidence": "medium"},
  ...
  {"step": N, "type": "conclusion", "content": "...", "confidence": "high"}
]}
```

Step types: context_setup, knowledge_application, causal_reasoning, comparison,
condition_analysis, evidence_integration, conclusion.

RULES:
- Every factual claim MUST cite evidence or be logically derived from cited steps
- If unsupported, set confidence to "low"
- Aim for 3-7 reasoning steps
- The final conclusion must connect back to the given answer"""

USER = """Question: How do I apply phosphorus fertilizer to save my bell peppers on a small farm in Massachusetts?
Answer: Apply phosphorus fertilizer by side-dressing each plant with 1/2 cup of 0-20-0 triple superphosphate when fruiting begins. Also ensure soil pH is 6.0-6.5 for optimal uptake.

What is your next Thought and Action?"""

print("Testing ReAct prompt with deepseek-v4-pro...")
for mt in [2048, 4096, 8192]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        temperature=0.3,
        max_tokens=mt,
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""
    fr = resp.choices[0].finish_reason
    usage = resp.usage
    print(f"\nmax_tokens={mt}: {elapsed:.1f}s, finish={fr}")
    print(f"  prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}")
    print(f"  content_len={len(content)}")
    if content:
        print(f"  preview: {content[:300]}")
    else:
        print(f"  EMPTY!")
