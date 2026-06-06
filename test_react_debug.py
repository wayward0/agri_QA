#!/usr/bin/env python3
"""Test ReAct generator with per-call timing."""
import json, sys, time
sys.path.insert(0, ".")

import config
from llm_client import create_llm_caller, create_stage_caller
from pipeline import load_rag_tool

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

log("Loading RAG...")
rag_tool = load_rag_tool()
log("RAG OK")

llm_call = create_llm_caller(api_key=config.API_KEY, base_url=config.API_BASE_URL, model=config.MODEL_PREMIUM)
log("Caller OK")

# Patch llm_call to add timing
original_call = llm_call
call_count = [0]
def timed_call(prompt, system="", temperature=0.2, max_tokens=2048, model=None):
    call_count[0] += 1
    n = call_count[0]
    prompt_preview = prompt[:100].replace('\n', ' ')
    log(f"  LLM call #{n} start (prompt={prompt_preview}...)")
    t1 = time.time()
    result = original_call(prompt, system=system, temperature=temperature, max_tokens=max_tokens, model=model)
    log(f"  LLM call #{n} done in {time.time()-t1:.1f}s, response={len(result)} chars")
    return result

with open(config.PATH_SAMPLE, "r") as f:
    items = json.load(f)[:1]
item = items[0]
log(f"Question: {item['Question'][:80]}")

# Test single ReAct chain
from thinker.react_generator import generate_react_chain

log("Starting generate_react_chain (single path, temp=0.3)...")
chain = generate_react_chain(
    item["Question"], item["Answer"], rag_tool, timed_call,
    max_rounds=3, temperature=0.3,
)
log(f"Done! {len(chain.steps)} steps, react_rounds={chain.react_rounds}")

# Test RAG retrieval speed
log("Testing RAG retrieval...")
t1 = time.time()
results = rag_tool.retrieve("phosphorus fertilizer bell pepper", intent="background", top_k=3)
log(f"RAG retrieval: {time.time()-t1:.1f}s, {len(results)} results")

log(f"Total: {time.time()-t0:.1f}s")
