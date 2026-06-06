#!/usr/bin/env python3
"""Debug pipeline test with timing per stage."""
import json, sys, time
sys.path.insert(0, ".")

import config
from llm_client import create_llm_caller, create_stage_caller
from pipeline import load_rag_tool
from models import DifficultyLevel

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

log("Loading RAG...")
rag_tool = load_rag_tool()
log("RAG OK")

llm_call = create_llm_caller(api_key=config.API_KEY, base_url=config.API_BASE_URL, model=config.MODEL_STANDARD)
classifier_call = create_stage_caller(llm_call, config.MODEL_LIGHT)
thinker_call = create_stage_caller(llm_call, config.MODEL_PREMIUM)
reviewer_call = create_stage_caller(llm_call, config.MODEL_STANDARD)
reviser_call = create_stage_caller(llm_call, config.MODEL_LIGHT)
evaluator_call = create_stage_caller(llm_call, config.MODEL_STANDARD)
log("Callers OK")

# Test LLM
resp = llm_call("Say hello.", system="Be brief.", temperature=0.0, max_tokens=2048)
log(f"LLM test: '{resp[:50]}'")

with open(config.PATH_SAMPLE, "r") as f:
    items = json.load(f)[:1]

item = items[0]
log(f"Question: {item['Question'][:80]}")

# Manual pipeline with timing
from classifier.difficulty_classifier import classify_difficulty
from thinker.self_consistency import generate_with_consistency
from reviewer.integrator import run_review
from reviser.reviser import execute_revision
from evaluator.auto_metrics import compute_auto_metrics
from evaluator.llm_judge import judge_faithfulness, judge_logical_completeness
from evaluator.ppl_scorer import compute_ppl

log("Stage 0: Classification...")
classification = classify_difficulty(item["Question"], item["Answer"], classifier_call)
difficulty = classification.difficulty
log(f"  -> {difficulty.value}")

n_samples = config.SELF_CONSISTENCY_N if difficulty == DifficultyLevel.HARD else 1
log(f"Stage 1: Thinker (n_samples={n_samples})...")
draft_chain = generate_with_consistency(
    item["Question"], item["Answer"], rag_tool, thinker_call,
    n_samples=n_samples, temperatures=config.REACT_TEMPERATURES,
)
log(f"  -> {len(draft_chain.steps)} steps, react_rounds={draft_chain.react_rounds}")

log("Stage 2: Reviewer...")
unified_actions, critiques = run_review(
    draft_chain, item["Question"], difficulty, rag_tool, reviewer_call,
    answer=item["Answer"],
)
log(f"  -> {len(unified_actions.priority_actions)} priority actions, {len(critiques)} critiques")

log("Stage 3: Reviser...")
if difficulty != DifficultyLevel.EASY:
    revised_chain = execute_revision(draft_chain, unified_actions, reviser_call, rag_tool)
else:
    revised_chain = draft_chain
log(f"  -> {len(revised_chain.steps)} steps")

log("Stage 4: Evaluator...")
faith, faith_notes = judge_faithfulness(revised_chain, evaluator_call)
logic, logic_notes = judge_logical_completeness(revised_chain, evaluator_call)
auto = compute_auto_metrics(revised_chain)
w = config.EVAL_WEIGHTS
overall = (
    faith / 5 * w["faithfulness"]
    + auto["structure"] * w["structure"]
    + auto["information_density"] * w["information_density"]
    + logic / 5 * w["logical_completeness"]
    + auto["traceability"] * w["traceability"]
    + auto["step_order"] * w["step_order"]
)
log(f"  -> faith={faith} logic={logic} struct={auto['structure']} order={auto['step_order']} overall={overall:.3f}")

log("PPL...")
ppl = compute_ppl(revised_chain.to_text(), api_key=config.API_KEY, base_url=config.API_BASE_URL, model=config.MODEL_LIGHT)
log(f"  -> ppl={ppl}")

log(f"DONE in {time.time()-t0:.1f}s")
