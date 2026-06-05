"""Pipeline orchestrator — the ONLY file that imports all modules.

Wires dependencies, runs stages, handles checkpointing and error recovery.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

import config
from llm_client import create_llm_caller, create_stage_caller
from rag.rag_tool import RAGTool
from rag.retriever import HybridRetriever
from rag.knowledge_builder import build_knowledge_base
from classifier.difficulty_classifier import classify_difficulty
from thinker.self_consistency import generate_with_consistency
from reviewer.integrator import run_review
from reviser.reviser import execute_revision
from evaluator.auto_metrics import compute_auto_metrics
from evaluator.llm_judge import judge_faithfulness, judge_logical_completeness
from models import (
    ClassificationResult,
    DifficultyLevel,
    PipelineItem,
    QualityScores,
    ReasoningChain,
)


def load_rag_tool() -> RAGTool:
    """Load pre-built indices and construct RAGTool."""
    import faiss
    import pickle
    import json as json_mod
    from rag.embedding_client import EmbeddingClient
    from rag.reranker import RerankerClient

    # Load FAISS index
    faiss_index = faiss.read_index(str(config.PATH_FAISS_INDEX))

    # Load BM25 model
    with open(config.PATH_BM25_INDEX, "rb") as f:
        bm25_model = pickle.load(f)

    # Load metadata
    with open(config.PATH_INDEX_METADATA, "r", encoding="utf-8") as f:
        metadata = json_mod.load(f)

    # Load embedding client (API-based)
    embedding_model = EmbeddingClient(
        base_url=config.EMBEDDING_API_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
    )

    # Load reranker (API-based)
    reranker = RerankerClient(
        base_url=config.RERANKER_API_URL,
        api_key=config.RERANKER_API_KEY,
        model=config.RERANKER_MODEL,
    )

    # Construct retriever and RAG tool
    retriever = HybridRetriever(
        faiss_index=faiss_index,
        bm25_model=bm25_model,
        metadata=metadata,
        embedding_model=embedding_model,
        rrf_k=config.RRF_K,
        reranker=reranker,
    )
    return RAGTool(retriever)


def subsample_agthoughts(n: int = 1000, seed: int = 42) -> List[dict]:
    """Stratified subsample from AgThoughts.json.

    Args:
        n: Number of items to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with Question, Answer, Question Type, Reasoning Traces.
    """
    import random

    random.seed(seed)

    with open(config.PATH_AGTHOUGHTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter items with valid reasoning traces
    valid = [item for item in data if item.get("Reasoning Traces")]
    print(f"Total valid items: {len(valid)}")

    # Stratified sample by Question Type
    from collections import defaultdict
    by_type = defaultdict(list)
    for item in valid:
        by_type[item.get("Question Type", "Unknown")].append(item)

    samples = []
    per_type = max(1, n // len(by_type))
    for qtype, items in by_type.items():
        count = min(per_type, len(items))
        samples.extend(random.sample(items, count))

    # Trim to n if over-sampled
    if len(samples) > n:
        samples = random.sample(samples, n)

    print(f"Sampled {len(samples)} items from {len(by_type)} question types.")
    return samples


def save_sample(items: List[dict], path: Path):
    """Save sampled items to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def process_item(
    item: dict,
    item_id: str,
    rag_tool: RAGTool,
    llm_call,          # base caller (unused directly, kept for compat)
    classifier_call,   # cheap: classification
    thinker_call,      # premium: core reasoning
    reviewer_call,     # standard: review & integration
    reviser_call,      # cheap: action execution
    evaluator_call,    # standard: quality judgment
) -> PipelineItem:
    """Run one item through the full pipeline with difficulty-based routing.

    Each stage uses a model-tuned caller for cost optimization:
    - classifier_call: cheap model for simple classification
    - thinker_call: premium model for core reasoning generation
    - reviewer_call: standard model for review & integration
    - reviser_call: cheap model for structured action execution
    - evaluator_call: standard model for quality evaluation

    Returns:
        PipelineItem with all pipeline outputs.
    """
    question = item["Question"]
    answer = item["Answer"]
    question_type = item.get("Question Type", "Unknown")

    # Stage 0: Difficulty classification (cheap model)
    classification = classify_difficulty(question, answer, classifier_call)
    difficulty = classification.difficulty

    # Stage 1: Thinker — core reasoning (premium model)
    if difficulty == DifficultyLevel.HARD:
        n_samples = config.SELF_CONSISTENCY_N
    else:
        n_samples = 1

    draft_chain = generate_with_consistency(
        question, answer, rag_tool, thinker_call,
        n_samples=n_samples,
        temperatures=config.REACT_TEMPERATURES,
    )

    # Stage 2: Reviewer (standard model)
    unified_actions, critiques = run_review(
        draft_chain, question, difficulty, rag_tool, reviewer_call
    )

    # Stage 3: Reviser (cheap model, skip for Easy)
    if difficulty != DifficultyLevel.EASY:
        revised_chain = execute_revision(draft_chain, unified_actions, reviser_call, rag_tool)
    else:
        revised_chain = draft_chain

    # Stage 4: Evaluator (standard model) — all difficulty levels
    faith, _ = judge_faithfulness(revised_chain, evaluator_call)
    logic, _ = judge_logical_completeness(revised_chain, evaluator_call)
    auto = compute_auto_metrics(revised_chain)
    w = config.EVAL_WEIGHTS
    overall = (
        faith / 5 * w["faithfulness"]
        + auto["structure"] * w["structure"]
        + auto["information_density"] * w["information_density"]
        + logic / 5 * w["logical_completeness"]
        + auto["traceability"] * w["traceability"]
    )
    scores = QualityScores(
        faithfulness=faith,
        logical_completeness=logic,
        overall=round(overall, 3),
        **auto,
    )

    # Quality gate: Hard path with low score → one more review+revise
    if difficulty == DifficultyLevel.HARD and scores.overall < config.QUALITY_GATE_THRESHOLD:
        actions2, critiques2 = run_review(
            revised_chain, question, difficulty, rag_tool, reviewer_call
        )
        revised_chain = execute_revision(revised_chain, actions2, reviser_call, rag_tool)
        critiques.extend(critiques2)

        faith, _ = judge_faithfulness(revised_chain, evaluator_call)
        logic, _ = judge_logical_completeness(revised_chain, evaluator_call)
        auto = compute_auto_metrics(revised_chain)
        w = config.EVAL_WEIGHTS
        overall = (
            faith / 5 * w["faithfulness"]
            + auto["structure"] * w["structure"]
            + auto["information_density"] * w["information_density"]
            + logic / 5 * w["logical_completeness"]
            + auto["traceability"] * w["traceability"]
        )
        scores = QualityScores(
            faithfulness=faith,
            logical_completeness=logic,
            overall=round(overall, 3),
            **auto,
        )

    return PipelineItem(
        id=item_id,
        question=question,
        answer=answer,
        question_type=question_type,
        difficulty=difficulty,
        draft_chain=draft_chain,
        unified_actions=unified_actions,
        revised_chain=revised_chain,
        quality_scores=scores,
        critique_history=critiques,
        metadata={
            "classification_raw": classification.raw_response,
            "n_samples": n_samples,
            "models": {
                "classifier": config.MODEL_LIGHT,
                "thinker": config.MODEL_PREMIUM,
                "reviewer": config.MODEL_STANDARD,
                "reviser": config.MODEL_LIGHT,
                "evaluator": config.MODEL_STANDARD,
            },
        },
    )


def _save_checkpoint(results: List[PipelineItem], path: Path):
    """Save current results as checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [item.to_dict() for item in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline(
    n_items: int = 1000,
    resume: bool = True,
):
    """Main entry point.

    Args:
        n_items: Number of items to process.
        resume: If True, resume from checkpoint if available.
    """
    print("=" * 60)
    print("Agricultural Reasoning Pipeline")
    print("=" * 60)

    # Load RAG tool
    print("Loading RAG tool...")
    rag_tool = load_rag_tool()
    print("RAG tool loaded.")

    # Create base LLM caller + stage-specific callers (model tiering)
    llm_call = create_llm_caller(
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        model=config.MODEL_STANDARD,
    )
    classifier_call = create_stage_caller(llm_call, config.MODEL_LIGHT)
    thinker_call = create_stage_caller(llm_call, config.MODEL_PREMIUM)
    reviewer_call = create_stage_caller(llm_call, config.MODEL_STANDARD)
    reviser_call = create_stage_caller(llm_call, config.MODEL_LIGHT)
    evaluator_call = create_stage_caller(llm_call, config.MODEL_STANDARD)

    # Load or create sample
    if config.PATH_SAMPLE.exists():
        print(f"Loading existing sample from {config.PATH_SAMPLE}")
        with open(config.PATH_SAMPLE, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        print(f"Creating new sample of {n_items} items...")
        items = subsample_agthoughts(n_items)
        save_sample(items, config.PATH_SAMPLE)

    # Resume from checkpoint
    results = []
    start_idx = 0
    if resume and config.PATH_OUTPUT.exists():
        print("Resuming from checkpoint...")
        with open(config.PATH_OUTPUT, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        for d in checkpoint_data:
            results.append(PipelineItem.from_dict(d))
        start_idx = len(results)
        print(f"Resumed {start_idx} items.")

    # Process remaining items
    consecutive_failures = 0
    for i in tqdm(range(start_idx, len(items)), desc="Processing"):
        item = items[i]
        item_id = f"item_{i:04d}"
        try:
            result = process_item(
                item, item_id, rag_tool, llm_call,
                classifier_call, thinker_call, reviewer_call,
                reviser_call, evaluator_call,
            )
            results.append(result)
            consecutive_failures = 0

            # Checkpoint after each item
            _save_checkpoint(results, config.PATH_OUTPUT)

        except Exception as e:
            print(f"\nError processing {item_id}: {e}")
            consecutive_failures += 1
            if consecutive_failures >= config.CONSECUTIVE_FAILURE_LIMIT:
                print(f"Too many consecutive failures ({consecutive_failures}). Stopping.")
                break

        # Rate limiting
        time.sleep(config.API_CALL_INTERVAL)

    # Final save
    _save_checkpoint(results, config.PATH_OUTPUT)
    print(f"\nDone. Processed {len(results)} items. Output: {config.PATH_OUTPUT}")

    # Summary stats
    difficulties = {}
    for r in results:
        d = r.difficulty.value if r.difficulty else "unknown"
        difficulties[d] = difficulties.get(d, 0) + 1
    print(f"Difficulty distribution: {difficulties}")

    if results:
        scores = [r.quality_scores.overall for r in results if r.quality_scores]
        if scores:
            print(f"Average quality score: {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    run_pipeline()
