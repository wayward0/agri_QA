"""Self-Consistency: generate multiple reasoning paths, select the best one.

Selection uses two-stage approach:
1. Semantic clustering of final conclusions (majority vote)
2. Heuristic scoring within the majority cluster

Imports generate_react_chain from the same package (acceptable intra-package dependency).
Tests mock this import for isolation.
"""

import json
import re
from typing import List, Optional

from models import ReasoningChain

from .react_generator import generate_react_chain
from .socratic_challenger import socratic_challenge, revise_with_socratic


CLUSTER_SYSTEM_PROMPT = "You are an agricultural reasoning evaluator."

CLUSTER_PROMPT = """You are given the final conclusions from {n} reasoning chains about the same
agricultural question. Group them into clusters of semantically equivalent conclusions.

Question: {question}
Expected Answer: {answer}

Conclusions:
{conclusions_text}

Rules:
- Two conclusions are "equivalent" if they recommend the same core action/answer, even with different wording
- Two conclusions are "different" if they contradict each other or recommend fundamentally different approaches
- A conclusion is "off-topic" if it doesn't address the question at all

Output (JSON only):
{{"clusters": [
  {{"label": "brief description of this conclusion group", "indices": [0, 2]}},
  {{"label": "...", "indices": [1]}}
]}}

Each index must appear in exactly one cluster."""


def _extract_conclusion(chain: ReasoningChain) -> str:
    """Extract the conclusion text from a reasoning chain."""
    # Prefer explicit conclusion step
    for step in chain.steps:
        if step.type == "conclusion":
            return step.content
    # Fallback: last step
    if chain.steps:
        return chain.steps[-1].content
    return ""


def _score_candidate(chain: ReasoningChain) -> float:
    """Score a candidate chain for selection.

    Criteria:
    - Step count: 3-7 is optimal (penalty outside range)
    - Evidence utilization: more steps with evidence = better
    - Type diversity: more step types used = better
    - High confidence ratio
    """
    n_steps = len(chain.steps)
    if n_steps == 0:
        return 0.0

    # Step count score (optimal 3-7)
    if 3 <= n_steps <= 7:
        step_score = 1.0
    elif n_steps < 3:
        step_score = n_steps / 3
    else:
        step_score = max(0.5, 1.0 - (n_steps - 7) * 0.1)

    # Evidence utilization
    with_evidence = sum(1 for s in chain.steps if s.evidence)
    evidence_score = with_evidence / n_steps

    # Type diversity
    types_used = set(s.type for s in chain.steps)
    diversity_score = len(types_used) / 5  # 5 main types expected

    # High confidence ratio
    high_conf = sum(1 for s in chain.steps if s.confidence == "high")
    conf_score = high_conf / n_steps

    return (step_score * 0.3 + evidence_score * 0.35 +
            diversity_score * 0.2 + conf_score * 0.15)


def _cluster_conclusions(
    candidates: List[ReasoningChain],
    question: str,
    answer: str,
    llm_call,
) -> Optional[List[List[int]]]:
    """Cluster candidate chains by semantic equivalence of their conclusions.

    Returns list of clusters, each cluster is a list of indices into candidates.
    Returns None if clustering fails.
    """
    conclusions = []
    for i, chain in enumerate(candidates):
        concl = _extract_conclusion(chain)
        conclusions.append(f"[{i}] {concl}")

    conclusions_text = "\n".join(conclusions)
    prompt = CLUSTER_PROMPT.format(
        n=len(candidates),
        question=question,
        answer=answer,
        conclusions_text=conclusions_text,
    )

    raw = llm_call(prompt, system=CLUSTER_SYSTEM_PROMPT, temperature=0.0, max_tokens=512)

    # Parse JSON
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    clusters = data.get("clusters", [])
    if not clusters:
        return None

    # Validate: each index appears exactly once
    all_indices = []
    for cluster in clusters:
        indices = cluster.get("indices", [])
        if not isinstance(indices, list):
            return None
        all_indices.extend(indices)

    expected = set(range(len(candidates)))
    if set(all_indices) != expected:
        # Allow partial clustering — fill in missing indices as singletons
        seen = set(all_indices)
        for i in expected - seen:
            clusters.append({"indices": [i]})

    return [c["indices"] for c in clusters]


def _select_best(
    candidates: List[ReasoningChain],
    question: str = "",
    answer: str = "",
    llm_call=None,
) -> ReasoningChain:
    """Select the best chain using semantic clustering + heuristic scoring.

    1. Cluster conclusions by semantic equivalence (majority vote)
    2. Pick the largest cluster (ties broken by total heuristic score)
    3. Within the winning cluster, pick the highest heuristic score
    """
    scored = [(i, chain, _score_candidate(chain)) for i, chain in enumerate(candidates) if chain.steps]
    if not scored:
        return candidates[0] if candidates else ReasoningChain(steps=[])

    # Try semantic clustering
    clusters = None
    if llm_call and len(candidates) > 1:
        clusters = _cluster_conclusions(candidates, question, answer, llm_call)

    if clusters:
        # Score each cluster: size first, then sum of heuristic scores as tiebreaker
        scored_map = {i: hs for i, _, hs in scored}
        cluster_scores = []
        for cluster_indices in clusters:
            size = len(cluster_indices)
            total_hs = sum(scored_map.get(i, 0.0) for i in cluster_indices)
            cluster_scores.append((cluster_indices, size, total_hs))

        # Sort: largest cluster first, then highest total heuristic score
        cluster_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        winner_indices = cluster_scores[0][0]

        # Within winner cluster, pick highest heuristic score
        winner_candidates = [
            (chain, hs) for i, chain, hs in scored if i in winner_indices
        ]
        winner_candidates.sort(key=lambda x: x[1], reverse=True)
        best = winner_candidates[0][0]
    else:
        # Fallback: pure heuristic
        scored.sort(key=lambda x: x[2], reverse=True)
        best = scored[0][1]

    best.self_consistency_selected = 1
    return best


def generate_with_consistency(
    question: str,
    answer: str,
    rag_tool,
    llm_call,
    n_samples: int = 1,
    temperatures: List[float] = None,
) -> ReasoningChain:
    """Generate reasoning chain with optional self-consistency and Socratic challenge.

    Args:
        question: The agricultural question.
        answer: The known answer.
        rag_tool: RAGTool instance.
        llm_call: Callable matching LLMCallFn protocol.
        n_samples: Number of paths to generate (1 = no self-consistency).
        temperatures: Temperature for each sample.

    Returns:
        The best ReasoningChain, Socratic-challenged and revised.
    """
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.0]

    if n_samples <= 1:
        # Single path — no self-consistency
        chain = generate_react_chain(
            question, answer, rag_tool, llm_call,
            temperature=temperatures[0],
        )
    else:
        # Multiple paths — self-consistency selection
        candidates = []
        for i in range(n_samples):
            temp = temperatures[i] if i < len(temperatures) else temperatures[-1]
            chain = generate_react_chain(
                question, answer, rag_tool, llm_call,
                temperature=temp,
            )
            candidates.append(chain)
        chain = _select_best(candidates, question=question, answer=answer, llm_call=llm_call)

    # Socratic self-challenge on the selected chain
    challenges = socratic_challenge(chain, question, answer, llm_call)
    if challenges:
        chain = revise_with_socratic(chain, challenges, question, answer, llm_call, rag_tool)

    return chain
