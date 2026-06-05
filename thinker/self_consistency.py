"""Self-Consistency: generate multiple reasoning paths, select the best one.

Imports generate_react_chain from the same package (acceptable intra-package dependency).
Tests mock this import for isolation.
"""

from typing import List

from models import ReasoningChain

from .react_generator import generate_react_chain
from .socratic_challenger import socratic_challenge, revise_with_socratic


def _score_candidate(chain: ReasoningChain) -> float:
    """Score a candidate chain for selection.

    Criteria:
    - Step count: 3-7 is optimal (penalty outside range)
    - Evidence utilization: more steps with evidence = better
    - Type diversity: more step types used = better
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


def _select_best(candidates: List[ReasoningChain]) -> ReasoningChain:
    """Select the best chain from candidates based on scoring."""
    scored = [(chain, _score_candidate(chain)) for chain in candidates if chain.steps]
    if not scored:
        # Return the first candidate as fallback
        return candidates[0] if candidates else ReasoningChain(steps=[])
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
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
        chain = _select_best(candidates)

    # Socratic self-challenge on the selected chain
    challenges = socratic_challenge(chain, question, answer, llm_call)
    if challenges:
        chain = revise_with_socratic(chain, challenges, question, answer, llm_call, rag_tool)

    return chain
