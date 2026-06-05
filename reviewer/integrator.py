"""Phase C: Integration — merge critiques into structured atomic actions.

Produces UnifiedActions consumed by the Reviser.
Also provides run_review() convenience function for the full review pipeline.
"""

import json
import re
from typing import List, Optional, Tuple

from models import (
    DifficultyLevel,
    ReasoningChain,
    ReviewAction,
    ReviewCritique,
    UnifiedActions,
)

from .logic_reviewer import review_logic
from .external_reviewer import review_external


SYSTEM_PROMPT = "You are an integrator merging critiques into structured modification actions."

INTEGRATION_PROMPT = """You are an integrator merging critiques from logic and external reviewers.
Output STRUCTURED modification actions, NOT natural language feedback.

Logic Critique:
{logic_critique}

External Critique:
{external_critique}

AVAILABLE ACTIONS (atomic operations):
- add_evidence:    {{"action": "add_evidence", "target_step": N, "params": {{"evidence": "..."}}}}
- revise_step:     {{"action": "revise_step", "target_step": N, "params": {{"revised_content": "..."}}}}
- insert_step:     {{"action": "insert_step", "target_step": N, "params": {{"new_step": {{"type": "...", "content": "...", "evidence": null, "confidence": "medium"}}}}}}
- remove_step:     {{"action": "remove_step", "target_step": N, "params": {{}}}}
- merge_steps:     {{"action": "merge_steps", "params": {{"step_a": N, "step_b": M}}}}
- adjust_confidence: {{"action": "adjust_confidence", "target_step": N, "params": {{"new_confidence": "low|medium|high"}}}}

RULES:
1. Each action is ATOMIC — one change per action
2. If a step needs both evidence and content revision → TWO separate actions
3. Priority: P0 (must fix: factual errors, contradictions), P1 (should fix: missing steps, weak evidence), P2 (optional: style, clarity)
4. Do NOT generate vague actions — use specific action types

Output (JSON only):
{{"priority_actions": [
    {{"priority": "P0", "action": "add_evidence", "target_step": 3,
      "params": {{"evidence": "..."}}, "reason": "Factual: unsupported claim"}}
], "optional_improvements": [...], "conflicts_resolved": [...]}}"""


# Default actions when no critique is provided
_EMPTY_UNIFIED_ACTIONS = UnifiedActions(
    priority_actions=[],
    optional_improvements=[],
    conflicts_resolved=[],
)


def _critique_to_text(critique: Optional[ReviewCritique]) -> str:
    """Format a ReviewCritique for the integration prompt."""
    if not critique or not critique.issues:
        return "No issues found."
    return json.dumps(critique.issues, indent=2)


def _parse_unified_actions(raw: str) -> UnifiedActions:
    """Parse UnifiedActions from LLM response."""
    # Try to find JSON object
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return _EMPTY_UNIFIED_ACTIONS

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        # Try code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return _EMPTY_UNIFIED_ACTIONS
        else:
            return _EMPTY_UNIFIED_ACTIONS

    priority_actions = []
    for a in data.get("priority_actions", []):
        try:
            priority_actions.append(ReviewAction(
                action=a.get("action", "revise_step"),
                target_step=a.get("target_step", 0),
                priority=a.get("priority", "P1"),
                params=a.get("params", {}),
                reason=a.get("reason", ""),
            ))
        except (ValueError, KeyError):
            continue

    optional_improvements = []
    for a in data.get("optional_improvements", []):
        try:
            optional_improvements.append(ReviewAction(
                action=a.get("action", "revise_step"),
                target_step=a.get("target_step", 0),
                priority="P2",
                params=a.get("params", {}),
                reason=a.get("reason", ""),
            ))
        except (ValueError, KeyError):
            continue

    return UnifiedActions(
        priority_actions=priority_actions,
        optional_improvements=optional_improvements,
        conflicts_resolved=data.get("conflicts_resolved", []),
    )


def integrate_critiques(
    logic_critique: ReviewCritique,
    external_critique: Optional[ReviewCritique],
    llm_call,
) -> UnifiedActions:
    """Merge logic and external critiques into structured actions.

    Args:
        logic_critique: Phase A output.
        external_critique: Phase B output (None if not run).
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        UnifiedActions with priority-sorted atomic operations.
    """
    prompt = INTEGRATION_PROMPT.format(
        logic_critique=_critique_to_text(logic_critique),
        external_critique=_critique_to_text(external_critique),
    )
    raw = llm_call(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
    return _parse_unified_actions(raw)


def run_review(
    chain: ReasoningChain,
    question: str,
    difficulty: DifficultyLevel,
    rag_tool,
    llm_call,
    answer: str = "",
) -> Tuple[UnifiedActions, List[ReviewCritique]]:
    """Run the appropriate review phases based on difficulty.

    Args:
        chain: Draft reasoning chain from Thinker.
        question: Original question.
        difficulty: Difficulty level (determines which phases to run).
        rag_tool: RAGTool instance.
        llm_call: Callable matching LLMCallFn protocol.
        answer: Original answer text for semantic drift detection.

    Returns:
        Tuple of (unified_actions, critique_history).
    """
    critiques = []

    # Phase A: logic + semantic alignment (always run)
    logic_critique = review_logic(chain, llm_call, answer=answer)
    critiques.append(logic_critique)

    # Phase B: Medium and Hard
    external_critique = None
    if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
        external_critique = review_external(chain, question, rag_tool, llm_call)
        critiques.append(external_critique)

    # Phase C: integration (always run if we have any critique)
    unified = integrate_critiques(logic_critique, external_critique, llm_call)

    return unified, critiques
