"""Structured action executor — Reviser module.

Executes atomic operations from UnifiedActions on a ReasoningChain.
~50% of operations are programmatic (no LLM needed).

Dependencies: types only. llm_call and rag_tool injected as parameters.
"""

import json
import re
from typing import List, Optional

from models import ReasoningChain, ReasoningStep, UnifiedActions, ReviewAction


# Execution order: deletions first to avoid index shifts
ACTION_PRIORITY_ORDER = {
    "remove_step": 0,
    "merge_steps": 1,
    "revise_step": 2,
    "insert_step": 3,
    "add_evidence": 4,
    "adjust_confidence": 5,
}


def _sort_actions(actions: List[ReviewAction]) -> List[ReviewAction]:
    """Sort actions by execution order to avoid index conflicts."""
    return sorted(actions, key=lambda a: ACTION_PRIORITY_ORDER.get(a.action, 99))


def _add_evidence(steps: List[ReasoningStep], action: ReviewAction) -> List[ReasoningStep]:
    """Programmatic: set evidence field on target step."""
    target = action.target_step
    evidence = action.params.get("evidence", "")
    for step in steps:
        if step.step == target:
            step.evidence = evidence
            break
    return steps


def _adjust_confidence(steps: List[ReasoningStep], action: ReviewAction) -> List[ReasoningStep]:
    """Programmatic: set confidence field on target step."""
    target = action.target_step
    new_conf = action.params.get("new_confidence", "medium")
    for step in steps:
        if step.step == target:
            step.confidence = new_conf
            break
    return steps


def _remove_step(steps: List[ReasoningStep], action: ReviewAction) -> List[ReasoningStep]:
    """Programmatic: remove target step."""
    target = action.target_step
    return [s for s in steps if s.step != target]


def _revise_step(steps: List[ReasoningStep], action: ReviewAction, llm_call) -> List[ReasoningStep]:
    """LLM-based: rewrite step content."""
    target = action.target_step
    revised_content = action.params.get("revised_content")
    if revised_content:
        # Use provided content directly
        for step in steps:
            if step.step == target:
                step.content = revised_content
                break
        return steps

    # LLM generates revised content
    for step in steps:
        if step.step == target:
            prompt = (
                f"Revise this reasoning step to address: {action.reason}\n\n"
                f"Original step: {step.content}\n\n"
                f"Provide ONLY the revised step content, nothing else."
            )
            revised = llm_call(prompt, temperature=0.3, max_tokens=512)
            step.content = revised
            break
    return steps


def _insert_step(
    steps: List[ReasoningStep],
    action: ReviewAction,
    llm_call,
) -> List[ReasoningStep]:
    """LLM-based: generate and insert a new reasoning step."""
    target = action.target_step
    new_step_data = action.params.get("new_step")

    if new_step_data and new_step_data.get("content"):
        # Use provided step data
        new_step = ReasoningStep(
            step=target,
            type=new_step_data.get("type", "knowledge_application"),
            content=new_step_data["content"],
            evidence=new_step_data.get("evidence"),
            confidence=new_step_data.get("confidence", "medium"),
        )
    else:
        # LLM generates new step
        context_steps = [s for s in steps if s.step <= target]
        context_text = "\n".join(f"Step {s.step}: {s.content}" for s in context_steps[-2:])
        prompt = (
            f"Generate a new reasoning step to insert at position {target}.\n"
            f"Reason: {action.reason}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Output JSON: {{\"type\": \"...\", \"content\": \"...\", \"confidence\": \"...\"}}"
        )
        raw = llm_call(prompt, temperature=0.3, max_tokens=512)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                new_step = ReasoningStep(
                    step=target,
                    type=data.get("type", "knowledge_application"),
                    content=data.get("content", ""),
                    evidence=data.get("evidence"),
                    confidence=data.get("confidence", "medium"),
                )
            except (json.JSONDecodeError, ValueError):
                return steps
        else:
            return steps

    # Insert at target position
    steps.insert(target - 1, new_step)
    return steps


def _merge_steps(steps: List[ReasoningStep], action: ReviewAction, llm_call) -> List[ReasoningStep]:
    """LLM-based: merge two steps into one."""
    step_a_idx = action.params.get("step_a", 0)
    step_b_idx = action.params.get("step_b", 0)

    step_a = next((s for s in steps if s.step == step_a_idx), None)
    step_b = next((s for s in steps if s.step == step_b_idx), None)

    if not step_a or not step_b:
        return steps

    prompt = (
        f"Merge these two reasoning steps into one:\n\n"
        f"Step {step_a_idx} [{step_a.type}]: {step_a.content}\n"
        f"Step {step_b_idx} [{step_b.type}]: {step_b.content}\n\n"
        f"Output JSON: {{\"type\": \"...\", \"content\": \"...\", \"confidence\": \"...\"}}"
    )
    raw = llm_call(prompt, temperature=0.3, max_tokens=512)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            merged = ReasoningStep(
                step=step_a_idx,
                type=data.get("type", step_a.type),
                content=data.get("content", step_a.content + " " + step_b.content),
                evidence=step_a.evidence or step_b.evidence,
                confidence=data.get("confidence", step_a.confidence),
            )
            steps = [s for s in steps if s.step not in (step_a_idx, step_b_idx)]
            steps.insert(step_a_idx - 1, merged)
        except (json.JSONDecodeError, ValueError):
            pass

    return steps


def _reindex(steps: List[ReasoningStep]) -> List[ReasoningStep]:
    """Re-number steps sequentially after modifications."""
    for i, step in enumerate(steps):
        step.step = i + 1
    return steps


def execute_revision(
    chain: ReasoningChain,
    actions: UnifiedActions,
    llm_call,
    rag_tool=None,
) -> ReasoningChain:
    """Execute structured atomic actions on a reasoning chain.

    Args:
        chain: Draft reasoning chain from Thinker.
        actions: UnifiedActions from Reviewer.
        llm_call: Callable matching LLMCallFn protocol.
        rag_tool: Optional RAGTool (unused currently, reserved for future).

    Returns:
        Revised ReasoningChain.
    """
    steps = [ReasoningStep(
        step=s.step,
        type=s.type,
        content=s.content,
        evidence=s.evidence,
        confidence=s.confidence,
    ) for s in chain.steps]  # deep copy

    all_actions = list(actions.priority_actions) + list(actions.optional_improvements)
    sorted_actions = _sort_actions(all_actions)

    for action in sorted_actions:
        if action.action == "add_evidence":
            steps = _add_evidence(steps, action)
        elif action.action == "adjust_confidence":
            steps = _adjust_confidence(steps, action)
        elif action.action == "remove_step":
            steps = _remove_step(steps, action)
        elif action.action == "revise_step":
            steps = _revise_step(steps, action, llm_call)
        elif action.action == "insert_step":
            steps = _insert_step(steps, action, llm_call)
        elif action.action == "merge_steps":
            steps = _merge_steps(steps, action, llm_call)

    steps = _reindex(steps)
    return ReasoningChain(
        steps=steps,
        react_rounds=chain.react_rounds,
        self_consistency_selected=chain.self_consistency_selected,
    )
