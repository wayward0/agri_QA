"""Phase B: External Standards Review.

Fact-checks reasoning chain claims against RAG evidence.
Checks completeness, domain expertise, practical advice quality.
"""

import json
import re
from typing import List

from models import ReasoningChain, ReviewCritique


SYSTEM_PROMPT = "You are an agricultural domain expert reviewing reasoning chains."

EXTERNAL_REVIEW_PROMPT = """Review this agricultural reasoning chain against external standards.

CHECK (for each step with a factual claim):
1. Correctness — is the claim supported by the evidence below?
2. Evidence quality — is the cited evidence reliable and relevant?

CHECK (for the overall chain):
3. Completeness — does it cover all sub-questions?
4. Domain accuracy — is agricultural terminology correct?
5. Practicality — is the advice specific and actionable?

OUTPUT (JSON array only, no other text):
[{{"step": N, "dimension": "factual|evidence|completeness|domain|practical",
  "issue": "...", "suggestion": "...", "severity": "high|medium|low"}}]
If no issues, output: []

---

Reasoning Chain:
{chain}

Fact-Check Evidence:
{fact_check}

Gap-Fill Evidence:
{gap_fill}"""


def _parse_issues(raw: str) -> List[dict]:
    """Parse JSON array of issues from LLM response."""
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            issues = json.loads(json_match.group(0))
            if isinstance(issues, list):
                return issues
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if json_match:
        try:
            issues = json.loads(json_match.group(1))
            if isinstance(issues, list):
                return issues
        except json.JSONDecodeError:
            pass

    return []


def review_external(
    chain: ReasoningChain,
    question: str,
    rag_tool,
    llm_call,
) -> ReviewCritique:
    """Run Phase B: external standards review.

    Args:
        chain: The reasoning chain to review.
        question: Original question (for gap-fill retrieval).
        rag_tool: RAGTool instance for fact-checking.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        ReviewCritique with phase="B_external".
    """
    # Collect all queries for batch retrieval
    fact_check_steps = [
        step for step in chain.steps
        if step.type in ("knowledge_application", "causal_reasoning", "evidence_integration")
    ]
    queries = [step.content for step in fact_check_steps]
    queries.append(question)  # gap-fill query

    # Single batch retrieval call (one embedding API call for all queries)
    batch_results = rag_tool.retrieve_batch(queries, intent="fact_check", top_k=3)

    # Map results back to steps
    fact_check_parts = []
    for step, evidence in zip(fact_check_steps, batch_results[:-1]):
        if evidence:
            fact_check_parts.append(f"Step {step.step}: " + "; ".join(
                f"[{e.source}] {e.content[:200]}" for e in evidence
            ))
    fact_check_text = "\n".join(fact_check_parts) if fact_check_parts else "No fact-check evidence retrieved."

    # Gap-fill results (last item in batch)
    gap_evidence = batch_results[-1]
    gap_text = "\n".join(
        f"[{e.source}] {e.content[:200]}" for e in gap_evidence
    ) if gap_evidence else "No gap-fill evidence retrieved."

    prompt = EXTERNAL_REVIEW_PROMPT.format(
        chain=chain.to_text(),
        fact_check=fact_check_text,
        gap_fill=gap_text,
    )
    raw = llm_call(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
    issues = _parse_issues(raw)
    return ReviewCritique(phase="B_external", issues=issues)
