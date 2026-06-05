"""Phase A: Internal Logic Review.

Pure logic analysis — no RAG calls.
Checks for logical gaps, missing steps, contradictions, circular reasoning.
"""

import json
import re
from typing import List

from models import ReasoningChain, ReviewCritique


SYSTEM_PROMPT = "You are a logical reasoning critic for agricultural reasoning chains."

LOGIC_REVIEW_PROMPT = """Analyze this reasoning chain for structural logic and semantic alignment.
Do NOT check facts or domain completeness — those are handled separately.

STRUCTURAL LOGIC — for each adjacent step pair (N, N+1):
1. Is there a logical connection? (logical_gap)
2. Any contradictions between non-adjacent steps? (contradiction)
3. Any circular reasoning? (circular)

SEMANTIC ALIGNMENT — for the chain as a whole:
4. Does the final conclusion align with the original Answer? (semantic_drift)
5. Does the conclusion follow from the premises? (non_sequitur)

OUTPUT (JSON array only, no other text):
[{{"step": N, "issue_type": "logical_gap|contradiction|circular|semantic_drift|non_sequitur",
  "description": "...", "severity": "high|medium|low"}}]
If no issues, output: []

---

Reasoning Chain:
{chain}

Original Answer:
{answer}"""


def _parse_issues(raw: str) -> List[dict]:
    """Parse JSON array of issues from LLM response."""
    # Try to find JSON array
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            issues = json.loads(json_match.group(0))
            if isinstance(issues, list):
                return issues
        except json.JSONDecodeError:
            pass

    # Try to find JSON in code block
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if json_match:
        try:
            issues = json.loads(json_match.group(1))
            if isinstance(issues, list):
                return issues
        except json.JSONDecodeError:
            pass

    return []


def review_logic(chain: ReasoningChain, llm_call, answer: str = "") -> ReviewCritique:
    """Run Phase A: logic + semantic alignment review.

    Args:
        chain: The reasoning chain to review.
        llm_call: Callable matching LLMCallFn protocol.
        answer: Original answer text for semantic drift detection.

    Returns:
        ReviewCritique with phase="A_logic".
    """
    chain_text = chain.to_text()
    prompt = LOGIC_REVIEW_PROMPT.format(chain=chain_text, answer=answer)
    raw = llm_call(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
    issues = _parse_issues(raw)
    return ReviewCritique(phase="A_logic", issues=issues)
