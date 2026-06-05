"""Phase A: Internal Logic Review.

Pure logic analysis — no RAG calls.
Checks for logical gaps, missing steps, contradictions, circular reasoning.
"""

import json
import re
from typing import List

from models import ReasoningChain, ReviewCritique


SYSTEM_PROMPT = "You are a logical reasoning critic for agricultural reasoning chains."

LOGIC_REVIEW_PROMPT = """You are a logical reasoning critic. Analyze this agricultural reasoning chain
for internal logic issues. Do NOT check facts — only check reasoning structure.

For each adjacent step pair (N, N+1):
1. Is there a logical connection?
2. Is there a missing intermediate step?

For all steps:
3. Any contradictions between non-adjacent steps?
4. Any circular reasoning?
5. Does the conclusion follow from the premises?

Reasoning Chain:
{chain}

Output (JSON array only, no other text):
[{{"step": N, "issue_type": "logical_gap|missing_step|contradiction|circular|non_sequitur",
  "description": "...", "severity": "high|medium|low"}}]

If no issues found, output an empty array: []"""


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


def review_logic(chain: ReasoningChain, llm_call) -> ReviewCritique:
    """Run Phase A: internal logic review.

    Args:
        chain: The reasoning chain to review.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        ReviewCritique with phase="A_logic".
    """
    chain_text = chain.to_text()
    prompt = LOGIC_REVIEW_PROMPT.format(chain=chain_text)
    raw = llm_call(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
    issues = _parse_issues(raw)
    return ReviewCritique(phase="A_logic", issues=issues)
