"""LLM-as-Judge evaluation: Faithfulness and Logical Completeness.

Dependencies: types only. llm_call injected as parameter.
"""

import json
import re
from typing import List, Tuple

from models import ReasoningChain


FAITHFULNESS_PROMPT = """Evaluate faithfulness of a reasoning chain (1-5 scale).
Faithfulness = every factual claim is (a) supported by cited evidence, or (b) logically derived from cited steps.

1 = Most claims unsupported | 2 = Many unsupported | 3 = Some unsupported | 4 = Few unsupported | 5 = All grounded

Output JSON only:
{{"score": N, "notes": [{{"step": N, "assessment": "...", "issue": "..."}}]}}

---

Reasoning Chain:
{chain}"""


LOGICAL_COMPLETENESS_PROMPT = """Evaluate logical completeness of a reasoning chain (1-5 scale).
Completeness = chain covers all necessary reasoning steps, no major gaps in logical flow.

1 = Major gaps, conclusion doesn't follow | 2 = Several missing | 3 = Some gaps | 4 = Minor gaps | 5 = Complete

Output JSON only:
{{"score": N, "notes": [{{"step": N, "assessment": "...", "issue": "..."}}]}}

---

Reasoning Chain:
{chain}"""


def _parse_score_and_notes(raw: str, dimension: str) -> Tuple[float, List[dict]]:
    """Parse score and notes from LLM response."""
    # Try JSON
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            score = float(data.get("score", 3))
            notes = data.get("notes", [])
            return max(1.0, min(5.0, score)), notes
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            score = float(data.get("score", 3))
            notes = data.get("notes", [])
            return max(1.0, min(5.0, score)), notes
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: extract number
    num_match = re.search(r"[1-5]", raw)
    if num_match:
        return float(num_match.group(0)), []

    return 3.0, []


def judge_faithfulness(chain: ReasoningChain, llm_call) -> Tuple[float, List[dict]]:
    """Evaluate faithfulness of a reasoning chain.

    Args:
        chain: Reasoning chain to evaluate.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        Tuple of (score 1-5, step_notes).
    """
    prompt = FAITHFULNESS_PROMPT.format(chain=chain.to_text())
    raw = llm_call(prompt, system="You are an evaluation judge.", temperature=0.1, max_tokens=1024)
    return _parse_score_and_notes(raw, "faithfulness")


def judge_logical_completeness(chain: ReasoningChain, llm_call) -> Tuple[float, List[dict]]:
    """Evaluate logical completeness of a reasoning chain.

    Args:
        chain: Reasoning chain to evaluate.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        Tuple of (score 1-5, step_notes).
    """
    prompt = LOGICAL_COMPLETENESS_PROMPT.format(chain=chain.to_text())
    raw = llm_call(prompt, system="You are an evaluation judge.", temperature=0.1, max_tokens=1024)
    return _parse_score_and_notes(raw, "logical_completeness")
