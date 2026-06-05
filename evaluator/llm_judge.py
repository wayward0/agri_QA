"""LLM-as-Judge evaluation: Faithfulness and Logical Completeness.

Dependencies: types only. llm_call injected as parameter.
"""

import json
import re
from typing import List, Tuple

from models import ReasoningChain


FAITHFULNESS_PROMPT = """Evaluate the faithfulness of this agricultural reasoning chain.

Faithfulness means: every factual claim is either (a) supported by cited evidence,
or (b) logically derived from steps that cite evidence.

Reasoning Chain:
{chain}

Rate faithfulness on a scale of 1-5:
1 = Most claims unsupported
2 = Many claims unsupported
3 = Some claims unsupported
4 = Few claims unsupported, most grounded
5 = All claims properly grounded

Output JSON only:
{{"score": N, "notes": [{{"step": N, "assessment": "...", "issue": "..."}}]}}"""


LOGICAL_COMPLETENESS_PROMPT = """Evaluate the logical completeness of this agricultural reasoning chain.

Logical completeness means: the chain covers all necessary reasoning steps
to justify the answer, with no major gaps in the logical flow.

Reasoning Chain:
{chain}

Rate logical completeness on a scale of 1-5:
1 = Major logical gaps, conclusion doesn't follow
2 = Several missing steps
3 = Some gaps but conclusion mostly supported
4 = Minor gaps, conclusion well-supported
5 = Complete logical chain, no gaps

Output JSON only:
{{"score": N, "notes": [{{"step": N, "assessment": "...", "issue": "..."}}]}}"""


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
