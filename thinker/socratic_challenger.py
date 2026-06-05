"""Socratic self-challenge for reasoning chains.

After ReAct generates a draft chain, this module:
1. Generates Socratic questions targeting weaknesses
2. Revises the chain to address those challenges

RAG is available as an on-demand tool during revision.
"""

import json
import re
from typing import List, Optional

from models import ReasoningChain, ReasoningStep


CHALLENGE_SYSTEM_PROMPT = """You are a Socratic challenger for agricultural reasoning chains.
Your role is to ask probing questions that expose weaknesses in reasoning."""

CHALLENGE_PROMPT = """Here is an agricultural reasoning chain. Generate 3-5 Socratic questions
that challenge its weakest points. Each question should expose a specific weakness.

Question: {question}
Answer: {answer}

Reasoning Chain:
{chain_text}

Target these weakness types:
- unsupported_claim: A factual assertion with no evidence cited
- logical_gap: A logical jump between steps without justification
- alternative_ignored: An alternative explanation not considered
- missing_edge_case: A condition or exception not addressed
- overgeneralization: A conclusion that is too broad for the evidence

Output (JSON array only):
[{{"question": "Your Socratic question here", "target_step": N, "issue_type": "unsupported_claim|logical_gap|alternative_ignored|missing_edge_case|overgeneralization"}}]

If the chain has no significant weaknesses, output: []"""


REVISE_SYSTEM_PROMPT = """You are an agricultural reasoning expert revising a reasoning chain
based on Socratic challenges. You may call retrieve: <query> to get evidence when needed."""

REVISE_PROMPT = """Revise this reasoning chain to address the Socratic challenges below.

Question: {question}
Answer: {answer}

Original Chain:
{chain_text}

Socratic Challenges:
{challenges_text}

Instructions:
1. For each challenge, either:
   - Strengthen the step with evidence (use retrieve: <query> if you need facts)
   - Add a missing step
   - Revise overgeneralized claims
   - Acknowledge the limitation in confidence
2. Preserve the overall chain structure (step types, conclusion)
3. Keep steps that are not challenged unchanged

When you need evidence, output: retrieve: <search query>
When done revising, output the revised chain as JSON:

```json
{{"steps": [
  {{"step": 1, "type": "context_setup", "content": "...", "evidence": "...", "confidence": "high"}},
  ...
]}}
```"""


def _parse_challenges(raw: str) -> List[dict]:
    """Parse JSON array of Socratic challenges."""
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            challenges = json.loads(json_match.group(0))
            if isinstance(challenges, list):
                return [c for c in challenges if isinstance(c, dict) and "question" in c]
        except json.JSONDecodeError:
            pass
    return []


def _parse_revised_chain(response: str) -> Optional[List[dict]]:
    """Extract JSON reasoning chain from revision response."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("steps", [])
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*"steps"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("steps", [])
        except json.JSONDecodeError:
            pass
    return None


def socratic_challenge(
    chain: ReasoningChain,
    question: str,
    answer: str,
    llm_call,
) -> List[dict]:
    """Generate Socratic questions targeting weaknesses in the chain.

    Args:
        chain: The reasoning chain to challenge.
        question: The original question.
        answer: The known answer.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        List of challenge dicts with question, target_step, issue_type.
    """
    chain_text = chain.to_text()
    prompt = CHALLENGE_PROMPT.format(
        question=question,
        answer=answer,
        chain_text=chain_text,
    )
    raw = llm_call(prompt, system=CHALLENGE_SYSTEM_PROMPT, temperature=0.3, max_tokens=1024)
    return _parse_challenges(raw)


def revise_with_socratic(
    chain: ReasoningChain,
    challenges: List[dict],
    question: str,
    answer: str,
    llm_call,
    rag_tool=None,
) -> ReasoningChain:
    """Revise a reasoning chain based on Socratic challenges.

    Supports on-demand RAG retrieval during revision (RAG-as-Tool).

    Args:
        chain: The original reasoning chain.
        challenges: List of Socratic challenge dicts.
        question: The original question.
        answer: The known answer.
        llm_call: Callable matching LLMCallFn protocol.
        rag_tool: Optional RAGTool for on-demand evidence retrieval.

    Returns:
        Revised ReasoningChain.
    """
    chain_text = chain.to_text()
    challenges_text = "\n".join(
        f"- [{c.get('issue_type', 'unknown')}] Step {c.get('target_step', '?')}: {c['question']}"
        for c in challenges
    )
    prompt = REVISE_PROMPT.format(
        question=question,
        answer=answer,
        chain_text=chain_text,
        challenges_text=challenges_text,
    )

    # ReAct-style revision: allow retrieve actions
    max_rounds = 3
    observations_text = ""

    for _ in range(max_rounds):
        full_prompt = prompt + observations_text
        response = llm_call(full_prompt, system=REVISE_SYSTEM_PROMPT, temperature=0.2, max_tokens=2048)

        # Check if LLM wants to retrieve evidence
        retrieve_match = re.search(r"retrieve:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if retrieve_match and rag_tool:
            query = retrieve_match.group(1).strip()
            evidence = rag_tool.retrieve(query, intent="fact_check", top_k=3)
            obs = "\n".join(f"  - [{e.source}] {e.content[:300]}" for e in evidence)
            observations_text += f"\n\nObservation for '{query}':\n{obs}\nContinue revising."
            continue

        # Try to parse the revised chain
        steps_data = _parse_revised_chain(response)
        if steps_data:
            steps = []
            for i, sd in enumerate(steps_data):
                try:
                    steps.append(ReasoningStep(
                        step=sd.get("step", i + 1),
                        type=sd.get("type", "knowledge_application"),
                        content=sd.get("content", ""),
                        evidence=sd.get("evidence"),
                        confidence=sd.get("confidence", "medium"),
                    ))
                except ValueError:
                    continue
            if steps:
                return ReasoningChain(
                    steps=steps,
                    react_rounds=chain.react_rounds,
                    self_consistency_selected=chain.self_consistency_selected,
                )

    # Fallback: return original chain unchanged
    return chain
