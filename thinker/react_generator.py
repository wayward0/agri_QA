"""ReAct reasoning generator.

Implements the Thought -> Action(RAG) -> Observation loop.
Dependencies: types only. rag_tool and llm_call injected as parameters.
"""

import json
import re
from typing import List, Optional, Tuple

from models import ReasoningChain, ReasoningStep, Evidence


REACT_SYSTEM_PROMPT = """You are an agricultural reasoning expert using the ReAct framework.

Given a question and its answer, generate a structured reasoning chain that explains
how to arrive at the answer. Use the Thought-Action-Observation loop:

1. THOUGHT: Analyze what you know and what you need to find out.
2. ACTION: Either:
   - retrieve: <search query>  (to get more evidence)
   - FINISH  (when you have enough information to build the reasoning chain)

After each ACTION, you will receive an OBSERVATION with retrieved evidence.

When you choose FINISH, output a JSON reasoning chain:
```json
{"steps": [
  {"step": 1, "type": "context_setup", "content": "...", "evidence": "...", "confidence": "high"},
  {"step": 2, "type": "knowledge_application", "content": "...", "evidence": "...", "confidence": "medium"},
  ...
  {"step": N, "type": "conclusion", "content": "...", "confidence": "high"}
]}
```

Step types: context_setup, knowledge_application, causal_reasoning, comparison,
condition_analysis, evidence_integration, conclusion.

RULES:
- Every factual claim MUST cite evidence or be logically derived from cited steps
- If unsupported, set confidence to "low"
- Aim for 3-7 reasoning steps
- The final conclusion must connect back to the given answer"""

STEP_TYPES = (
    "context_setup", "knowledge_application", "causal_reasoning",
    "comparison", "condition_analysis", "evidence_integration", "conclusion",
)


def _build_react_prompt(
    question: str,
    answer: str,
    thoughts: List[str],
    observations: List[List[Evidence]],
) -> str:
    """Build the prompt for the current ReAct round."""
    parts = [f"Question: {question}\nAnswer: {answer}\n"]

    for i, thought in enumerate(thoughts):
        parts.append(f"Thought {i+1}: {thought}")
        if i < len(observations):
            parts.append(f"Observation {i+1}:")
            for e in observations[i]:
                parts.append(f"  - [{e.source}] {e.content[:300]}")
        parts.append("")

    parts.append("What is your next Thought and Action?")
    return "\n".join(parts)


def _parse_react_response(response: str) -> Tuple[str, str]:
    """Parse Thought and Action from LLM response.

    Returns (thought, action) where action is "FINISH" or "retrieve: <query>".
    """
    thought = ""
    action = "FINISH"

    # Try to extract Thought
    thought_match = re.search(r"Thought\s*\d*:\s*(.+?)(?=Action|$)", response, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Try to extract Action
    action_match = re.search(r"Action\s*\d*:\s*(.+?)$", response, re.MULTILINE | re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()
        if action_text.lower().startswith("retrieve:"):
            action = action_text
        elif "finish" in action_text.lower():
            action = "FINISH"
        else:
            action = action_text

    # Check for FINISH keyword
    if "FINISH" in response and "retrieve" not in response.lower():
        action = "FINISH"

    return thought, action


def _parse_chain_from_response(response: str) -> Optional[List[dict]]:
    """Extract JSON reasoning chain from LLM response."""
    # Try to find JSON block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("steps", [])
        except json.JSONDecodeError:
            pass

    # Try to find JSON without code block
    json_match = re.search(r'\{[^{}]*"steps"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("steps", [])
        except json.JSONDecodeError:
            pass

    return None


def _build_chain_from_thoughts(
    question: str,
    answer: str,
    thoughts: List[str],
    observations: List[List[Evidence]],
) -> ReasoningChain:
    """Fallback: build a ReasoningChain from accumulated thoughts when JSON parsing fails."""
    steps = []

    # Context setup from first thought
    if thoughts:
        steps.append(ReasoningStep(
            step=1,
            type="context_setup",
            content=thoughts[0],
            confidence="medium",
        ))

    # Knowledge application from observations
    for i, obs_list in enumerate(observations):
        if obs_list:
            evidence_text = "; ".join(e.content[:200] for e in obs_list[:2])
            thought_text = thoughts[i + 1] if i + 1 < len(thoughts) else ""
            steps.append(ReasoningStep(
                step=len(steps) + 1,
                type="evidence_integration",
                content=thought_text or evidence_text,
                evidence=evidence_text,
                confidence="medium",
            ))

    # Conclusion
    steps.append(ReasoningStep(
        step=len(steps) + 1,
        type="conclusion",
        content=f"Therefore, the answer is: {answer[:200]}",
        confidence="high",
    ))

    return ReasoningChain(steps=steps, react_rounds=len(thoughts))


def generate_react_chain(
    question: str,
    answer: str,
    rag_tool,
    llm_call,
    max_rounds: int = 5,
    temperature: float = 0.3,
) -> ReasoningChain:
    """Generate a reasoning chain using the ReAct framework.

    Args:
        question: The agricultural question.
        answer: The known answer.
        rag_tool: RAGTool instance for evidence retrieval.
        llm_call: Callable matching LLMCallFn protocol.
        max_rounds: Maximum ReAct loop iterations.
        temperature: LLM temperature.

    Returns:
        A ReasoningChain.
    """
    # ReAct loop — RAG is on-demand, called only when the LLM issues a retrieve action
    thoughts = []
    observations = []

    for round_num in range(max_rounds):
        prompt = _build_react_prompt(question, answer, thoughts, observations)
        response = llm_call(
            prompt,
            system=REACT_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=2048,
        )
        thought, action = _parse_react_response(response)
        thoughts.append(thought)

        if action == "FINISH":
            # Try to parse structured chain from the final response
            chain_data = _parse_chain_from_response(response)
            if chain_data:
                steps = []
                for i, step_d in enumerate(chain_data):
                    try:
                        steps.append(ReasoningStep(
                            step=step_d.get("step", i + 1),
                            type=step_d.get("type", "knowledge_application"),
                            content=step_d.get("content", ""),
                            evidence=step_d.get("evidence"),
                            confidence=step_d.get("confidence", "medium"),
                        ))
                    except ValueError:
                        continue
                if steps:
                    return ReasoningChain(steps=steps, react_rounds=len(thoughts))

            # Fallback: build chain from accumulated thoughts
            return _build_chain_from_thoughts(question, answer, thoughts, observations)

        elif action.startswith("retrieve:"):
            query = action[len("retrieve:"):].strip()
            evidence = rag_tool.retrieve(query, intent="background", top_k=3)
            observations.append(evidence)
        else:
            # Unknown action, treat as FINISH
            observations.append([])

    # Max rounds reached without FINISH
    return _build_chain_from_thoughts(question, answer, thoughts, observations)
