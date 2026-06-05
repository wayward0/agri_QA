"""Difficulty classifier — Stage 0 of the pipeline.

1 LLM call to classify a QA pair as easy/medium/hard.
Dependencies: types only. Receives llm_call as parameter.
"""

import re
from models import ClassificationResult, DifficultyLevel


SYSTEM_PROMPT = "You are an agricultural question difficulty classifier."

CLASSIFICATION_PROMPT = """Classify the difficulty of this agricultural QA pair.

Question: {question}
Answer: {answer}

Criteria:
- EASY: Single-topic, factual, direct answer.
  Example: "What is the optimal planting depth for corn?"
- MEDIUM: Multiple aspects but focused, requires some domain knowledge.
  Example: "How to treat tomato blight in humid conditions?"
- HARD: Multi-part question, conditional reasoning, requires regional
  context or practical trade-offs.
  Example: "What issues affect my collards AND how can I prevent
  silverspotted skipper infestations next year in New Jersey?"

Output ONLY one word: easy OR medium OR hard"""


def classify_difficulty(
    question: str,
    answer: str,
    llm_call,
) -> ClassificationResult:
    """Classify QA pair difficulty.

    Args:
        question: The agricultural question.
        answer: The answer text.
        llm_call: Callable matching LLMCallFn protocol.

    Returns:
        ClassificationResult with difficulty level.
    """
    prompt = CLASSIFICATION_PROMPT.format(question=question, answer=answer)
    raw = llm_call(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=512)
    cleaned = raw.strip().lower()

    # Extract the last occurrence of a difficulty label (handles "not hard, it's easy" cases)
    # Use word boundary matching to avoid false positives
    easy_match = re.search(r'\beasy\b', cleaned)
    medium_match = re.search(r'\bmedium\b', cleaned)
    hard_match = re.search(r'\bhard\b', cleaned)

    # Priority: take the LAST match (LLM often gives reasoning then final answer)
    matches = []
    if easy_match:
        matches.append(('easy', easy_match.start()))
    if medium_match:
        matches.append(('medium', medium_match.start()))
    if hard_match:
        matches.append(('hard', hard_match.start()))

    if matches:
        # Use the last match as the final answer
        label = max(matches, key=lambda x: x[1])[0]
        difficulty = DifficultyLevel(label)
    else:
        difficulty = DifficultyLevel.EASY

    return ClassificationResult(difficulty=difficulty, raw_response=raw)
