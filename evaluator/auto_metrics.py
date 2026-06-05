"""Automated quality metrics — pure functions, no external dependencies.

Metrics:
- structure_score: checks chain structure (context_setup, conclusion, step count)
- info_density: ratio of non-formulaic words to total words
- traceability: ratio of steps with evidence to total steps
"""

from models import ReasoningChain


def structure_score(chain: ReasoningChain) -> float:
    """Returns 0-1. Checks: has context_setup, has conclusion, step count 3-7."""
    has_context = any(s.type == "context_setup" for s in chain.steps)
    has_conclusion = any(s.type == "conclusion" for s in chain.steps)
    optimal_steps = 3 <= len(chain.steps) <= 7
    return (int(has_context) + int(has_conclusion) + int(optimal_steps)) / 3


def info_density(chain: ReasoningChain) -> float:
    """Returns 0-1. Ratio of non-formulaic words to total words."""
    formulaic = {"okay", "so", "let", "me", "i", "need", "to", "should", "right"}
    text = " ".join(s.content for s in chain.steps)
    words = text.lower().split()
    if not words:
        return 0.0
    filler = sum(1 for w in words if w in formulaic)
    return (len(words) - filler) / len(words)


def traceability(chain: ReasoningChain) -> float:
    """Returns 0-1. Ratio of steps with evidence to total steps."""
    if not chain.steps:
        return 0.0
    with_evidence = sum(1 for s in chain.steps if s.evidence)
    return with_evidence / len(chain.steps)


def compute_auto_metrics(chain: ReasoningChain) -> dict:
    """Compute all automated metrics. Returns dict with structure, information_density, traceability."""
    return {
        "structure": round(structure_score(chain), 3),
        "information_density": round(info_density(chain), 3),
        "traceability": round(traceability(chain), 3),
    }
