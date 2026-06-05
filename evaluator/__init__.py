from .auto_metrics import compute_auto_metrics
from .llm_judge import judge_faithfulness, judge_logical_completeness

__all__ = ["compute_auto_metrics", "judge_faithfulness", "judge_logical_completeness"]


def evaluate_chain(chain, llm_call, weights=None):
    """Convenience wrapper: combine auto metrics + LLM judge into QualityScores."""
    from models import QualityScores

    auto = compute_auto_metrics(chain)
    faith, _ = judge_faithfulness(chain, llm_call)
    logic, _ = judge_logical_completeness(chain, llm_call)

    w = weights or {
        "faithfulness": 0.25,
        "structure": 0.20,
        "information_density": 0.15,
        "logical_completeness": 0.25,
        "traceability": 0.15,
    }
    overall = (
        faith / 5 * w["faithfulness"]
        + auto["structure"] * w["structure"]
        + auto["information_density"] * w["information_density"]
        + logic / 5 * w["logical_completeness"]
        + auto["traceability"] * w["traceability"]
    )

    return QualityScores(
        faithfulness=faith,
        logical_completeness=logic,
        overall=round(overall, 3),
        **auto,
    )
