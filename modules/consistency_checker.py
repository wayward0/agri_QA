"""Self-consistency checker: sample N times, pick the most consistent result."""
from collections import Counter
from typing import List, Dict, Tuple
from modules.reasoning_generator import generate_reasoning
import config


def check_consistency(samples: List[dict]) -> dict:
    """Check diagnosis consistency across N samples."""
    diagnoses = [s.get("diagnosis", "Unknown") for s in samples]
    counter = Counter(diagnoses)
    most_common_diagnosis, count = counter.most_common(1)[0]
    total = len(diagnoses)

    if count >= total:
        status = "high_confidence"
    elif count >= total * 0.6:
        status = "medium_confidence"
    else:
        status = "low_confidence"

    return {
        "most_common_diagnosis": most_common_diagnosis,
        "agreement_count": count,
        "total_samples": total,
        "status": status,
        "all_diagnoses": diagnoses,
    }


def select_best(samples: List[dict]) -> dict:
    """Select the best sample: prefer majority diagnosis, then first sample."""
    if len(samples) == 1:
        return samples[0]

    diagnoses = [s.get("diagnosis", "Unknown") for s in samples]
    counter = Counter(diagnoses)
    best_diagnosis, _ = counter.most_common(1)[0]

    # Return first sample with the majority diagnosis
    for s in samples:
        if s.get("diagnosis") == best_diagnosis:
            return s
    return samples[0]


def run_self_consistency(
    question: str,
    kg_context: str,
    entities: List[dict],
    client=None,
    n_samples: int = None,
) -> Tuple[dict, dict]:
    """Generate N samples and return (best_result, consistency_report)."""
    from modules.reasoning_generator import get_client
    if client is None:
        client = get_client()
    if n_samples is None:
        n_samples = config.SELF_CONSISTENCY_N

    samples = []
    for i in range(n_samples):
        raw, parsed = generate_reasoning(
            question, kg_context, entities,
            client=client,
            temperature=config.TEMPERATURE_SAMPLE,
        )
        if parsed:
            samples.append({**parsed, "raw_xml": raw, "sample_idx": i})

    if not samples:
        # All samples failed to parse -- try once with lower temperature
        raw, parsed = generate_reasoning(
            question, kg_context, entities,
            client=client,
            temperature=config.TEMPERATURE_GENERATE,
        )
        if parsed:
            samples.append({**parsed, "raw_xml": raw, "sample_idx": 0})
        else:
            return {"diagnosis": "Unknown", "raw_xml": raw}, {"status": "all_failed"}

    consistency = check_consistency(samples)
    best = select_best(samples)
    best["consistency"] = consistency
    return best, consistency
