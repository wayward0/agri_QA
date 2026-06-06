#!/usr/bin/env python3
"""Quantitative comparison: original AgThoughts reasoning vs pipeline enhanced reasoning.

Scores both on comparable dimensions, outputs per-sample side-by-side table.
"""

import json
import re
import sys
from collections import Counter

# ── Metric functions ──

def word_count(text: str) -> int:
    return len(text.split())

def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r'[.!?]+(?:\s|$)', text)))

def avg_sentence_length(text: str) -> float:
    sents = sentence_count(text)
    return word_count(text) / sents

def specificity_score(text: str) -> float:
    """Count named entities, numbers, units, proper nouns → normalize to 0-1."""
    # Numbers with units
    numbers = len(re.findall(r'\d+\s*(?:°[CF]|%|lbs?|oz|gal|inches?|ft|feet|cm|mm|pH|ppm|mg|kg|acres?|ha)', text, re.I))
    # Standalone numbers
    numbers += len(re.findall(r'\b\d+\.?\d*\b', text))
    # Capitalized words (potential proper nouns / named entities), exclude sentence starts
    caps = len(re.findall(r'(?<=[a-z.] )\b[A-Z][a-z]+\b', text))
    # Technical terms (chemical formulas, abbreviations)
    tech = len(re.findall(r'\b(?:NPK|Bt|pH|UAN|MAP|DAP|KCl|Ca|Mg|Fe|Zn|Mn|Cu|B|S)\b', text))
    raw = numbers * 2 + caps + tech * 3
    # Normalize: ~20+ raw points → score 1.0
    return min(1.0, raw / 20)

def evidence_score(text: str) -> float:
    """Mentions of sources, references, citations."""
    patterns = [
        r'(?i)(?:extension|university|USDA|NRCS|FAO|research|study|studies|publication|guide|handbook|fact\s*sheet)',
        r'(?i)(?:according to|published|journal|doi|pmid|ref\.?\s*\d)',
        r'(?i)(?:UMass|Purdue|Penn State|Cornell|Iowa State|NC State|UC Davis)',
    ]
    count = sum(len(re.findall(p, text)) for p in patterns)
    return min(1.0, count / 5)

def actionability_score(text: str) -> float:
    """Imperative verbs / concrete action phrases."""
    imperatives = re.findall(
        r'\b(?:apply|add|use|mix|spray|water|plant|test|check|remove|prune|stake|mulch|'
        r'rotate|monitor|scout|treat|fertilize|irrigate|drain|cover|space|thin|harvest|'
        r'incorporate|broadcast|side-dress|drench|soak|measure|adjust|maintain|install|'
        r'ensure|avoid|reduce|increase|select|choose|consider|confirm|identify)\b',
        text, re.I
    )
    # Normalize: ~15+ imperatives → 1.0
    return min(1.0, len(imperatives) / 15)

def coherence_score(text: str) -> float:
    """Transition words and logical connectors."""
    transitions = re.findall(
        r'\b(?:therefore|however|additionally|furthermore|moreover|consequently|'
        r'thus|hence|because|since|although|while|whereas|first|second|third|'
        r'next|then|finally|also|importantly|specifically|for example|in summary|'
        r'overall|alternatively|otherwise|meanwhile|subsequently)\b',
        text, re.I
    )
    return min(1.0, len(transitions) / 8)

def structure_score(text: str) -> float:
    """Headers, numbered lists, bullet points, sections."""
    headers = len(re.findall(r'^#{1,3}\s', text, re.M))
    numbered = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.M))
    bullets = len(re.findall(r'^\s*[-*•]\s', text, re.M))
    bold_sections = len(re.findall(r'\*\*[^*]+\*\*', text))
    raw = headers * 3 + numbered + bullets + bold_sections
    return min(1.0, raw / 15)

def constraint_coverage(text: str, question: str) -> float:
    """Check how many question constraints/details are addressed in the reasoning."""
    # Extract key constraint words from question
    q_lower = question.lower()
    constraints = []
    # Location
    locations = re.findall(r'\b(?:in|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', question)
    constraints.extend(locations)
    # Practices
    practices = re.findall(r'\b(?:organic|conventional|no-till|raised.bed|greenhouse|hydroponic)\b', q_lower)
    constraints.extend(practices)
    # Crop/subject
    crops = re.findall(r'\b(?:peppers?|tomatoes?|corn|wheat|soybean|potato|blueberry|carrot|kale|collard|beet|bean|flower|pistachio|wheat)\b', q_lower)
    constraints.extend(crops)
    # Conditions
    conditions = re.findall(r'\b(?:drought|heat|hail|wind|frost|erosion|lodging|pest|disease|mold|flood)\b', q_lower)
    constraints.extend(conditions)
    # Specific numbers
    numbers = re.findall(r'\b\d+\b', question)
    constraints.extend(numbers)

    if not constraints:
        return 0.5  # No constraints to check

    t_lower = text.lower()
    covered = sum(1 for c in constraints if c.lower() in t_lower)
    return covered / len(constraints)


def extract_enhanced_text(item: dict) -> str:
    """Extract full text from enhanced draft_chain."""
    chain = item.get("draft_chain", {})
    parts = []
    for step in chain.get("steps", []):
        parts.append(step.get("content", ""))
        ev = step.get("evidence", "")
        if ev:
            parts.append(ev)
    return " ".join(parts)


def score_text(text: str, question: str = "") -> dict:
    """Compute all metrics for a text."""
    return {
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "avg_sent_len": round(avg_sentence_length(text), 1),
        "specificity": round(specificity_score(text), 3),
        "evidence": round(evidence_score(text), 3),
        "actionability": round(actionability_score(text), 3),
        "coherence": round(coherence_score(text), 3),
        "structure": round(structure_score(text), 3),
        "constraint_coverage": round(constraint_coverage(text, question), 3) if question else 0.0,
    }


def weighted_overall(s: dict) -> float:
    """Weighted overall score (0-1)."""
    weights = {
        "specificity": 0.15,
        "evidence": 0.20,
        "actionability": 0.15,
        "coherence": 0.15,
        "structure": 0.10,
        "constraint_coverage": 0.25,
    }
    return round(sum(s[k] * w for k, w in weights.items()), 3)


def main():
    with open("AgThoughts.json", "r") as f:
        originals = {o["Question"].strip(): o for o in json.load(f)}

    with open("output/enhanced_dataset.json", "r") as f:
        enhanced = json.load(f)

    results = []
    for item in enhanced:
        q = item["question"].strip()
        orig = originals.get(q)
        if not orig:
            print(f"WARNING: no match for '{q[:60]}...'", file=sys.stderr)
            continue

        orig_text = orig["Reasoning Traces"]
        enh_text = extract_enhanced_text(item)

        orig_scores = score_text(orig_text, q)
        enh_scores = score_text(enh_text, q)
        orig_scores["overall"] = weighted_overall(orig_scores)
        enh_scores["overall"] = weighted_overall(enh_scores)

        results.append({
            "question": q,
            "difficulty": item.get("difficulty", "?"),
            "original": orig_scores,
            "enhanced": enh_scores,
        })

    # ── Print per-sample comparison ──
    metrics = ["word_count", "sentence_count", "specificity", "evidence",
               "actionability", "coherence", "structure", "constraint_coverage", "overall"]

    print("=" * 120)
    print("QUANTITATIVE COMPARISON: Original AgThoughts vs Pipeline Enhanced Reasoning")
    print("=" * 120)

    for r in results:
        print(f"\n{'─' * 120}")
        print(f"Q: {r['question'][:100]}...")
        print(f"Difficulty: {r['difficulty']}")
        print(f"{'─' * 120}")
        print(f"{'Metric':<25} {'Original':>12} {'Enhanced':>12} {'Delta':>12} {'Winner':>10}")
        print(f"{'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10}")
        for m in metrics:
            ov = r["original"][m]
            ev = r["enhanced"][m]
            delta = ev - ov
            if m == "word_count" or m == "sentence_count":
                # For these, more isn't always better — show raw
                winner = "orig" if ov > ev else ("enh" if ev > ov else "tie")
            else:
                winner = "enh" if ev > ov else ("orig" if ov > ev else "tie")
            sign = "+" if delta > 0 else ""
            print(f"{m:<25} {ov:>12} {ev:>12} {sign}{delta:>11} {winner:>10}")

    # ── Aggregate summary ──
    print(f"\n{'=' * 120}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 120}")
    print(f"{'Metric':<25} {'Orig Avg':>12} {'Enh Avg':>12} {'Delta':>12}")
    print(f"{'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")
    for m in metrics:
        orig_avg = sum(r["original"][m] for r in results) / len(results)
        enh_avg = sum(r["enhanced"][m] for r in results) / len(results)
        delta = enh_avg - orig_avg
        sign = "+" if delta > 0 else ""
        print(f"{m:<25} {orig_avg:>12.3f} {enh_avg:>12.3f} {sign}{delta:>11.3f}")

    # Win counts
    print(f"\n{'─' * 60}")
    enh_wins = sum(1 for r in results if r["enhanced"]["overall"] > r["original"]["overall"])
    orig_wins = sum(1 for r in results if r["original"]["overall"] > r["enhanced"]["overall"])
    ties = len(results) - enh_wins - orig_wins
    print(f"Overall score wins: Enhanced={enh_wins}, Original={orig_wins}, Tie={ties}")


if __name__ == "__main__":
    main()
