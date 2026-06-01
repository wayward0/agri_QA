"""Tests for AgThoughts domain filtering logic."""
import pytest
from typing import List


BIOTIC_CATEGORIES = [
    "Biotic Diseases Questions",
    "Biotic Insects Questions",
    "Biotic Weeds Questions",
]


def filter_agthroughts(items: List[dict], term_dict: dict) -> List[dict]:
    """Filter AgThoughts to disease/pest subset."""
    filtered = []
    for item in items:
        q_type = item.get("Question Type", "")
        if q_type in BIOTIC_CATEGORIES:
            filtered.append(item)
        elif q_type == "Plant and Seed Health Questions":
            question = item.get("Question", "").lower()
            if any(term in question for term in term_dict):
                filtered.append(item)
    return filtered


def test_biotic_always_included():
    items = [
        {"Question Type": "Biotic Diseases Questions", "Question": "What is this?"},
        {"Question Type": "Biotic Insects Questions", "Question": "Bug help"},
        {"Question Type": "Biotic Weeds Questions", "Question": "Weed control"},
    ]
    term_dict = {}
    result = filter_agthroughts(items, term_dict)
    assert len(result) == 3


def test_plant_health_filtered_by_term():
    items = [
        {"Question Type": "Plant and Seed Health Questions", "Question": "My tomatoes have early blight"},
        {"Question Type": "Plant and Seed Health Questions", "Question": "How to plant seeds?"},
    ]
    term_dict = {"early blight": {"cn": "早疫病", "type": "disease"}}
    result = filter_agthroughts(items, term_dict)
    assert len(result) == 1
    assert "early blight" in result[0]["Question"]


def test_other_categories_excluded():
    items = [
        {"Question Type": "Crop Management Questions", "Question": "When to harvest?"},
        {"Question Type": "Abiotic Weather Questions", "Question": "Frost damage?"},
    ]
    term_dict = {"frost": {"cn": "霜冻", "type": "disease"}}
    result = filter_agthroughts(items, term_dict)
    assert len(result) == 0
