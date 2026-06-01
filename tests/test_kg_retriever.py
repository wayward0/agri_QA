"""Tests for kg_retriever module."""
import pytest
from modules.kg_retriever import (
    keyword_fallback_search,
    merge_kg_results,
    format_kg_context,
)


def test_keyword_fallback_search():
    kg_entries = [
        {"name_en": "Banana crown rot", "symptoms": "Crown rot...", "occurrence": "...", "prevention": "..."},
        {"name_en": "Aphid infestation", "symptoms": "Small insects...", "occurrence": "...", "prevention": "..."},
    ]
    entities = [{"en": "banana crown rot", "cn": "香蕉冠腐病", "type": "disease"}]
    results = keyword_fallback_search(entities, kg_entries)
    assert len(results) >= 1
    assert results[0]["name_en"] == "Banana crown rot"


def test_merge_kg_results_deduplicates():
    results = [
        {"name_en": "Banana crown rot", "symptoms": "A", "occurrence": "B", "prevention": "C", "score": 0.9},
        {"name_en": "Banana crown rot", "symptoms": "A", "occurrence": "B", "prevention": "C", "score": 0.8},
    ]
    merged = merge_kg_results(results)
    assert len(merged) == 1


def test_format_kg_context():
    entries = [
        {"name_en": "Banana crown rot", "symptoms": "Crown rot symptoms.", "occurrence": "Fungus.", "prevention": "Treatment."},
    ]
    ctx = format_kg_context(entries)
    assert "Banana crown rot" in ctx
    assert "Crown rot symptoms" in ctx
    assert "Treatment" in ctx
