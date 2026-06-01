"""Tests for entity_recognizer module."""
import pytest
from modules.entity_recognizer import (
    build_term_dict,
    keyword_match,
    merge_entities,
)


def test_build_term_dict_from_kg():
    rows = [
        {"name_en": "Banana crown rot", "name_cn": "香蕉冠腐病", "symptoms": "..."},
        {"name_en": "Aphid", "name_cn": "蚜虫", "symptoms": "..."},
    ]
    term_dict = build_term_dict(rows)
    assert "banana crown rot" in term_dict
    assert "aphid" in term_dict
    assert term_dict["banana crown rot"]["cn"] == "香蕉冠腐病"


def test_keyword_match_finds_entity():
    term_dict = {
        "banana crown rot": {"cn": "香蕉冠腐病", "type": "disease"},
        "aphid": {"cn": "蚜虫", "type": "pest"},
    }
    question = "My banana plants have crown rot symptoms"
    matches = keyword_match(question, term_dict)
    assert len(matches) == 1
    assert matches[0]["en"] == "banana crown rot"


def test_keyword_match_case_insensitive():
    term_dict = {"aphid": {"cn": "蚜虫", "type": "pest"}}
    question = "How to control APHID infestations?"
    matches = keyword_match(question, term_dict)
    assert len(matches) == 1


def test_merge_entities_deduplicates():
    entities = [
        {"en": "aphid", "cn": "蚜虫", "type": "pest", "confidence": 0.9},
        {"en": "aphid", "cn": "蚜虫", "type": "pest", "confidence": 0.7},
        {"en": "blight", "cn": "枯萎病", "type": "disease", "confidence": 0.8},
    ]
    merged = merge_entities(entities)
    assert len(merged) == 2
    aphid = next(e for e in merged if e["en"] == "aphid")
    assert aphid["confidence"] == 0.9
