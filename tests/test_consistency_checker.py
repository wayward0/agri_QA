"""Tests for consistency_checker module."""
import pytest
from modules.consistency_checker import (
    check_consistency,
    select_best,
)


def test_check_consistency_all_agree():
    samples = [
        {"diagnosis": "Early blight", "prevention_plan": "Fungicide A"},
        {"diagnosis": "Early blight", "prevention_plan": "Fungicide A"},
        {"diagnosis": "Early blight", "prevention_plan": "Fungicide B"},
    ]
    result = check_consistency(samples)
    assert result["agreement_count"] == 3
    assert result["status"] == "high_confidence"


def test_check_consistency_split():
    samples = [
        {"diagnosis": "Early blight", "prevention_plan": "A"},
        {"diagnosis": "Late blight", "prevention_plan": "B"},
        {"diagnosis": "Leaf spot", "prevention_plan": "C"},
    ]
    result = check_consistency(samples)
    assert result["status"] == "low_confidence"


def test_select_best_prefers_consistent():
    samples = [
        {"diagnosis": "Early blight", "prevention_plan": "A", "raw_xml": "<xml1>"},
        {"diagnosis": "Late blight", "prevention_plan": "B", "raw_xml": "<xml2>"},
        {"diagnosis": "Early blight", "prevention_plan": "A", "raw_xml": "<xml3>"},
    ]
    best = select_best(samples)
    assert best["diagnosis"] == "Early blight"


def test_select_best_single_sample():
    samples = [{"diagnosis": "X", "prevention_plan": "Y", "raw_xml": "<x>"}]
    best = select_best(samples)
    assert best["diagnosis"] == "X"
