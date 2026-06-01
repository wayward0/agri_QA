"""Tests for reasoning_generator module."""
import pytest
from modules.reasoning_generator import (
    build_reasoning_prompt,
    parse_xml_reasoning,
    extract_diagnosis,
)


def test_build_reasoning_prompt_contains_question():
    prompt = build_reasoning_prompt(
        question="My tomatoes have brown spots on leaves",
        kg_context="KG Entry 1: Early blight...",
        entities=[{"en": "tomato", "cn": "番茄", "type": "crop"}],
    )
    assert "brown spots" in prompt
    assert "Early blight" in prompt
    assert "reasoning_chain" in prompt


def test_parse_xml_reasoning_valid():
    xml_str = """<reasoning_chain>
  <question_analysis>
    <type>Disease identification</type>
    <crop>tomato</crop>
    <symptoms>Brown spots on leaves</symptoms>
  </question_analysis>
  <evidence_retrieval>
    <kg_match entity="Early blight" confidence="0.9">
      <symptoms>Brown spots with concentric rings</symptoms>
    </kg_match>
  </evidence_retrieval>
  <reasoning_steps>
    <step n="1">The symptoms match early blight.</step>
  </reasoning_steps>
  <conclusion>
    <diagnosis>Early blight</diagnosis>
    <prevention_plan>Apply fungicide, remove infected leaves</prevention_plan>
  </conclusion>
</reasoning_chain>"""
    result = parse_xml_reasoning(xml_str)
    assert result is not None
    assert result["diagnosis"] == "Early blight"
    assert "fungicide" in result["prevention_plan"]


def test_parse_xml_reasoning_invalid():
    result = parse_xml_reasoning("not xml at all")
    assert result is None


def test_extract_diagnosis():
    parsed = {"diagnosis": "Early blight", "prevention_plan": "Fungicide"}
    assert extract_diagnosis(parsed) == "Early blight"
    assert extract_diagnosis(None) == "Unknown"
