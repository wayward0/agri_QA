"""Tests for verifier module."""
import pytest
from modules.verifier import (
    build_verification_prompt,
    parse_verification_result,
)


def test_build_verification_prompt_contains_claims():
    prompt = build_verification_prompt(
        reasoning_xml="<reasoning_chain>...</reasoning_chain>",
        kg_context="KG Entry 1: Early blight symptoms...",
    )
    assert "reasoning_chain" in prompt
    assert "Early blight" in prompt


def test_parse_verification_result_verified():
    raw = """<verification>
  <status>verified</status>
  <contradictions>none</contradictions>
  <unsupported_claims>none</unsupported_claims>
</verification>"""
    result = parse_verification_result(raw)
    assert result["status"] == "verified"
    assert result["contradictions"] == "none"


def test_parse_verification_result_contradicted():
    raw = """<verification>
  <status>contradicted</status>
  <contradictions>The reasoning states late blight but KG describes early blight</contradictions>
  <unsupported_claims>none</unsupported_claims>
</verification>"""
    result = parse_verification_result(raw)
    assert result["status"] == "contradicted"
    assert "late blight" in result["contradictions"]


def test_parse_verification_result_malformed():
    result = parse_verification_result("not xml")
    assert result["status"] == "parse_failed"
