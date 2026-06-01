"""Tests for kg_translator module."""
import pytest
import pandas as pd
from modules.kg_translator import build_translation_prompt, validate_translation, has_chinese


def test_build_translation_prompt_contains_fields():
    row = pd.Series({
        "名称": "香蕉冠腐病",
        "英文名": "Banana crown rot",
        "为害症状": "症状描述",
        "发生规律": "发生规律描述",
        "防治": "防治方法",
    })
    prompt = build_translation_prompt(row)
    assert "香蕉冠腐病" in prompt
    assert "Banana crown rot" in prompt
    assert "症状描述" in prompt


def test_has_chinese_detects_chinese():
    assert has_chinese("这是中文") is True
    assert has_chinese("This is English") is False
    assert has_chinese("mixed 混合 text") is True
    assert has_chinese("") is False


def test_validate_translation_passes_good():
    result = {
        "name_en": "Banana crown rot",
        "symptoms": "The pathogen causes crown rot in bananas...",
        "occurrence": "The fungus enters through wounds...",
        "prevention": "Post-harvest treatment with Bacillus subtilis...",
    }
    assert validate_translation(result) is True


def test_validate_translation_fails_empty():
    result = {"name_en": "", "symptoms": "", "occurrence": "", "prevention": ""}
    assert validate_translation(result) is False


def test_validate_translation_fails_chinese残留():
    result = {
        "name_en": "Banana crown rot",
        "symptoms": "这是中文症状",
        "occurrence": "The fungus enters...",
        "prevention": "Treatment...",
    }
    assert validate_translation(result) is False
