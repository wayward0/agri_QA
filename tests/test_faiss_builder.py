"""Tests for faiss_builder module."""
import pytest
import numpy as np
import os
import tempfile
from modules.faiss_builder import (
    build_text_for_kg_entry,
    build_text_for_entity,
    FaissIndex,
)


def test_build_text_for_kg_entry():
    row = {
        "name_en": "Banana crown rot",
        "symptoms": "Crown rot symptoms...",
        "occurrence": "Fungus enters through wounds...",
        "prevention": "Treatment with Bacillus...",
    }
    text = build_text_for_kg_entry(row)
    assert "Banana crown rot" in text
    assert "Crown rot symptoms" in text
    assert "Treatment with Bacillus" in text


def test_build_text_for_entity():
    row = {"name_en": "Banana crown rot", "name_cn": "香蕉冠腐病"}
    text = build_text_for_entity(row)
    assert "Banana crown rot" in text


def test_faiss_index_build_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = FaissIndex(dim=4, index_dir=tmpdir)
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        texts = ["apple disease", "corn pest", "soil management"]
        index.build(vectors, texts)
        assert index.index.ntotal == 3
        results = index.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert results[0][0] == "apple disease"


def test_faiss_index_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = FaissIndex(dim=4, index_dir=tmpdir)
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        texts = ["text a", "text b"]
        index.build(vectors, texts)
        index.save("test_index")

        loaded = FaissIndex(dim=4, index_dir=tmpdir)
        loaded.load("test_index")
        assert loaded.index.ntotal == 2
        results = loaded.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=1)
        assert results[0][0] == "text a"
