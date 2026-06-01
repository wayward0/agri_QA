"""Build and persist FAISS indices for KG retrieval and entity recognition."""
import os
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import config


def build_text_for_kg_entry(row: dict) -> str:
    """Concatenate KG fields into a single text for embedding."""
    parts = [
        f"Name: {row.get('name_en', '')}",
        f"Symptoms: {row.get('symptoms', '')}",
        f"Occurrence: {row.get('occurrence', '')}",
        f"Prevention: {row.get('prevention', '')}",
    ]
    return " | ".join(p for p in parts if len(p) > 10)


def build_text_for_entity(row: dict) -> str:
    """Build text for entity index entry."""
    return f"{row.get('name_en', '')} ({row.get('name_cn', '')})"


class FaissIndex:
    """FAISS index with save/load and search."""

    def __init__(self, dim: int = config.EMBEDDING_DIM, index_dir: str = config.FAISS_INDEX_DIR):
        self.dim = dim
        self.index_dir = index_dir
        self.index = None
        self.texts = []

    def build(self, vectors: np.ndarray, texts: list[str]):
        """Build index from vectors and associated texts."""
        assert vectors.shape[1] == self.dim, f"Expected dim {self.dim}, got {vectors.shape[1]}"
        assert vectors.shape[0] == len(texts)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.texts = texts

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Search index. Returns list of (text, score) tuples."""
        faiss.normalize_L2(query_vectors)
        scores, indices = self.index.search(query_vectors, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
        return results

    def save(self, name: str):
        """Save index and texts to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_dir, f"{name}.faiss"))
        with open(os.path.join(self.index_dir, f"{name}_texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

    def load(self, name: str):
        """Load index and texts from disk."""
        self.index = faiss.read_index(os.path.join(self.index_dir, f"{name}.faiss"))
        with open(os.path.join(self.index_dir, f"{name}_texts.json"), "r", encoding="utf-8") as f:
            self.texts = json.load(f)


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def build_kg_index(kg_en_path: str, index_dir: str) -> FaissIndex:
    """Build FAISS index from translated KG CSV."""
    df = pd.read_csv(kg_en_path)
    model = load_embedding_model()
    texts = [build_text_for_kg_entry(row) for _, row in df.iterrows()]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index = FaissIndex(index_dir=index_dir)
    index.build(vectors, texts)
    index.save("kg_index")
    return index


def build_entity_index(kg_en_path: str, index_dir: str) -> FaissIndex:
    """Build FAISS index for entity recognition from KG entity names."""
    df = pd.read_csv(kg_en_path)
    model = load_embedding_model()
    texts = [build_text_for_entity(row) for _, row in df.iterrows()]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index = FaissIndex(index_dir=index_dir)
    index.build(vectors, texts)
    index.save("entity_index")
    return index


def build_all_indices(kg_en_path: str = config.PATH_CROPDP_KG_EN):
    """Build both kg_index and entity_index."""
    print("Building KG index...")
    build_kg_index(kg_en_path, config.FAISS_INDEX_DIR)
    print("Building entity index...")
    build_entity_index(kg_en_path, config.FAISS_INDEX_DIR)
    print("Done. Indices saved to", config.FAISS_INDEX_DIR)


if __name__ == "__main__":
    build_all_indices()
