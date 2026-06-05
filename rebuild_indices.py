#!/usr/bin/env python3
"""Rebuild FAISS + BM25 indices from fetched articles using bge-m3 API."""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
import config
from rag.embedding_client import EmbeddingClient
from rag.knowledge_builder import (
    hierarchical_chunk, build_faiss_index, build_bm25_index,
    save_metadata, save_passages_jsonl, save_articles,
)


def main():
    out = Path("data")

    # Load articles
    raw_path = out / "raw" / "articles.json"
    print(f"Loading articles from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles.")

    # Stats
    total_chars = sum(len(a.get("content", "")) for a in articles)
    total_mb = total_chars / (1024 * 1024)
    print(f"Total content size: {total_mb:.1f} MB")

    # Chunk
    print("Chunking articles...")
    passages = hierarchical_chunk(articles)
    print(f"Created {len(passages)} passages.")

    # Save passages
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    save_passages_jsonl(passages, str(out / "chunks" / "passages.jsonl"))

    # Init embedding client
    print(f"Initializing embedding client: {config.EMBEDDING_MODEL_NAME}")
    embedding_model = EmbeddingClient(
        base_url=config.EMBEDDING_API_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
    )

    # Build indices
    (out / "index").mkdir(parents=True, exist_ok=True)

    print(f"Building FAISS index (dim={config.EMBEDDING_DIM})...")
    build_faiss_index(passages, embedding_model, config.EMBEDDING_DIM, str(out / "index" / "faiss.index"))
    print("FAISS index built.")

    print("Building BM25 index...")
    build_bm25_index(passages, output_path=str(out / "index" / "bm25.pkl"))
    print("BM25 index built.")

    print("Saving metadata...")
    save_metadata(passages, str(out / "index" / "metadata.json"))

    print("Done. Knowledge base rebuilt.")
    print(f"  Articles: {len(articles)}")
    print(f"  Passages: {len(passages)}")
    print(f"  Content: {total_mb:.1f} MB")
    print(f"  Embedding: {config.EMBEDDING_MODEL_NAME} ({config.EMBEDDING_DIM}d)")


if __name__ == "__main__":
    main()
