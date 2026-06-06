#!/usr/bin/env python3
"""Rebuild FAISS + BM25 indices + Knowledge Graph from fetched articles.

Usage:
    python rebuild_indices.py            # Full rebuild (all 9 stages)
    python rebuild_indices.py --kg-only  # Only rebuild KG (stages 7-9), skip indices
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path
import config
from rag.embedding_client import EmbeddingClient
from rag.knowledge_builder import (
    hierarchical_chunk, build_faiss_index, build_bm25_index,
    save_metadata, save_passages_jsonl, save_articles,
)
from rag.kg_builder import (
    extract_kg_from_articles, merge_entities, merge_relations,
    build_entity_faiss, save_kg, load_kg_checkpoint,
)


def _stage(n: int, total: int, msg: str):
    """Print a stage progress line."""
    print(f"\n[{n}/{total}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild knowledge base indices")
    parser.add_argument("--kg-only", action="store_true",
                        help="Only rebuild Knowledge Graph (skip FAISS/BM25/metadata)")
    args = parser.parse_args()

    out = Path("data")
    raw_path = out / "raw" / "articles.json"
    ckpt_path = str(out / "index" / "kg_checkpoint.json")

    if args.kg_only:
        # KG-only mode: skip stages 1-6
        STAGES = 3

        _stage(1, STAGES, "Loading articles...")
        with open(raw_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"  {len(articles)} articles")

        _stage(2, STAGES, "Extracting Knowledge Graph (parallel, 10 workers)...")
        from llm_client import create_llm_caller
        kg_llm = create_llm_caller(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
            model=config.MODEL_LIGHT,
        )
        raw_entities, raw_relations = extract_kg_from_articles(
            articles, kg_llm, max_workers=10, checkpoint_path=ckpt_path,
        )
        print(f"  Raw: {len(raw_entities)} entities, {len(raw_relations)} relations")

        _stage(3, STAGES, "Merging, saving KG + entity FAISS...")
        entities = merge_entities(raw_entities)
        relations = merge_relations(raw_relations)
        print(f"  Merged: {len(entities)} entities, {len(relations)} relations")
        (out / "index").mkdir(parents=True, exist_ok=True)
        save_kg(entities, relations,
                 str(out / "index" / "kg_entities.json"),
                 str(out / "index" / "kg_relations.json"))

        embedding_model = EmbeddingClient(
            base_url=config.EMBEDDING_API_BASE_URL,
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
        )
        build_entity_faiss(entities, embedding_model, config.EMBEDDING_DIM,
                           str(out / "index" / "entity_faiss.index"))

        # Clean up checkpoint
        Path(ckpt_path).unlink(missing_ok=True)

        print(f"\n{'='*60}")
        print(f"Done. Knowledge Graph rebuilt.")
        print(f"  KG: {len(entities)} entities, {len(relations)} relations")
        print(f"{'='*60}", flush=True)
        return

    # Full rebuild
    STAGES = 9

    _stage(1, STAGES, "Loading articles...")
    with open(raw_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    total_chars = sum(len(a.get("content", "")) for a in articles)
    total_mb = total_chars / (1024 * 1024)
    print(f"  {len(articles)} articles, {total_mb:.1f} MB")

    _stage(2, STAGES, "Chunking articles...")
    passages = hierarchical_chunk(articles)
    print(f"  {len(passages)} passages")

    _stage(3, STAGES, "Saving passages JSONL...")
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    save_passages_jsonl(passages, str(out / "chunks" / "passages.jsonl"))

    _stage(4, STAGES, f"Building FAISS index (embedding={config.EMBEDDING_MODEL_NAME}, dim={config.EMBEDDING_DIM})...")
    embedding_model = EmbeddingClient(
        base_url=config.EMBEDDING_API_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
    )
    (out / "index").mkdir(parents=True, exist_ok=True)
    build_faiss_index(passages, embedding_model, config.EMBEDDING_DIM, str(out / "index" / "faiss.index"))

    _stage(5, STAGES, "Building BM25 index...")
    build_bm25_index(passages, output_path=str(out / "index" / "bm25.pkl"))

    _stage(6, STAGES, "Saving metadata...")
    save_metadata(passages, str(out / "index" / "metadata.json"))

    _stage(7, STAGES, "Extracting Knowledge Graph (parallel, 10 workers)...")
    from llm_client import create_llm_caller
    kg_llm = create_llm_caller(
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        model=config.MODEL_LIGHT,
    )
    raw_entities, raw_relations = extract_kg_from_articles(
        articles, kg_llm, max_workers=10, checkpoint_path=ckpt_path,
    )
    print(f"  Raw: {len(raw_entities)} entities, {len(raw_relations)} relations")

    _stage(8, STAGES, "Merging and saving Knowledge Graph...")
    entities = merge_entities(raw_entities)
    relations = merge_relations(raw_relations)
    print(f"  Merged: {len(entities)} entities, {len(relations)} relations")
    save_kg(entities, relations,
             str(out / "index" / "kg_entities.json"),
             str(out / "index" / "kg_relations.json"))

    _stage(9, STAGES, "Building entity FAISS index...")
    build_entity_faiss(entities, embedding_model, config.EMBEDDING_DIM,
                       str(out / "index" / "entity_faiss.index"))

    # Clean up checkpoint
    Path(ckpt_path).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"Done. Knowledge base rebuilt.")
    print(f"  Articles:  {len(articles)}")
    print(f"  Passages:  {len(passages)}")
    print(f"  Content:   {total_mb:.1f} MB")
    print(f"  Embedding: {config.EMBEDDING_MODEL_NAME} ({config.EMBEDDING_DIM}d)")
    print(f"  KG:        {len(entities)} entities, {len(relations)} relations")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
