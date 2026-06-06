"""Knowledge Graph builder — offline entity/relation extraction from Wikipedia.

Uses LLM to extract agricultural entities and relationships from passages,
grouped by article for context coherence.

Output: kg_entities.json, kg_relations.json, entity_faiss.index
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import faiss
import numpy as np


ENTITY_TYPES = ("crop", "disease", "pest", "chemical", "practice", "nutrient", "soil", "climate")

RELATION_TYPES = (
    "affects", "causes", "treats", "prevents",
    "requires", "interacts_with", "found_in", "applied_to",
)

EXTRACT_SYSTEM_PROMPT = "You are an agricultural knowledge graph extraction expert."

EXTRACT_PROMPT = """Extract agricultural entities and relationships from these Wikipedia passages.

ENTITY TYPES: {entity_types}
RELATION TYPES: {relation_types}

PASSAGES:
{passages_text}

RULES:
1. Entity names should be lowercase, normalized (e.g. "fusarium wilt" not "Fusarium Wilt Disease")
2. Include common aliases in the aliases array
3. Description should be 1 sentence summarizing the entity
4. Relations must reference entities defined in the same output
5. Confidence: 0.0-1.0 based on how explicitly the passage states the relation
6. Only extract relations that are explicitly stated or strongly implied

Output ONLY valid JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "aliases": ["..."], "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "relation": "...", "confidence": 0.9}}
  ]
}}

If no entities found, output: {{"entities": [], "relations": []}}"""


def _parse_extraction(raw: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse LLM extraction output into entities and relations."""
    # Try to find JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    # Try direct JSON parse
    json_match = re.search(r'\{[^{}]*"entities"\s*:\s*\[.*?\]\s*,\s*"relations"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)

    if not json_match:
        return [], []

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return [], []

    entities = []
    for e in data.get("entities", []):
        if not isinstance(e, dict) or "name" not in e:
            continue
        etype = e.get("type", "").lower()
        if etype not in ENTITY_TYPES:
            continue
        entities.append({
            "name": e["name"].strip().lower(),
            "type": etype,
            "aliases": [a.strip().lower() for a in e.get("aliases", []) if a],
            "description": e.get("description", "").strip(),
        })

    relations = []
    valid_names = {e["name"] for e in entities}
    for r in data.get("relations", []):
        if not isinstance(r, dict):
            continue
        src = r.get("source", "").strip().lower()
        tgt = r.get("target", "").strip().lower()
        rel = r.get("relation", "").strip().lower()
        if rel not in RELATION_TYPES:
            continue
        # Allow relations even if entities aren't in this batch (they may be in another article)
        if not src or not tgt:
            continue
        relations.append({
            "source": src,
            "target": tgt,
            "relation": rel,
            "confidence": min(1.0, max(0.0, float(r.get("confidence", 0.5)))),
        })

    return entities, relations


def _group_passages_by_article(passages: List[Dict]) -> Dict[str, List[Dict]]:
    """Group passages by article_title."""
    groups = defaultdict(list)
    for p in passages:
        groups[p["article_title"]].append(p)
    return dict(groups)


def _build_passages_text(passages: List[Dict], max_chars: int = 6000) -> str:
    """Format passages for the extraction prompt, with truncation."""
    parts = []
    total = 0
    for p in passages:
        text = p["text"][:500]
        section = p.get("section_title", "Main")
        entry = f"[{section}] {text}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _extract_single_article(article: Dict, llm_call) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from a single article.

    Returns (entities, relations) with passage_ids and source_article tagged.
    """
    from .knowledge_builder import hierarchical_chunk

    passages = hierarchical_chunk([article])
    if not passages:
        return [], []

    passages_text = _build_passages_text(passages)
    if len(passages_text) < 100:
        return [], []

    prompt = EXTRACT_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relation_types=", ".join(RELATION_TYPES),
        passages_text=passages_text,
    )

    raw = llm_call(prompt, system=EXTRACT_SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
    entities, relations = _parse_extraction(raw)

    passage_ids = [p["id"] for p in passages]
    for ent in entities:
        ent["passage_ids"] = passage_ids
        ent["source_article"] = article["title"]
    for rel in relations:
        rel["evidence_passage_ids"] = passage_ids

    return entities, relations


def extract_kg_from_articles(
    articles: List[Dict],
    llm_call,
    batch_size: int = 1,
    max_workers: int = 5,
    checkpoint_path: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from articles using LLM (parallel).

    Args:
        articles: List of article dicts (from fetch_wikipedia_articles).
        llm_call: LLM call function.
        batch_size: Unused, kept for API compatibility.
        max_workers: Number of parallel LLM calls.
        checkpoint_path: If provided, save incremental results every 50 articles.

    Returns:
        (entities, relations) — raw, may contain duplicates.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import threading

    all_entities = []
    all_relations = []
    lock = threading.Lock()
    done_count = 0

    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path, "r") as f:
                ckpt = json.load(f)
            all_entities = ckpt.get("entities", [])
            all_relations = ckpt.get("relations", [])
            start_idx = ckpt.get("done_count", 0)
            print(f"  Resuming KG extraction from article {start_idx} "
                  f"({len(all_entities)} entities, {len(all_relations)} relations loaded)")
        except (json.JSONDecodeError, KeyError):
            pass

    remaining = articles[start_idx:]
    if not remaining:
        return all_entities, all_relations

    pbar = tqdm(total=len(remaining), desc="KG extraction", unit="article",
                initial=0)

    def _process_and_collect(idx: int, article: Dict):
        nonlocal done_count
        ents, rels = _extract_single_article(article, llm_call)
        with lock:
            all_entities.extend(ents)
            all_relations.extend(rels)
            done_count += 1
            pbar.update(1)
            # Incremental checkpoint
            if checkpoint_path and done_count % 50 == 0:
                _save_checkpoint(checkpoint_path, all_entities, all_relations,
                                 start_idx + done_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, article in enumerate(remaining):
            future = executor.submit(_process_and_collect, i, article)
            futures[future] = i

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                print(f"  Warning: article {start_idx + idx} failed: {e}")

    pbar.close()

    # Final checkpoint
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, all_entities, all_relations,
                         start_idx + len(remaining))

    return all_entities, all_relations


def _save_checkpoint(path: str, entities: List[Dict], relations: List[Dict],
                     done_count: int):
    """Save KG extraction checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": entities,
            "relations": relations,
            "done_count": done_count,
        }, f, ensure_ascii=False)


def load_kg_checkpoint(path: str) -> Tuple[List[Dict], List[Dict], int]:
    """Load KG checkpoint. Returns (entities, relations, done_count)."""
    if not Path(path).exists():
        return [], [], 0
    with open(path, "r") as f:
        ckpt = json.load(f)
    return ckpt.get("entities", []), ckpt.get("relations", []), ckpt.get("done_count", 0)


def merge_entities(entities: List[Dict]) -> List[Dict]:
    """Deduplicate entities by name, merge passage_ids and aliases."""
    by_name = {}
    for ent in entities:
        name = ent["name"]
        if name in by_name:
            existing = by_name[name]
            # Merge passage_ids
            existing["passage_ids"] = list(set(existing["passage_ids"] + ent.get("passage_ids", [])))
            # Merge aliases
            existing["aliases"] = list(set(existing["aliases"] + ent.get("aliases", [])))
            # Keep longer description
            if len(ent.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = ent["description"]
        else:
            by_name[name] = dict(ent)
            by_name[name]["passage_ids"] = list(ent.get("passage_ids", []))
            by_name[name]["aliases"] = list(ent.get("aliases", []))

    # Add sequential IDs
    result = []
    for i, (name, ent) in enumerate(sorted(by_name.items())):
        ent["id"] = f"entity_{i}"
        result.append(ent)

    return result


def merge_relations(relations: List[Dict]) -> List[Dict]:
    """Deduplicate relations by (source, target, relation), keep highest confidence."""
    by_key = {}
    for rel in relations:
        key = (rel["source"], rel["target"], rel["relation"])
        if key in by_key:
            existing = by_key[key]
            existing["confidence"] = max(existing["confidence"], rel["confidence"])
            existing["evidence_passage_ids"] = list(
                set(existing.get("evidence_passage_ids", []) + rel.get("evidence_passage_ids", []))
            )
        else:
            by_key[key] = dict(rel)

    return sorted(by_key.values(), key=lambda r: (-r["confidence"], r["source"]))


def build_entity_faiss(
    entities: List[Dict],
    embedding_model,
    dim: int = 1024,
    output_path: Optional[str] = None,
) -> faiss.Index:
    """Build FAISS index over entity names + aliases for vector matching."""
    # Build text representations for each entity
    texts = []
    entity_map = []  # maps FAISS index → entity index

    for i, ent in enumerate(entities):
        # Entity name
        texts.append(ent["name"])
        entity_map.append(i)
        # Aliases
        for alias in ent.get("aliases", []):
            if alias != ent["name"]:
                texts.append(alias)
                entity_map.append(i)

    if not texts:
        index = faiss.IndexFlatIP(dim)
        if output_path:
            faiss.write_index(index, output_path)
        return index

    # Encode
    from tqdm import tqdm
    pbar = tqdm(total=len(texts), desc="Encoding entities", unit="ent")
    embeddings = embedding_model.encode(texts, normalize_embeddings=True, progress_bar=pbar)
    pbar.close()
    embeddings = np.array(embeddings, dtype="float32")

    # Build index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if output_path:
        faiss.write_index(index, output_path)

    # Save entity_map alongside the index
    if output_path:
        map_path = output_path.replace(".index", "_map.json")
        with open(map_path, "w") as f:
            json.dump(entity_map, f)

    return index


def save_kg(
    entities: List[Dict],
    relations: List[Dict],
    entities_path: str,
    relations_path: str,
):
    """Save KG to JSON files."""
    Path(entities_path).parent.mkdir(parents=True, exist_ok=True)
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(entities)} entities, {len(relations)} relations")
