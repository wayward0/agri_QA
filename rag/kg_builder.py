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


def extract_kg_from_articles(
    articles: List[Dict],
    llm_call,
    batch_size: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from articles using LLM.

    Processes one article at a time for context coherence.

    Args:
        articles: List of article dicts (from fetch_wikipedia_articles).
        llm_call: LLM call function.
        batch_size: Articles per LLM call (currently 1 for quality).

    Returns:
        (entities, relations) — raw, may contain duplicates.
    """
    from .knowledge_builder import hierarchical_chunk

    all_entities = []
    all_relations = []

    for i, article in enumerate(articles):
        # Chunk the article
        passages = hierarchical_chunk([article])
        if not passages:
            continue

        # Build prompt
        passages_text = _build_passages_text(passages)
        if len(passages_text) < 100:
            continue

        prompt = EXTRACT_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            passages_text=passages_text,
        )

        raw = llm_call(prompt, system=EXTRACT_SYSTEM_PROMPT, temperature=0.1, max_tokens=2048)
        entities, relations = _parse_extraction(raw)

        # Tag each entity with passage_ids
        passage_ids = [p["id"] for p in passages]
        for ent in entities:
            ent["passage_ids"] = passage_ids
            ent["source_article"] = article["title"]

        for rel in relations:
            rel["evidence_passage_ids"] = passage_ids

        all_entities.extend(entities)
        all_relations.extend(relations)

        if (i + 1) % 10 == 0:
            print(f"  Extracted from {i+1}/{len(articles)} articles: "
                  f"{len(all_entities)} entities, {len(all_relations)} relations")

    return all_entities, all_relations


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
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
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
