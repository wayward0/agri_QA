"""Hybrid entity recognition: keyword matching + FAISS semantic completion."""
import re
import json
import numpy as np
import pandas as pd
from modules.faiss_builder import FaissIndex, load_embedding_model
import config


def build_term_dict(kg_rows: list) -> dict:
    """Build term dictionary from KG rows. Keys are lowercase English names."""
    term_dict = {}
    for row in kg_rows:
        name_en = str(row.get("name_en", "")).strip()
        name_cn = str(row.get("name_cn", "")).strip()
        if not name_en:
            continue
        key = name_en.lower()
        if key not in term_dict:
            term_dict[key] = {
                "cn": name_cn,
                "type": _classify_entity(row),
            }
    return term_dict


def _classify_entity(row: dict) -> str:
    """Classify entity type based on available data. Default to 'disease'."""
    name = str(row.get("name_en", "")).lower()
    symptoms = str(row.get("symptoms", "")).lower()
    if any(w in name for w in ["insect", "beetle", "worm", "aphid", "mite", "bug", "fly", "moth", "caterpillar"]):
        return "pest"
    if any(w in name for w in ["weed", "grass"]):
        return "weed"
    if any(w in symptoms for w in ["leaf spot", "rot", "blight", "mildew", "rust", "wilt", "mold"]):
        return "disease"
    return "disease"


def keyword_match(question: str, term_dict: dict) -> list:
    """Match entities by keyword substring (case-insensitive).

    For single-word terms, uses direct substring search.
    For multi-word terms, checks that all words appear in the question
    in order (allowing gaps between words).
    """
    text = question.lower()
    found = []
    matched_spans = set()

    for term in sorted(term_dict, key=len, reverse=True):
        # Try exact substring first
        idx = text.find(term)
        if idx != -1:
            span = set(range(idx, idx + len(term)))
            if not span & matched_spans:
                info = term_dict[term]
                found.append({
                    "en": term,
                    "cn": info["cn"],
                    "type": info["type"],
                    "confidence": 1.0,
                    "source": "keyword",
                })
                matched_spans |= span
            continue

        # For multi-word terms, check ordered word presence
        words = term.split()
        if len(words) < 2:
            continue

        search_from = 0
        all_found = True
        word_positions = []
        for w in words:
            pos = text.find(w, search_from)
            if pos == -1:
                all_found = False
                break
            word_positions.append((pos, pos + len(w)))
            search_from = pos + len(w)

        if all_found:
            # Check no overlapping span already matched
            span = set()
            for start, end in word_positions:
                span |= set(range(start, end))
            if not span & matched_spans:
                info = term_dict[term]
                found.append({
                    "en": term,
                    "cn": info["cn"],
                    "type": info["type"],
                    "confidence": 1.0,
                    "source": "keyword",
                })
                matched_spans |= span

    return found


def faiss_entity_search(
    question: str,
    entity_index: FaissIndex,
    model,
    top_k: int = 5,
    threshold: float = 0.7,
) -> list:
    """Semantic entity search via FAISS. Returns entities above confidence threshold."""
    embedding = model.encode([question], convert_to_numpy=True).astype(np.float32)
    results = entity_index.search(embedding, top_k=top_k)
    entities = []
    for text, score in results:
        if score >= threshold:
            match = re.match(r'^(.+?)\s*\((.+?)\)$', text)
            if match:
                en, cn = match.group(1).strip(), match.group(2).strip()
                entities.append({
                    "en": en.lower(),
                    "cn": cn,
                    "type": "disease",
                    "confidence": round(score, 3),
                    "source": "faiss",
                })
    return entities


def merge_entities(entities: list) -> list:
    """Deduplicate entities, keeping highest confidence per 'en' key."""
    best = {}
    for e in entities:
        key = e["en"]
        if key not in best or e["confidence"] > best[key]["confidence"]:
            best[key] = e
    return list(best.values())


def recognize_entities(
    question: str,
    term_dict: dict,
    entity_index: FaissIndex = None,
    model=None,
) -> list:
    """Full hybrid entity recognition pipeline."""
    kw_entities = keyword_match(question, term_dict)
    faiss_entities = []
    if entity_index is not None and model is not None:
        faiss_entities = faiss_entity_search(question, entity_index, model)
    all_entities = kw_entities + faiss_entities
    return merge_entities(all_entities)


def load_term_dict(path: str = config.PATH_TERM_DICT) -> dict:
    """Load term dict from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_term_dict(term_dict: dict, path: str = config.PATH_TERM_DICT):
    """Save term dict to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(term_dict, f, ensure_ascii=False, indent=2)


def build_and_save_term_dict(kg_en_path: str = config.PATH_CROPDP_KG_EN):
    """Build term dict from translated KG and save."""
    df = pd.read_csv(kg_en_path)
    rows = df.to_dict("records")
    term_dict = build_term_dict(rows)
    save_term_dict(term_dict)
    print(f"Term dict: {len(term_dict)} entries saved to {config.PATH_TERM_DICT}")
    return term_dict
