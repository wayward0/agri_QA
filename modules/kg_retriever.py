"""KG retrieval: FAISS top-k + keyword fallback."""
import numpy as np
import pandas as pd
from modules.faiss_builder import FaissIndex, load_embedding_model
import config


def faiss_kg_search(
    question: str,
    kg_index: FaissIndex,
    model,
    top_k: int = config.FAISS_TOP_K,
) -> list[dict]:
    """Search KG index with FAISS. Returns list of {text, score}."""
    embedding = model.encode([question], convert_to_numpy=True).astype(np.float32)
    results = kg_index.search(embedding, top_k=top_k)
    return [{"text": text, "score": score} for text, score in results]


def keyword_fallback_search(entities: list[dict], kg_entries: list[dict]) -> list[dict]:
    """Keyword fallback: match entity names against KG entry names."""
    entity_names = {e["en"].lower() for e in entities}
    found = []
    for entry in kg_entries:
        entry_name = str(entry.get("name_en", "")).lower()
        for ename in entity_names:
            if ename in entry_name or entry_name in ename:
                found.append(entry)
                break
    return found


def merge_kg_results(results: list[dict]) -> list[dict]:
    """Deduplicate KG results by name_en, keeping highest score."""
    seen = {}
    for r in results:
        key = str(r.get("name_en", "")).lower()
        if key not in seen:
            seen[key] = r
        elif r.get("score", 0) > seen[key].get("score", 0):
            seen[key] = r
    return list(seen.values())


def format_kg_context(entries: list[dict]) -> str:
    """Format KG entries into a readable context string for the LLM."""
    if not entries:
        return "(No KG knowledge found)"
    parts = []
    for i, e in enumerate(entries, 1):
        parts.append(
            f"[KG Entry {i}] {e.get('name_en', 'Unknown')}\n"
            f"  Symptoms: {e.get('symptoms', 'N/A')}\n"
            f"  Occurrence: {e.get('occurrence', 'N/A')}\n"
            f"  Prevention: {e.get('prevention', 'N/A')}"
        )
    return "\n\n".join(parts)


def retrieve_kg(
    question: str,
    entities: list[dict],
    kg_index: FaissIndex,
    kg_entries: list[dict],
    model,
) -> tuple[list[dict], str]:
    """Full KG retrieval pipeline. Returns (entries, formatted_context)."""
    faiss_results = faiss_kg_search(question, kg_index, model)
    faiss_entries = _parse_faiss_results(faiss_results, kg_entries)
    kw_entries = keyword_fallback_search(entities, kg_entries)
    all_entries = faiss_entries + kw_entries
    merged = merge_kg_results(all_entries)
    context = format_kg_context(merged)
    return merged, context


def _parse_faiss_results(faiss_results: list[dict], kg_entries: list[dict]) -> list[dict]:
    """Match FAISS text results back to structured KG entries."""
    faiss_texts = {r["text"][:50].lower(): r["score"] for r in faiss_results}
    matched = []
    for entry in kg_entries:
        entry_text = (
            f"Name: {entry.get('name_en', '')} | "
            f"Symptoms: {entry.get('symptoms', '')[:30]}"
        ).lower()
        for prefix, score in faiss_texts.items():
            if prefix in entry_text or entry_text[:50] in prefix:
                matched.append({**entry, "score": score})
                break
    return matched


def load_kg_entries(kg_en_path: str = config.PATH_CROPDP_KG_EN) -> list[dict]:
    """Load translated KG as list of dicts."""
    df = pd.read_csv(kg_en_path)
    return df.to_dict("records")
