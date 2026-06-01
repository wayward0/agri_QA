"""Agri-QA Pipeline v2 — main orchestrator."""
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from xml.dom import minidom

from lxml import etree
from tqdm import tqdm

import config
from modules.kg_translator import run_translation
from modules.faiss_builder import FaissIndex, load_embedding_model, build_all_indices
from modules.entity_recognizer import (
    build_and_save_term_dict, recognize_entities, save_term_dict, load_term_dict,
)
from modules.kg_retriever import retrieve_kg, load_kg_entries
from modules.reasoning_generator import generate_reasoning, get_client
from modules.consistency_checker import run_self_consistency
from modules.verifier import verify_reasoning


# ============================================================
#  Step 0: Data preparation
# ============================================================

def prepare_data():
    """Run all data preparation: translate KG, build indices, filter AgThoughts."""
    print("=" * 60)
    print("Step 0: Data Preparation")
    print("=" * 60)

    # 0a. Translate KG (skip if already exists)
    if not os.path.exists(config.PATH_CROPDP_KG_EN):
        print("Translating CropDP-KG to English...")
        run_translation(config.PATH_CROPDP_KG, config.PATH_CROPDP_KG_EN)
    else:
        print(f"Translated KG already exists: {config.PATH_CROPDP_KG_EN}")

    # 0b. Build term dict
    print("Building term dictionary...")
    term_dict = build_and_save_term_dict()

    # 0c. Build FAISS indices
    print("Building FAISS indices...")
    build_all_indices()

    # 0d. Filter AgThoughts
    print("Filtering AgThoughts to disease/pest subset...")
    subset = filter_agthroughts_to_file(term_dict)

    return term_dict, subset


def filter_agthroughts_to_file(term_dict: dict) -> List[dict]:
    """Filter AgThoughts and save subset."""
    with open(config.PATH_AGTHOUGHTS, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    filtered = []
    for item in all_items:
        q_type = item.get("Question Type", "")
        if q_type in config.BIOTIC_CATEGORIES:
            filtered.append(item)
        elif q_type == "Plant and Seed Health Questions":
            question = item.get("Question", "").lower()
            if any(term in question for term in term_dict):
                filtered.append(item)

    with open(config.PATH_AGRI_SUBSET, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"  AgThoughts: {len(all_items)} -> {len(filtered)} (disease/pest subset)")
    return filtered


# ============================================================
#  Pipeline main loop
# ============================================================

def build_xml_output(items_data: List[dict]) -> str:
    """Build XML output from processed items."""
    root = ET.Element("agri_qa_dataset")
    for item in items_data:
        item_elem = ET.SubElement(root, "item", id=str(item["id"]))

        q_elem = ET.SubElement(item_elem, "question")
        q_elem.text = item["question"]

        qt_elem = ET.SubElement(item_elem, "question_type")
        qt_elem.text = item["question_type"]

        # Entities
        ents_elem = ET.SubElement(item_elem, "entities")
        for e in item.get("entities", []):
            ent_elem = ET.SubElement(ents_elem, "entity",
                                     type=e.get("type", ""),
                                     name=e.get("en", ""))
            ent_elem.text = e.get("cn", "")

        # KG evidence
        kg_elem = ET.SubElement(item_elem, "kg_evidence")
        for kg in item.get("kg_entries", []):
            entry_elem = ET.SubElement(kg_elem, "entry")
            for field in ["name_en", "symptoms", "occurrence", "prevention"]:
                field_elem = ET.SubElement(entry_elem, field)
                field_elem.text = str(kg.get(field, ""))

        # Reasoning chain (raw XML)
        raw_xml = item.get("raw_xml", "<reasoning_chain/>")
        rc_elem = ET.SubElement(item_elem, "reasoning_chain")
        try:
            rc_elem.append(etree.fromstring(raw_xml.encode()))
        except Exception:
            rc_elem.text = raw_xml

        # Verification
        ver_elem = ET.SubElement(item_elem, "verification")
        ver = item.get("verification", {})
        for field in ["status", "contradictions", "unsupported_claims"]:
            field_elem = ET.SubElement(ver_elem, field)
            field_elem.text = str(ver.get(field, "unknown"))

    # Pretty print
    rough = ET.tostring(root, encoding="unicode")
    parsed = etree.fromstring(rough.encode())
    return etree.tostring(parsed, pretty_print=True, encoding="unicode")


def run_pipeline():
    """Full pipeline execution."""
    # Prepare data
    term_dict, subset = prepare_data()

    # Load indices and model
    print("\nLoading FAISS indices and embedding model...")
    model = load_embedding_model()
    kg_index = FaissIndex(index_dir=config.FAISS_INDEX_DIR)
    kg_index.load("kg_index")
    entity_index = FaissIndex(index_dir=config.FAISS_INDEX_DIR)
    entity_index.load("entity_index")
    kg_entries = load_kg_entries()

    # LLM client
    client = get_client()

    # Process items
    items_to_process = subset[:config.TEST_LIMIT] if config.TEST_LIMIT else subset
    print(f"\n{'=' * 60}")
    print(f"Processing {len(items_to_process)} items")
    print(f"{'=' * 60}")

    results = []
    for idx, item in enumerate(tqdm(items_to_process, desc="Pipeline")):
        question = item.get("Question", "")
        q_type = item.get("Question Type", "")

        # Step 1: Entity recognition
        entities = recognize_entities(question, term_dict, entity_index, model)

        # Step 2: KG retrieval
        kg_entries_matched, kg_context = retrieve_kg(question, entities, kg_index, kg_entries, model)

        # Step 3+4: Reasoning generation with self-consistency
        best_result, consistency = run_self_consistency(
            question, kg_context, entities, client=client,
        )

        # Step 5: Post-verification
        verification = verify_reasoning(
            best_result.get("raw_xml", ""), kg_context, client=client,
        )

        results.append({
            "id": idx + 1,
            "question": question,
            "question_type": q_type,
            "entities": entities,
            "kg_entries": kg_entries_matched,
            "raw_xml": best_result.get("raw_xml", ""),
            "diagnosis": best_result.get("diagnosis", "Unknown"),
            "consistency": consistency,
            "verification": verification,
        })

        # Rate limiting
        if idx < len(items_to_process) - 1:
            time.sleep(config.API_CALL_INTERVAL)

    # Build XML output
    print("\nBuilding XML output...")
    xml_str = build_xml_output(results)
    os.makedirs(os.path.dirname(config.PATH_OUTPUT_XML), exist_ok=True)
    with open(config.PATH_OUTPUT_XML, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # Summary
    verified = sum(1 for r in results if r["verification"].get("status") == "verified")
    high_conf = sum(1 for r in results if r["consistency"].get("status") == "high_confidence")
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Total: {len(results)}")
    print(f"  Verified: {verified}/{len(results)}")
    print(f"  High confidence: {high_conf}/{len(results)}")
    print(f"  Output: {config.PATH_OUTPUT_XML}")


if __name__ == "__main__":
    run_pipeline()
