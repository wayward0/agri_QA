"""Structured CoT reasoning chain generation with XML output."""
import re
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple

from openai import OpenAI
import config


SYSTEM_PROMPT = """You are an expert agricultural pathologist. Given a question and knowledge graph evidence, produce a structured reasoning chain in XML format.

Your output MUST be a single <reasoning_chain> XML element with these children:
- <question_analysis>: contains <type>, <crop>, <symptoms>
- <evidence_retrieval>: contains one or more <kg_match> elements
- <reasoning_steps>: contains numbered <step> elements
- <conclusion>: contains <diagnosis> and <prevention_plan>

Output ONLY the XML. No markdown, no explanation outside the XML."""


def get_client() -> OpenAI:
    return OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


def build_reasoning_prompt(question: str, kg_context: str, entities: List[Dict]) -> str:
    entity_desc = ", ".join(f"{e['en']}({e['cn']})" for e in entities) if entities else "None identified"
    return f"""Question: {question}

Identified entities: {entity_desc}

Knowledge Graph Evidence:
{kg_context}

Produce a <reasoning_chain> in XML format to answer this agricultural question."""


def call_llm(prompt: str, client: OpenAI, temperature: float = config.TEMPERATURE_GENERATE) -> str:
    response = client.chat.completions.create(
        model=config.LLM_MODEL_GENERATE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def parse_xml_reasoning(raw: str) -> Optional[Dict]:
    """Parse XML reasoning chain into a dict. Returns None if unparseable."""
    xml_match = re.search(r'<reasoning_chain>.*?</reasoning_chain>', raw, re.DOTALL)
    if not xml_match:
        return None
    xml_str = xml_match.group()
    try:
        root = ET.fromstring(xml_str)
        diagnosis_elem = root.find(".//diagnosis")
        prevention_elem = root.find(".//prevention_plan")
        steps = root.findall(".//step")
        kg_matches = root.findall(".//kg_match")
        return {
            "diagnosis": diagnosis_elem.text.strip() if diagnosis_elem is not None and diagnosis_elem.text else "Unknown",
            "prevention_plan": prevention_elem.text.strip() if prevention_elem is not None and prevention_elem.text else "Unknown",
            "steps": [s.text.strip() for s in steps if s.text],
            "kg_match_count": len(kg_matches),
            "raw_xml": xml_str,
        }
    except ET.ParseError:
        return None


def extract_diagnosis(parsed: Optional[Dict]) -> str:
    if parsed is None:
        return "Unknown"
    return parsed.get("diagnosis", "Unknown")


def generate_reasoning(
    question: str,
    kg_context: str,
    entities: List[Dict],
    client: Optional[OpenAI] = None,
    temperature: float = config.TEMPERATURE_GENERATE,
) -> Tuple[str, Optional[Dict]]:
    """Generate a single reasoning chain. Returns (raw_xml, parsed_dict)."""
    if client is None:
        client = get_client()
    prompt = build_reasoning_prompt(question, kg_context, entities)
    raw = call_llm(prompt, client, temperature=temperature)
    parsed = parse_xml_reasoning(raw)
    return raw, parsed
