"""Post-verification: check reasoning chain against KG evidence."""
import re
import xml.etree.ElementTree as ET
from typing import Optional
from openai import OpenAI
import config


VERIFY_SYSTEM_PROMPT = """You are an agricultural fact-checker. Given a reasoning chain (XML) and knowledge graph evidence, verify whether the reasoning is consistent with the evidence.

Output ONLY a <verification> XML element with:
- <status>: one of "verified", "partially_verified", "contradicted"
- <contradictions>: list specific contradictions, or "none"
- <unsupported_claims>: claims in the reasoning not supported by KG evidence, or "none"

Output ONLY the XML."""


def get_client() -> OpenAI:
    return OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


def build_verification_prompt(reasoning_xml: str, kg_context: str) -> str:
    return f"""Reasoning Chain to verify:
{reasoning_xml}

Knowledge Graph Evidence:
{kg_context}

Verify the reasoning chain against the KG evidence."""


def parse_verification_result(raw: str) -> dict:
    """Parse verification XML. Returns dict with status, contradictions, unsupported_claims."""
    xml_match = re.search(r'<verification>.*?</verification>', raw, re.DOTALL)
    if not xml_match:
        return {"status": "parse_failed", "contradictions": raw[:200], "unsupported_claims": "unknown"}
    try:
        root = ET.fromstring(xml_match.group())
        status = root.findtext("status", "unknown").strip()
        contradictions = root.findtext("contradictions", "unknown").strip()
        unsupported = root.findtext("unsupported_claims", "unknown").strip()
        return {
            "status": status,
            "contradictions": contradictions,
            "unsupported_claims": unsupported,
        }
    except ET.ParseError:
        return {"status": "parse_failed", "contradictions": raw[:200], "unsupported_claims": "unknown"}


def verify_reasoning(
    reasoning_xml: str,
    kg_context: str,
    client: Optional[OpenAI] = None,
) -> dict:
    """Verify a reasoning chain against KG evidence."""
    if client is None:
        client = get_client()
    prompt = build_verification_prompt(reasoning_xml, kg_context)
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_VERIFY,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        return parse_verification_result(raw)
    except Exception as e:
        return {"status": "api_error", "contradictions": str(e)[:200], "unsupported_claims": "unknown"}
