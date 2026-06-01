"""Translate CropDP-KG from Chinese to English using LLM."""
import json
import re
import time
import pandas as pd
from openai import OpenAI
import config


def get_client():
    return OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


def has_chinese(text: str) -> bool:
    return bool(re.search(r'[一-鿿]', text))


def build_translation_prompt(row: pd.Series) -> str:
    return f"""Translate the following Chinese agricultural knowledge entry to English.
Return ONLY a JSON object with these exact keys: name_en, symptoms, occurrence, prevention.
Do not add any explanation or markdown formatting.

Input:
- 名称 (Name): {row['名称']}
- 英文名 (English name): {row.get('英文名', '')}
- 为害症状 (Symptoms): {row['为害症状']}
- 发生规律 (Occurrence pattern): {row['发生规律']}
- 防治 (Prevention/treatment): {row['防治']}

Output JSON:"""


def validate_translation(result: dict) -> bool:
    """Check translation is non-empty, reasonable length, no Chinese residue."""
    required_keys = ["name_en", "symptoms", "occurrence", "prevention"]
    for k in required_keys:
        val = result.get(k, "")
        if not val or len(val) < 5:
            return False
        if has_chinese(val):
            return False
    return True


def translate_batch(rows: pd.DataFrame, client: OpenAI, batch_size: int = 50) -> list[dict]:
    """Translate a batch of KG rows. Returns list of translated dicts."""
    results = []
    for i in range(0, len(rows), batch_size):
        batch = rows.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            result = translate_single(row, client)
            results.append(result)
    return results


def translate_single(row: pd.Series, client: OpenAI) -> dict:
    """Translate one KG row. Retries once on failure."""
    prompt = build_translation_prompt(row)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL_TRANSLATE,
                messages=[
                    {"role": "system", "content": "You are an agricultural translator. Translate Chinese agricultural knowledge to English accurately. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown wrapping)
            json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if validate_translation(result):
                    return {
                        "name_cn": str(row["名称"]),
                        "name_en": result["name_en"],
                        "symptoms": result["symptoms"],
                        "occurrence": result["occurrence"],
                        "prevention": result["prevention"],
                    }
        except Exception:
            if attempt == 0:
                time.sleep(2)
                continue
    # Fallback: use original English name if available
    return {
        "name_cn": str(row["名称"]),
        "name_en": str(row.get("英文名", "")),
        "symptoms": str(row.get("为害症状", "")),
        "occurrence": str(row.get("发生规律", "")),
        "prevention": str(row.get("防治", "")),
        "translation_failed": True,
    }


def run_translation(input_path: str, output_path: str) -> pd.DataFrame:
    """Full translation pipeline: read CSV -> translate -> write CSV."""
    df = pd.read_csv(input_path)
    client = get_client()
    results = translate_batch(df, client)
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    failed = out_df.get("translation_failed", pd.Series(dtype=bool)).sum()
    print(f"Translated {len(out_df)} rows. Failed: {failed}")
    return out_df


if __name__ == "__main__":
    run_translation(config.PATH_CROPDP_KG, config.PATH_CROPDP_KG_EN)
