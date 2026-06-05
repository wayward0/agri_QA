"""Query processing: rewriting and entity extraction.

Pure functions — no external dependencies.
"""

from typing import List


# Agricultural entity lexicon for keyword-based extraction
AG_LEXICON = {
    # Crops
    "corn", "wheat", "rice", "soybean", "cotton", "tomato", "potato",
    "lettuce", "cabbage", "collard", "pepper", "bean", "pea", "carrot",
    "onion", "garlic", "cucumber", "squash", "pumpkin", "melon",
    "strawberry", "blueberry", "apple", "grape", "citrus", "orange",
    "lemon", "peach", "cherry", "plum", "pear", "avocado", "mango",
    # Pests & Diseases
    "blight", "rust", "mildew", "wilt", "rot", "spot", "scab",
    "aphid", "beetle", "caterpillar", "mite", "nematode", "weevil",
    "thrips", "whitefly", "skipper", "borer", "fly", "mosquito",
    "fungus", "bacteria", "virus", "pathogen", "nematode",
    # Soil & Inputs
    "nitrogen", "phosphorus", "potassium", "lime", "compost", "mulch",
    "fertilizer", "pesticide", "herbicide", "insecticide", "fungicide",
    "irrigation", "drainage", "tillage", "cover crop",
    # Conditions
    "drought", "frost", "heat", "flooding", "erosion", "salinity",
    "acidity", "alkalinity", "ph", "organic matter",
}


def rewrite_query(query: str, intent: str) -> str:
    """Apply intent-specific query rewriting.

    Args:
        query: Original query text.
        intent: One of "background", "fact_check", "gap_fill".

    Returns:
        Rewritten query string.
    """
    if intent == "background":
        return query
    elif intent == "fact_check":
        # For fact-checking, extract factual claims by removing question words
        cleaned = query
        for prefix in ("how ", "what ", "why ", "when ", "where ", "which ", "is ", "are ", "can ", "do ", "does "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
        return cleaned.strip()
    elif intent == "gap_fill":
        return query
    else:
        return query


def extract_entities(query: str) -> List[str]:
    """Extract agricultural entities from query text.

    Uses simple keyword matching against the agricultural lexicon.

    Args:
        query: Query text.

    Returns:
        List of matched entity strings.
    """
    query_lower = query.lower()
    found = []
    for term in AG_LEXICON:
        if term in query_lower:
            found.append(term)
    return found
