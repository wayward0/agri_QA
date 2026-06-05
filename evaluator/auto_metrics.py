"""Automated quality metrics — pure functions, no external dependencies.

Metrics:
- structure_score: chain structure (context_setup, conclusion, step count, non-empty content)
- info_density: ratio of non-formulaic words to total words (EN + CN)
- traceability: ratio of steps with evidence to total steps
- step_order_score: logical ordering of step types
"""

import re
from models import ReasoningChain

# English filler / low-information words
_EN_FILLER = {
    "okay", "so", "let", "me", "i", "need", "to", "should", "right",
    "basically", "actually", "well", "just", "really", "like", "things",
    "stuff", "very", "quite", "somewhat", "rather", "pretty", "maybe",
    "perhaps", "probably", "certainly", "definitely", "simply", "merely",
    "honestly", "frankly", "obviously", "clearly", "essentially",
    "fundamentally", "literally", "seriously", "absolutely", "totally",
    "completely", "entirely", "exactly", "precisely", "approximately",
    "roughly", "generally", "typically", "usually", "normally",
    "frequently", "occasionally", "sometimes", "often", "always", "never",
    "rarely", "seldom", "hardly", "barely", "scarcely", "basically",
    "um", "uh", "er", "ah", "you", "we", "they", "it", "the", "a", "an",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "might", "may", "can",
    "shall", "must", "this", "that", "these", "those", "here", "there",
}

# Chinese stopwords
_CN_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "被",
    "从", "把", "对", "让", "用", "为", "以", "所", "但", "而", "如果",
    "虽然", "因为", "所以", "可以", "这个", "那个", "什么", "怎么", "为什么",
    "哪", "谁", "多少", "吗", "吧", "呢", "啊", "呀", "哦", "嗯", "么",
    "与", "及", "或", "等", "之", "其", "此", "该", "本", "各", "每",
    "中", "内", "外", "前", "后", "左", "右", "大", "小", "多", "少",
}

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def _is_cjk(text: str) -> bool:
    """Detect if text is predominantly Chinese/CJK."""
    cjk_chars = len(_CJK_RE.findall(text))
    return cjk_chars > len(text) * 0.3


def structure_score(chain: ReasoningChain) -> float:
    """Returns 0-1. Checks: context_setup exists, conclusion exists,
    step count 3-7, and all steps have non-empty content."""
    if not chain.steps:
        return 0.0

    has_context = any(s.type == "context_setup" for s in chain.steps)
    has_conclusion = any(s.type == "conclusion" for s in chain.steps)
    optimal_steps = 3 <= len(chain.steps) <= 7

    # All steps must have non-trivial content
    all_nonempty = all(len(s.content.strip()) > 20 for s in chain.steps)

    # Conclusion must be the last step
    conclusion_last = (
        chain.steps[-1].type == "conclusion" if has_conclusion else False
    )

    score = (
        int(has_context) * 0.20
        + int(has_conclusion) * 0.20
        + int(optimal_steps) * 0.20
        + int(all_nonempty) * 0.20
        + int(conclusion_last) * 0.20
    )
    return score


def info_density(chain: ReasoningChain) -> float:
    """Returns 0-1. Ratio of non-formulaic words to total words.

    Auto-detects language (English vs Chinese) and applies appropriate
    stopword set.
    """
    text = " ".join(s.content for s in chain.steps)
    if not text.strip():
        return 0.0

    if _is_cjk(text):
        # Chinese: count characters (excluding punctuation and stopwords)
        chars = list(text)
        total = sum(1 for c in chars if _CJK_RE.match(c))
        if total == 0:
            return 0.0
        stopwords_count = sum(1 for c in chars if c in _CN_STOPWORDS)
        return max(0.0, (total - stopwords_count) / total)
    else:
        # English: count words
        words = text.lower().split()
        if not words:
            return 0.0
        filler = sum(1 for w in words if w.strip(".,;:!?\"'()[]{}") in _EN_FILLER)
        return max(0.0, (len(words) - filler) / len(words))


def traceability(chain: ReasoningChain) -> float:
    """Returns 0-1. Ratio of steps with evidence to total steps."""
    if not chain.steps:
        return 0.0
    with_evidence = sum(1 for s in chain.steps if s.evidence)
    return with_evidence / len(chain.steps)


def step_order_score(chain: ReasoningChain) -> float:
    """Returns 0-1. Checks logical ordering of step types.

    - context_setup should appear before knowledge_application / causal_reasoning
    - conclusion must be the last step
    - conclusion should not appear before step 3
    """
    if len(chain.steps) < 2:
        return 0.0

    types = [s.type for s in chain.steps]
    score = 0.0
    checks = 0

    # context_setup before factual steps
    if "context_setup" in types:
        checks += 1
        ctx_idx = types.index("context_setup")
        factual_types = {"knowledge_application", "causal_reasoning", "evidence_integration"}
        first_factual = next((i for i, t in enumerate(types) if t in factual_types), len(types))
        if ctx_idx < first_factual:
            score += 1.0

    # conclusion must be last
    if "conclusion" in types:
        checks += 1
        if types[-1] == "conclusion":
            score += 1.0

        # conclusion should not appear before step 3
        checks += 1
        con_idx = types.index("conclusion")
        if con_idx >= 2:  # 0-indexed, so >= 2 means step 3+
            score += 1.0

    return score / checks if checks > 0 else 1.0


def compute_auto_metrics(chain: ReasoningChain) -> dict:
    """Compute all automated metrics."""
    return {
        "structure": round(structure_score(chain), 3),
        "information_density": round(info_density(chain), 3),
        "traceability": round(traceability(chain), 3),
        "step_order": round(step_order_score(chain), 3),
    }
