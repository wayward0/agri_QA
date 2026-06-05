# Pipeline Optimization Design — 2026-06-05

Four incremental improvements targeting performance, quality metrics, code deduplication, and prompt caching.

## 1. RAG Batch Retrieval

**Problem**: `external_reviewer.py` calls `rag_tool.retrieve()` sequentially per step (5 factual steps = 5+1 serial API calls). Each call involves an embedding API roundtrip + FAISS search.

**Solution**: Add `retrieve_batch()` interface.

### Changes

**`rag/retriever.py`** — add `HybridRetriever.retrieve_batch()`:
```python
def retrieve_batch(
    self, queries: List[str], intent: str = "background", top_k: int = 5
) -> List[List[Evidence]]:
    """Batch retrieval: one embedding call, per-query BM25 + FAISS."""
    # 1. Batch embed all queries (single API call)
    query_vecs = self._embedding_model.encode(queries, normalize_embeddings=True)
    # 2. Per-query FAISS + BM25 (loop, but embedding is the bottleneck)
    all_results = []
    for i, query in enumerate(queries):
        dense = self._dense_search(query_vecs[i], top_k * 2)
        sparse = self._sparse_search(query, top_k * 2)
        fused = self._rrf_fusion(dense, sparse)
        all_results.append(fused[:top_k])
    # 3. Optional: batch rerank if enabled
    if self._reranker:
        # rerank per-query (API doesn't support batch)
        for i in range(len(all_results)):
            all_results[i] = self._rerank(queries[i], all_results[i], top_k)
    return all_results
```

**`rag/rag_tool.py`** — add `RAGTool.retrieve_batch()`:
```python
def retrieve_batch(self, queries: List[str], intent: str = "background", top_k: int = 5) -> List[List[Evidence]]:
    return self._retriever.retrieve_batch(queries, intent, top_k)
```

**`reviewer/external_reviewer.py`** — refactor `review_external()`:
- Collect all step queries into a list
- Call `rag_tool.retrieve_batch(queries, intent="fact_check", top_k=3)` once
- Map results back to steps
- Gap-fill query appended to batch (or separate call)

### Expected Impact
- Embedding API calls: 6 → 1 per item (83% reduction)
- Wall time per item: ~3s → ~1s for fact-checking phase

---

## 2. auto_metrics Hardening

**Problem**: Metrics are too shallow — empty steps score full marks, filler word set has 9 words, no step ordering check.

### Changes in `evaluator/auto_metrics.py`

**`structure_score`**:
- Add: each counted step must have `len(content.strip()) > 20`
- Add: `conclusion` step must be the last step (highest step number)
- Existing: context_setup exists, conclusion exists, step count in [3,7]

**`info_density`**:
- English filler set: expand from 9 to ~80 words (add: "basically", "actually", "well", "just", "really", "like", "things", "stuff", "very", "quite", "somewhat", "rather", "pretty", "maybe", "perhaps", "probably", "certainly", "definitely", "simply", "merely", "honestly", "frankly", "obviously", "clearly", "essentially", "fundamentally", "basically", "literally", "seriously", "absolutely", "totally", "completely", "entirely", "exactly", "precisely", "approximately", "roughly", "generally", "typically", "usually", "normally", "frequently", "occasionally", "sometimes", "often", "always", "never", "rarely", "seldom", "hardly", "barely", "scarcely")
- Chinese stopword set: "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "对", "让", "用", "为", "以", "所", "但", "而", "如果", "虽然", "因为", "所以", "可以", "这个", "那个", "什么", "怎么", "为什么", "哪", "谁", "多少"
- Auto-detect language by checking if >30% of characters are CJK

**New metric: `step_order_score`** (0-1):
- `context_setup` must appear before any `knowledge_application` or `causal_reasoning`
- `conclusion` must be the last step
- No `conclusion` before step 3

**Weight rebalancing** in `config.py`:
- Add `step_order` to `EVAL_WEIGHTS`, redistribute from `structure`

---

## 3. Evaluation Logic Deduplication

**Problem**: `pipeline.py:198-224` and `pipeline.py:246-270` contain nearly identical evaluation code.

### Changes in `pipeline.py`

Extract shared logic:
```python
def _evaluate_chain(
    chain: ReasoningChain, evaluator_call, auto_metrics_fn=compute_auto_metrics
) -> QualityScores:
    """Evaluate a reasoning chain and return quality scores."""
    faith, faith_notes = judge_faithfulness(chain, evaluator_call)
    logic, logic_notes = judge_logical_completeness(chain, evaluator_call)
    auto = auto_metrics_fn(chain)
    w = config.EVAL_WEIGHTS
    overall = (
        faith / 5 * w["faithfulness"]
        + auto["structure"] * w["structure"]
        + auto["information_density"] * w["information_density"]
        + logic / 5 * w["logical_completeness"]
        + auto["traceability"] * w["traceability"]
    )
    ppl_score = compute_ppl(chain.to_text(), api_key=config.API_KEY, base_url=config.API_BASE_URL, model=config.MODEL_LIGHT)
    return QualityScores(
        faithfulness=faith, logical_completeness=logic,
        overall=round(overall, 3), ppl=ppl_score, **auto,
    ), faith_notes, logic_notes
```

Both call sites become:
```python
scores, faith_notes, logic_notes = _evaluate_chain(revised_chain, evaluator_call)
```

---

## 4. ReAct Prompt Caching

**Problem**: `_build_react_prompt()` rebuilds the entire prompt each round. System prompt + early observations are repeated verbatim, wasting tokens and preventing prefix caching.

### Changes in `thinker/react_generator.py`

**Current flow**:
```
Round 1: llm_call(system_prompt + QA + Thought1 + Obs1)
Round 2: llm_call(system_prompt + QA + Thought1 + Obs1 + Thought2 + Obs2)  ← system repeated
```

**New flow**:
```
Round 1: llm_call(system=static_instructions, user=QA + Thought1 + Obs1)
Round 2: llm_call(system=static_instructions, user=QA + Thought1 + Obs1 + Thought2 + Obs2)
         ↑ DeepSeek caches this prefix automatically
```

**Implementation**:
- `_build_react_prompt()` returns `(system_text, user_text)` tuple
- `system_text` = static ReAct instructions (same every round)
- `user_text` = QA + accumulated observations (grows each round)
- `llm_call()` already supports `system` parameter — no interface change needed

---

## Execution Order

1. **RAG batch retrieval** (highest performance impact)
2. **auto_metrics hardening** (quality improvement)
3. **Evaluation logic dedup** (code cleanup, enables cleaner future changes)
4. **ReAct prompt caching** (performance + cost saving)

## Files Changed Summary

| File | Changes |
|------|---------|
| `rag/retriever.py` | Add `retrieve_batch()` |
| `rag/rag_tool.py` | Add `retrieve_batch()` |
| `reviewer/external_reviewer.py` | Use batch retrieval |
| `evaluator/auto_metrics.py` | Harden metrics, add `step_order_score` |
| `config.py` | Add `step_order` to EVAL_WEIGHTS |
| `pipeline.py` | Extract `_evaluate_chain()`, update both call sites |
| `thinker/react_generator.py` | Separate system/user prompt, incremental build |
