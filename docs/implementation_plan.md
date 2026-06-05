# Implementation Plan: Multi-Agent Agricultural Reasoning Pipeline

## Context

Build a reasoning-enhanced agricultural QA dataset by generating structured reasoning chains for AgThoughts QA pairs. Architecture design is complete in `docs/architecture_design.md`. No code exists yet. User requires modular programming with strict module independence.

## Project Structure

```
/Users/xuyuhang/agri_QA/
├── config.py                      # Constants, paths, hyperparams (no project imports)
├── models.py                      # Shared Pydantic models (the ONLY shared dependency)
├── llm_client.py                  # OpenAI wrapper -> returns LLMCallFn callable
├── pipeline.py                    # Orchestrator (ONLY file that imports all modules)
├── classifier/
│   ├── __init__.py
│   └── difficulty_classifier.py   # Stage 0: easy/medium/hard routing
├── rag/
│   ├── __init__.py
│   ├── knowledge_builder.py       # Offline: Wikipedia -> chunks -> FAISS + BM25
│   ├── retriever.py               # Hybrid retrieval: FAISS + BM25 + RRF
│   ├── query_processor.py         # Query rewriting + entity extraction
│   └── rag_tool.py                # Unified RAGTool interface
├── thinker/
│   ├── __init__.py
│   ├── react_generator.py         # ReAct loop: Thought -> Action -> Observation
│   └── self_consistency.py        # Multi-sample generation + best selection
├── reviewer/
│   ├── __init__.py                # run_review() convenience function
│   ├── logic_reviewer.py          # Phase A: internal logic analysis
│   ├── external_reviewer.py       # Phase B: fact-check via RAG + gap-fill
│   └── integrator.py              # Phase C: merge -> UnifiedActions
├── reviser/
│   ├── __init__.py
│   └── reviser.py                 # Structured atomic action executor
├── evaluator/
│   ├── __init__.py                # evaluate_chain() convenience wrapper
│   ├── llm_judge.py               # Faithfulness + Logical Completeness (LLM)
│   ├── auto_metrics.py            # Structure + Density + Traceability (pure)
│   └── ppl_scorer.py              # Perplexity (local model)
├── data/agthoughts/sample_1000.json
├── data/chunks/passages.jsonl
├── data/index/ (faiss.index, bm25.pkl, metadata.json)
├── output/enhanced_dataset.json
├── tests/
├── requirements.txt
└── pytest.ini
```

## Module Independence Rule

No module imports another module. All modules import only `types.py`. Dependencies injected as parameters. Only `pipeline.py` imports everything.

## Implementation Order

| Phase | Files | What |
|-------|-------|------|
| 1 | `types.py` | Pydantic models: Evidence, ReasoningStep, ReasoningChain, DifficultyLevel, ReviewAction, UnifiedActions, ReviewCritique, QualityScores, PipelineItem, LLMCallFn Protocol |
| 2 | `config.py`, `llm_client.py` | Config constants + `create_llm_caller()` |
| 3 | `requirements.txt`, `pytest.ini` | Dependencies + test config |
| 4 | `evaluator/auto_metrics.py` | Pure functions: structure_score(), info_density(), traceability() |
| 5 | `rag/knowledge_builder.py` | Offline: Wikipedia -> chunk -> FAISS + BM25 |
| 6 | `rag/query_processor.py`, `rag/retriever.py`, `rag/rag_tool.py` | Hybrid retrieval + RRF |
| 7 | `classifier/difficulty_classifier.py` | 1 LLM call -> easy/medium/hard |
| 8 | `thinker/react_generator.py`, `thinker/self_consistency.py` | ReAct + multi-sample |
| 9 | `reviewer/` (3 files) | 3-phase review -> UnifiedActions |
| 10 | `reviser/reviser.py` | Execute atomic actions |
| 11 | `evaluator/llm_judge.py`, `evaluator/ppl_scorer.py`, `evaluator/__init__.py` | LLM scoring + PPL |
| 12 | `pipeline.py` | Orchestrator with difficulty routing |
| 13 | `tests/` | Unit + integration tests |
| 14 | Data prep | Subsample 1000 items from AgThoughts.json |

## Key Design Decisions

- **LLMCallFn Protocol**: Every LLM-using module receives a callable, not an OpenAI client. Testing = pass a mock function.
- **3 Difficulty Paths**: Easy (~5 LLM calls), Medium (~8), Hard (~18). ~47% cost savings.
- **RAG-as-Tool**: Shared callable, not a pipeline stage. Any agent calls `rag_tool.retrieve()`.
- **Structured Actions**: Reviewer outputs JSON atomic ops. Reviser executes -- 50% programmatic.

## Verification

1. `pytest tests/` -- each module testable in isolation
2. Integration: 3 items (one per difficulty) with mocked LLM
3. RAG: build index with 10 articles, verify retrieval
4. E2E: 5 real items, inspect output chains
