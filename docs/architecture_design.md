# Multi-Agent Architecture for Agricultural Reasoning Dataset

## Project Goal

Build a reasoning-enhanced agricultural QA dataset. The primary output is **structured reasoning chains**, not answers. Answers are already available from AgThoughts (44,615 QA pairs).

**Core principle:** Every design decision should be evaluated against: "Does this improve the quality of the reasoning chain?" The answer is secondary — it's the anchor, not the product.

---

## Source Data: AgThoughts

- **Dataset:** AgThoughts.json (44,615 items, 44,571 with valid reasoning traces)
- **Fields:** Question, Reasoning Traces, Answer, Question Type
- **Question Types:**
  - Plant and Seed Health Questions: 15,811
  - Crop Management Questions: 7,539
  - Crop Inputs Questions: 6,105
  - Abiotic Harvest Questions: 4,139
  - Abiotic Soil Questions: 3,896
  - Abiotic Weather Questions: 3,527
  - Cover Crop Questions: 1,969
  - Biotic Diseases Questions: 819
  - Biotic Insects Questions: 702
  - Biotic Weeds Questions: 108
- **Stats:** Question avg 24 words, Reasoning avg 710 words, Answer avg 355 words
- **Key finding:** Existing Reasoning Traces are formulaic LLM-generated text (100% start with "Okay", 67.8% contain "I need to")

---

## Architecture: 4-Module Multi-Agent Pipeline

### Core Design Principle: RAG-as-Tool (not RAG-as-Stage)

RAG is a shared tool callable by ALL agents at any point, not a one-shot stage at the beginning. This enables:
- Thinker to request evidence mid-reasoning
- Reviewer to fact-check specific claims on demand
- Evaluator to verify assertions as needed

### Pipeline Flow (with Dynamic Routing)

```
Input: {Question, Answer} from AgThoughts
          │
          ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Stage 0: Difficulty Classifier                          │
    │  轻量 LLM 调用 (1次) → easy / medium / hard              │
    │                                                          │
    │  分类标准:                                                │
    │  - EASY:   单主题、事实性、直接答案                         │
    │            "What depth to plant corn?"                    │
    │  - MEDIUM: 多维度但聚焦                                   │
    │            "How to treat tomato blight?"                  │
    │  - HARD:   多部分、有条件、需要区域/实践推理                 │
    │            "What issues affect my crops and how to        │
    │             prevent them next year in New Jersey?"        │
    └──────┬──────────┬──────────┬────────────────────────────┘
           │          │          │
      easy │     medium│     hard│
           ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────────────────────┐
    │ Path A   │ │ Path B   │ │ Path C                       │
    │ 精简版    │ │ 标准版    │ │ 完整版                        │
    │          │ │          │ │                              │
    │ Thinker: │ │ Thinker: │ │ Thinker:                     │
    │ ReAct ×1 │ │ ReAct ×1 │ │ ReAct ×3 (Self-Consistency) │
    │          │ │          │ │                              │
    │ Reviewer:│ │ Reviewer:│ │ Reviewer:                    │
    │ Phase A  │ │ Phase A+B│ │ Phase A+B+C                  │
    │          │ │          │ │                              │
    │ Reviser: │ │ Reviser: │ │ Reviser:                     │
    │ skip     │ │ yes      │ │ yes                          │
    │          │ │          │ │                              │
    │Evaluator:│ │Evaluator:│ │ Evaluator:                   │
    │auto only │ │ full     │ │ full + quality gate           │
    └────┬─────┘ └────┬─────┘ └────────────┬─────────────────┘
         │            │                     │
         └────────────┼─────────────────────┘
                      ▼
    Output: {
        Question, Answer,
        Original_Reasoning,    ← AgThoughts original (for comparison)
        Enhanced_Reasoning,    ← PRIMARY OUTPUT (structured chain)
        Quality_Comparison,    ← Original vs Enhanced scores
        Critique_History,      ← review records
        Metadata               ← pipeline info + difficulty level
    }
```

#### Three Paths Detail

| Stage | Path A (Easy) | Path B (Medium) | Path C (Hard) |
|-------|---------------|-----------------|---------------|
| **Thinker** | ReAct × 1 | ReAct × 1 | ReAct × 3 (Self-Consistency) |
| **Reviewer** | Phase A only (logic) | Phase A + B (logic + external) | Phase A + B + C (full) |
| **Reviser** | Skip | Yes | Yes |
| **Evaluator** | Automated metrics only | Full (LLM + auto + PPL) | Full + quality gate (< 3.0 → revise) |
| **LLM calls/item** | ~5 | ~8 | ~18 |
| **RAG calls/item** | ~2 | ~4 | ~6 |

#### Difficulty Classifier Prompt

```
Classify the difficulty of this agricultural QA pair.

Question: {question}
Answer: {answer}

Criteria:
- EASY: Single-topic, factual, direct answer. 
  Example: "What is the optimal planting depth for corn?"
- MEDIUM: Multiple aspects but focused, requires some domain knowledge.
  Example: "How to treat tomato blight in humid conditions?"
- HARD: Multi-part question, conditional reasoning, requires regional 
  context or practical trade-offs.
  Example: "What issues affect my collards AND how can I prevent 
  silverspotted skipper infestations next year in New Jersey?"

Output ONLY: easy / medium / hard
```

#### Cost Comparison

Assuming distribution: Easy 50%, Medium 30%, Hard 20%

| | Easy (500) | Medium (300) | Hard (200) | Total |
|---|-----------|-------------|-----------|-------|
| **No routing** | 9,000 | 5,400 | 3,600 | 18,000 |
| **With routing** | 2,500 | 2,400 | 3,600 | 8,500 + 1,000(classifier) |
| **Savings** | | | | **~47%** |

Total with routing: ~9,500 LLM calls (~$3-5 with GPT-4o-mini)

---

## Module Details

### Module 1: RAG-Tool (Shared Tool) — Detailed Design

#### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        RAG-Tool                               │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Knowledge Base (离线构建，一次性的)                       │ │
│  │  Wikipedia API ──→ 筛选农业条目 ──→ 分块 ──→ Embedding   │ │
│  │                                         ──→ FAISS Index  │ │
│  │                                         ──→ BM25 Index   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Query Pipeline (在线检索，每次调用)                       │ │
│  │  Agent 调用 ──→ Query Rewriter ──→ Hybrid Retriever      │ │
│  │  (意图识别+查询改写)    (Dense + Sparse) ──→ Post-Processor│ │
│  │                                              │            │ │
│  │                                        list[Evidence]     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

#### Knowledge Base Construction (Offline)

**数据源:** Wikipedia 英文版，通过 `wikipedia-api` 自动获取

**种子主题（对应 AgThoughts Question Type）:**

| Question Type | Wikipedia 种子主题 |
|--------------|-------------------|
| Plant and Seed Health | Plant_disease, Seed_treatment, Germination, Plant_pathology |
| Crop Management | Crop_rotation, Tillage, Cover_crop, Companion_planting, Harvest |
| Crop Inputs | Fertilizer, Pesticide, Herbicide, Insecticide, Fungicide |
| Abiotic Harvest | Post-harvest, Food_preservation, Grain_storage |
| Abiotic Soil | Soil, Soil_health, Soil_pH, Compost, Mulching |
| Abiotic Weather | Drought, Frost, Heat_storm, Agricultural_meteorology |
| Cover Crop | Cover_crop, Green_manure, Nitrogen_fixation |
| Biotic Diseases | Plant_pathology, Fungal_infection, Bacterial_wilt |
| Biotic Insects | Agricultural_pest, Integrated_pest_management, Beneficial_insect |
| Biotic Weeds | Weed, Weed_control, Allelopathy |

每个种子主题递归获取 1 层关联条目，总计约 750-1,000 篇。

**分块策略（Hierarchical Chunking）:**

```
Level 1: Article（整篇条目）
  metadata: {title, categories, url, last_edited}

Level 2: Section（按 == 标题分割）
  metadata: {article_title, section_title, section_index}

Level 3: Passage（语义段落，200-400 tokens）← 存入向量库的单元
  metadata: {article_title, section_title, passage_index}
```

- 按 Wikipedia 标题分割为 Section
- 每个 Section 按自然段落分割为 Passage（200-400 tokens）
- 相邻短段落合并（< 100 tokens 向前合并）
- 预估总 chunks: ~8,000-12,000

**Embedding 模型:** `BAAI/bge-small-en-v1.5`（384 维，~130MB，CPU 可跑）

**向量存储:**
- Dense: FAISS (IndexFlatIP，内积相似度)
- Sparse: BM25 (rank_bm25 库，内存)
- Metadata: JSON 文件

#### Online Retrieval Pipeline

**接口:**

```python
class RAGTool:
    def retrieve(
        self,
        query: str,
        intent: str,              # "background" | "fact_check" | "gap_fill"
        top_k: int = 5,
        crop_filter: str = None,  # 可选：按作物过滤
        region_filter: str = None # 可选：按地区过滤
    ) -> list[Evidence]:
        """
        Returns: [
            Evidence(
                content="passage text",
                source="Wikipedia: Article_Title, Section",
                relevance_score=0.85,
                metadata={...}
            )
        ]
        """
```

**Intent 处理差异:**

| Intent | 查询改写策略 | Top-K | 相似度阈值 | 调用场景 |
|--------|-------------|-------|-----------|---------|
| `background` | 原始 Question 直接作为 query | 5 | 0.3 (低) | Thinker 初始推理 |
| `fact_check` | 从推理链提取具体事实声明 | 3 | 0.6 (高) | Reviewer-Factual 验证 |
| `gap_fill` | Reviewer 标注的缺失知识点 | 3 | 0.4 (中) | Reviewer/Thinker 补充 |

**检索流程:**

```
query 进入
    │
    ▼
Query Rewriter:
  1. 意图识别 (intent)
  2. 实体提取 (crop/pest/disease/soil)
  3. 查询扩展 (可选): "collard pests"
     → "collard greens Brassica pests silverspotted skipper larvae"
    │
    ▼
Hybrid Retriever:
  Dense (FAISS): embed(query) → top-10 by cosine similarity
  Sparse (BM25): tokenize(query) → top-10 by BM25 score
  Fusion (RRF, k=60): rank_dense + rank_sparse → fused ranking
    │
    ▼
Post-Processor:
  1. Metadata 过滤 (crop/region if given)
  2. 去重 (同一 Section 的多个 Passage 合并)
  3. 相似度阈值过滤
  4. 截断到 top_k
  5. 返回 list[Evidence]
```

#### Agent 调用示例

```
Thinker 调用:
  rag.retrieve(query="common collard issues New Jersey", intent="background", top_k=5)
  → 5 条相关知识 → 用于构建推理链

Reviewer-Factual 调用:
  rag.retrieve(query="Bt effectiveness on silverspotted skipper", intent="fact_check", top_k=3)
  → 3 条精确匹配 → 验证推理链中的事实声明

Reviewer-Domain 调用:
  rag.retrieve(query="collard companion planting", intent="gap_fill", top_k=3)
  → 3 条补充知识 → 判断推理链是否遗漏关键信息
```

#### Key Parameters

| 参数 | 值 |
|------|-----|
| 知识源 | Wikipedia 英文版 (~750-1,000 篇) |
| Chunk 大小 | 200-400 tokens (Passage 级) |
| 预估 Chunks | ~8,000-12,000 |
| Embedding | `BAAI/bge-small-en-v1.5` (384 维) |
| 向量库 | FAISS (IndexFlatIP) |
| 稀疏检索 | BM25 (rank_bm25) |
| 融合策略 | RRF (k=60) |
| 默认 top_k | 5 (background) / 3 (fact_check, gap_fill) |

### Module 2: Thinker (ReAct + Self-Consistency + Faithful CoT)

#### Reasoning Enhancement Methods

Three research-backed methods combined:

| Method | Paper | Role in Thinker |
|--------|-------|----------------|
| **ReAct** | Yao et al. (2023), ICLR | Core reasoning loop: Thought → Action(RAG) → Observation → Thought → ... |
| **Self-Consistency** | Wang et al. (2023), ICLR | Generate 3 reasoning paths, select the best one |
| **Faithful CoT** | Lyu et al. (2023) | Every reasoning step must cite evidence or be logically derived from cited steps |

#### Internal Flow

```
输入: {Question, Answer}
          │
          ▼
┌───────────────────────────────┐
│  Step 1: Background Retrieval  │
│  rag.retrieve(Q, "background") │
│  → Initial Evidence            │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 2: ReAct Generation      │
│  Q + A + Initial Evidence      │
│  → ReAct Loop (max 5 rounds)   │
│  × Self-Consistency (3 samples)│  ← temperature: 0.3 / 0.7 / 1.0
│  → 3 candidate chains          │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 3: Selection             │
│  Evaluator scores 3 chains     │
│  → Select highest-scoring chain│
└──────────────┬────────────────┘
               │
               ▼
Step 4: 输出 Draft Reasoning Chain → 进入 Reviewer 模块
```

#### ReAct Loop Detail

```
Input: Question + Answer + Initial Evidence

Round 1:
  Thought: "Corn planting depth depends on soil type and climate.
            I need to know Georgia's soil characteristics."
  Action:  rag.retrieve("Georgia soil types", intent="background")
  Observation: "Georgia Coastal Plain has sandy loam soils..."

Round 2:
  Thought: "Sandy loam drains fast, need deeper planting for moisture.
            I should verify the recommended depth for sandy soils."
  Action:  rag.retrieve("corn planting depth sandy soil", intent="fact_check")
  Observation: "Recommended 2 inches in sandy soils..."

Round 3:
  Thought: "Enough information gathered. Synthesize reasoning chain."
  Action:  FINISH
  Final:   <structured reasoning chain in JSON>
```

#### Structured Reasoning Chain — Step Types

| Step Type | Meaning | When to Use |
|-----------|---------|-------------|
| `context_setup` | Establish problem context | First step: climate, soil, region |
| `knowledge_application` | Apply domain knowledge | Citing agricultural facts |
| `causal_reasoning` | Cause-effect analysis | "X leads to Y because Z" |
| `comparison` | Compare alternatives | "Sandy vs clay: different depths" |
| `condition_analysis` | Conditional reasoning | "If pH < 5.5, then lime needed" |
| `evidence_integration` | Integrate RAG evidence | "According to Wikipedia: ..." |
| `conclusion` | Tie back to Answer | Final step |

#### Faithful CoT Constraint

```
RULE: Every factual claim in a reasoning step MUST either:
  (a) cite an evidence source, OR
  (b) be a logical inference from previous steps that cite evidence

If unsupported → confidence must be "low"
Never state domain-specific facts without grounding
```

#### Self-Consistency Selection

```
3 samples per QA pair:
  Sample 1 (temperature=0.3): Conservative, evidence-heavy
  Sample 2 (temperature=0.7): Balanced
  Sample 3 (temperature=1.0): Exploratory, more reasoning leaps

Selection criteria (not voting — answer is known):
  - Reasoning step count (3-7 optimal)
  - Evidence utilization rate
  - Logical coherence between steps
  - Answer consistency
```

#### Cost (Course Project: 1,000 items)

| Step | LLM Calls | Note |
|------|-----------|------|
| Background Retrieval | 1,000 × RAG | Non-LLM |
| ReAct × 3 samples | 12,000 | ~4 rounds × 3 |
| Selection scoring | 1,000 | |
| **Thinker Total** | **~13,000** | GPT-4o-mini: ~$4-6 |

### Module 3: Reviewer (Unified Review Module)

#### Responsibility Clarification

| Module | Input | Output | Purpose |
|--------|-------|--------|---------|
| **Thinker** | Q + A | Draft Reasoning Chain | Generate reasoning |
| **Reviewer** | Draft Chain | unified_actions (structured JSON) | Atomic modification指令 |
| **Reviser** | Draft Chain + unified_actions | Revised Chain | Execute structured operations |
| **Evaluator** | Revised Chain | Quality Scores | Quantitative scoring ("how good, 4.2/5") |

Key distinction:
- Reviewer → outputs structured actions (not natural language评语)
- Reviser → executes actions (50% programmatic, 50% LLM)
- Evaluator → scores final quality (diagnostic, not directional)

#### Internal Structure: 3-Phase Review

```
Draft Reasoning Chain (from Thinker)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Phase A: Internal Logic Review (原 Challenger 角色)      │
│                                                          │
│  审查维度:                                                │
│  - Logical Gap:    步骤间是否有推理跳跃                    │
│  - Missing Step:   是否遗漏关键推理步骤                    │
│  - Unsupported:    无证据支撑的事实声明                     │
│  - Overgeneral:    过度泛化的结论                         │
│  - Alternative:    未考虑的替代解释                        │
│                                                          │
│  调用 RAG: 否（纯逻辑分析）                               │
│  输出: logic_critique                                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  Phase B: External Standards Review                      │
│                                                          │
│  审查维度:                                                │
│  - Factual Accuracy:   事实声明是否正确                   │
│    → RAG (fact_check) 逐步验证                           │
│  - Evidence Quality:   引用的证据是否可靠、相关            │
│  - Completeness:       问题的所有子部分是否覆盖            │
│    → RAG (gap_fill) 检索可能遗漏的知识                    │
│  - Domain Expertise:   农业知识运用是否专业                │
│  - Practical Advice:   建议是否可操作、具体                │
│                                                          │
│  调用 RAG: 是 (fact_check + gap_fill)                    │
│  输出: external_critique                                 │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  Phase C: Integration (原 Meta-Reviewer 角色)             │
│                                                          │
│  1. 去重: 多维度指出同一问题 → 合并，severity 取最高       │
│  2. 优先级排序:                                           │
│     P0 (必须改): 事实错误、逻辑矛盾                       │
│     P1 (建议改): 缺失步骤、证据不足                       │
│     P2 (可选):   可操作性优化                             │
│  3. 冲突解决:                                             │
│     Factual="supported" + Domain="not best practice"     │
│     → Domain 优先（事实正确 ≠ 做法推荐）                   │
│                                                          │
│  输出: unified_actions (优先级排序的修改指令)              │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
          进入 Reviser
```

#### Phase A Prompt (Internal Logic)

```
You are a logical reasoning critic. Analyze this agricultural reasoning chain 
for internal logic issues. Do NOT check facts — only check reasoning structure.

For each adjacent step pair (N, N+1):
1. Is there a logical connection?
2. Is there a missing intermediate step?

For all steps:
3. Any contradictions between non-adjacent steps?
4. Any circular reasoning?
5. Does the conclusion follow from the premises?

Output (JSON array):
[{"step": N, "issue_type": "logical_gap|missing_step|contradiction|circular|non_sequitur",
  "description": "...", "severity": "high|medium|low"}]
```

#### Phase B Prompt (External Standards)

```
You are an agricultural domain expert reviewing a reasoning chain. 
Check external standards — facts, evidence, completeness, practicality.

For each step with a factual claim:
1. Is it correct? (check against evidence)
2. Is the cited evidence reliable and relevant?

For the overall chain:
3. Does it cover all sub-questions in the original question?
4. Is agricultural terminology used correctly?
5. Is the advice specific and actionable?

Output (JSON array):
[{"step": N, "dimension": "factual|evidence|completeness|domain|practical",
  "issue": "...", "suggestion": "...", "severity": "high|medium|low"}]
```

#### Phase C Prompt (Integration — Structured Actions)

```
You are an integrator merging critiques from logic and external reviewers.
Output STRUCTURED modification actions, NOT natural language feedback.

Given:
- logic_critique: internal logic issues
- external_critique: external standards issues

AVAILABLE ACTIONS (atomic operations):
- add_evidence:    {"action": "add_evidence", "target_step": N, 
                    "params": {"evidence": "..."}}
- revise_step:     {"action": "revise_step", "target_step": N, 
                    "params": {"revised_content": "..."}}
- insert_step:     {"action": "insert_step", "target_step": N, 
                    "params": {"new_step": {"type": "...", "content": "...", 
                    "evidence": null, "confidence": "medium"}}}
- remove_step:     {"action": "remove_step", "target_step": N, "params": {}}
- merge_steps:     {"action": "merge_steps", "params": {"step_a": N, "step_b": M}}
- adjust_confidence: {"action": "adjust_confidence", "target_step": N, 
                      "params": {"new_confidence": "low|medium|high"}}

RULES:
1. Each action is ATOMIC — one change per action
2. If a step needs both evidence and content revision → TWO separate actions
3. Execution order: remove > revise > insert > add_evidence > adjust_confidence
   (deletions first to avoid index shifts)
4. Do NOT generate vague actions like "clarify" — use specific action types

Output:
{"priority_actions": [
    {"priority": "P0", "action": "add_evidence", "target_step": 3,
     "params": {"evidence": "..."}, "reason": "Factual: unsupported claim"}
], "optional_improvements": [...], "conflicts_resolved": [...]}
```

#### Cost (Reviewer: 1,000 items)

| Phase | LLM Calls | RAG Calls |
|-------|-----------|-----------|
| Phase A: Logic | 1,000 | 0 |
| Phase B: External | 1,000 | ~2,000 (fact_check + gap_fill) |
| Phase C: Integration | 1,000 | 0 |
| **Reviewer Total** | **3,000 LLM** | **~2,000 RAG** |

---

### Module 4: Reviser (Structured Action Executor)

#### Responsibility

Input: Draft Chain + unified_actions (structured JSON from Reviewer Phase C)
Output: Revised Reasoning Chain

**Key design change:** Reviser executes atomic operations, not interprets natural language.
50% of operations are programmatic (no LLM needed).

#### Operation Types

| Operation | Needs LLM? | Description |
|-----------|-----------|-------------|
| `add_evidence` | No | Directly set evidence field on target step |
| `adjust_confidence` | No | Directly set confidence field |
| `remove_step` | No | Delete step and reindex |
| `reorder` | No | Array reorder |
| `revise_step` | **Yes** | Rewrite step content (needs context understanding) |
| `insert_step` | **Yes** | Generate new reasoning step |
| `merge_steps` | **Yes** | Synthesize two steps into one |

#### Execution Logic

```python
def execute_revision(chain, unified_actions):
    actions = unified_actions["priority_actions"]
    # Sort: remove > revise > insert > add_evidence > adjust_confidence
    actions = sort_by_execution_order(actions)
    
    for action in actions:
        op = action["action"]
        step = action["target_step"]
        params = action["params"]
        
        if op == "add_evidence":
            # Programmatic — no LLM
            chain[step]["evidence"] = params["evidence"]
        
        elif op == "adjust_confidence":
            # Programmatic — no LLM
            chain[step]["confidence"] = params["new_confidence"]
        
        elif op == "remove_step":
            # Programmatic — no LLM
            chain.pop(step)
            reindex(chain)
        
        elif op == "revise_step":
            # LLM needed — rewrite step with context
            chain[step] = llm_revise_step(chain, step, params["revised_content"])
        
        elif op == "insert_step":
            # LLM needed — generate new step
            chain.insert(step, params["new_step"])
            reindex(chain)
        
        elif op == "merge_steps":
            # LLM needed — synthesize two steps
            a, b = params["step_a"], params["step_b"]
            chain[a] = llm_merge_steps(chain[a], chain[b])
            chain.pop(b)
            reindex(chain)
    
    # Final check: conclusion must match original Answer
    assert_answer_consistency(chain)
    return chain
```

#### Reviser Prompt (only for LLM-required operations)

```
You are revising a single step in an agricultural reasoning chain.

Context:
- Full reasoning chain (for context)
- The specific operation to perform
- The target step to modify

Rules:
1. Preserve the step's type and logical role in the chain
2. Do NOT change the final conclusion
3. Mark revised steps with confidence="revised"
4. If you need evidence, call RAG (intent="fact_check")

Operation: {operation}
Target step: {step_number}
Modification: {params}

Output: The revised step content (JSON object).
```

#### Cost (Reviser: 1,000 items)

| Operation Type | Count (est.) | LLM Calls | RAG Calls |
|---------------|-------------|-----------|-----------|
| Programmatic (add_evidence, adjust_confidence, remove) | ~60% | 0 | 0 |
| LLM-required (revise, insert, merge) | ~40% | ~400 | ~200 |
| **Reviser Total** | | **~400** | **~200** |

Compared to original design (1,000 LLM calls): **60% reduction**.

---

### Module 5: Evaluator (Scorer)

#### Responsibility

Input: Revised Reasoning Chain (final version)
Output: Quality Scores (written to dataset)

**Evaluator does NOT:**
- Give modification suggestions (that's Reviewer)
- Modify the chain (that's Reviser)

**Evaluator DOES:**
- Produce quantitative scores on 5 dimensions
- Compute PPL
- Compare with original AgThoughts reasoning

#### Scoring Dimensions

| Dimension | Method | Weight | Description |
|-----------|--------|--------|-------------|
| Faithfulness | LLM-as-Judge | 25% | Each step's factual claim supported by evidence? |
| Structure | Automated | 20% | Has context_setup + conclusion, step count 3-7, typed steps? |
| Information Density | Automated | 15% | (total words - formulaic words) / total words |
| Logical Completeness | LLM-as-Judge | 25% | Logical dependency between adjacent steps? |
| Traceability | Automated | 15% | Steps with evidence / total steps |

PPL is computed separately as a filter (not in the weighted score):
- PPL too high → language incoherent → flag for review
- PPL too low → template repetition → flag for review

#### Evaluator Prompt (LLM-as-Judge dimensions)

```
You are evaluating the quality of an agricultural reasoning chain.

Score each dimension 1-5:

1. FAITHFULNESS (1-5):
   - 1: Most claims unsupported
   - 3: Some claims supported, some not
   - 5: Every factual claim cites evidence or is logically derived

2. LOGICAL COMPLETENESS (1-5):
   - 1: Major logical jumps, conclusion doesn't follow
   - 3: Some gaps but overall logical flow is clear
   - 5: Every step logically follows from previous, no gaps

For each step, note:
- Does it have evidence? (yes/no)
- Is the logical connection to the previous step clear? (yes/no)

Output:
{"faithfulness": N, "logical_completeness": N, 
 "step_notes": [{"step": 1, "has_evidence": true, "logic_clear": true}, ...]}
```

#### Automated Metrics

```python
# Structure Score
def structure_score(chain):
    has_context = any(s["type"] == "context_setup" for s in chain)
    has_conclusion = any(s["type"] == "conclusion" for s in chain)
    step_count = len(chain)
    optimal_steps = 3 <= step_count <= 7
    return (has_context + has_conclusion + optimal_steps) / 3

# Information Density
def info_density(chain, original_traces):
    formulaic_words = {"okay", "so", "let me", "I need to", "I should", "right?"}
    chain_text = " ".join(s["content"] for s in chain)
    total = len(chain_text.split())
    filler = sum(1 for w in chain_text.lower().split() if w in formulaic_words)
    return (total - filler) / total if total > 0 else 0

# Traceability
def traceability(chain):
    with_evidence = sum(1 for s in chain if s.get("evidence"))
    return with_evidence / len(chain)

# PPL (using a language model)
def compute_ppl(chain_text, model):
    # Standard perplexity computation
    return model.perplexity(chain_text)
```

#### Quality Gate

```
overall_score = weighted_sum(scores)

if overall_score < 3.0:
    → send back to Reviewer Phase B + Reviser (max 1 iteration)
    → if still < 3.0 after revision → mark as "low_quality" in dataset

if overall_score >= 3.0:
    → accept → write to dataset
```

#### Cost (Evaluator: 1,000 items)

| Step | LLM Calls | Note |
|------|-----------|------|
| LLM-as-Judge scoring | 1,000 | 2 dimensions in 1 call |
| PPL computation | 0 | Non-LLM (local model) |
| **Evaluator Total** | **1,000** | |

---

## Output Dataset Schema

Dataset contains BOTH original and enhanced reasoning chains for direct comparison.

```json
{
    "id": "agthoughts_0001",
    "Question": "string",
    "Answer": "string (from AgThoughts, unchanged)",
    "Question_Type": "string",

    "Original_Reasoning": {
        "content": "string (original AgThoughts Reasoning Traces)",
        "word_count": 0,
        "source": "AgThoughts original"
    },

    "Enhanced_Reasoning": {
        "chain": [
            {
                "step": 1,
                "type": "context_setup | knowledge_application | causal_reasoning | comparison | condition_analysis | evidence_integration | conclusion",
                "content": "reasoning text",
                "evidence": "supporting knowledge snippet or null",
                "confidence": "high | medium | low"
            }
        ],
        "evidence_utilization": 0.0,
        "step_count": 0,
        "word_count": 0,
        "react_rounds": 0,
        "self_consistency_selected": 0
    },

    "Quality_Comparison": {
        "original_score": {
            "faithfulness": 0.0,
            "structure": 0.0,
            "information_density": 0.0,
            "logical_completeness": 0.0,
            "traceability": 0.0,
            "overall": 0.0
        },
        "enhanced_score": {
            "faithfulness": 0.0,
            "structure": 0.0,
            "information_density": 0.0,
            "logical_completeness": 0.0,
            "traceability": 0.0,
            "overall": 0.0
        },
        "improvement": "+X%"
    },

    "Critique_History": [
        {
            "round": 1,
            "reviewer": "Challenger | Reviewer-Factual | Reviewer-Logical",
            "issues": [
                {
                    "target_step": 3,
                    "issue_type": "logical_gap | missing_step | unsupported_claim | overgeneralization | alternative_ignored",
                    "question": "Socratic question exposing the weakness",
                    "severity": "high | medium | low"
                }
            ]
        }
    ],

    "Metadata": {
        "pipeline_version": "string",
        "llm_model": "string",
        "rag_source": "Wikipedia",
        "processing_time_sec": 0.0
    }
}
```

---

## Evaluation Methodology

### Quality Dimensions (5 dimensions)

| Dimension | Original AgThoughts Problem | Our Improvement |
|-----------|---------------------------|-----------------|
| Faithfulness | Factual claims unsupported | Each step cites evidence via RAG |
| Structure | Continuous text, no step划分 | 7 typed steps, interpretable |
| Information Density | Formulaic ("Okay, so I need to...") | No filler, each step carries reasoning |
| Logical Completeness | Logical jumps, missing steps | Challenger + Reviser fill gaps |
| Traceability | No way to trace reasoning basis | Each step linked to evidence source |

### Quantitative Evaluation

| Metric | Measurement | Method |
|--------|-------------|--------|
| Faithfulness Score | Factual claims with evidence support | LLM-as-Judge per step |
| Structure Score | Has context_setup + conclusion, step count 3-7 | Automated check |
| Information Density | (total words - formulaic words) / total words | Automated |
| Logical Completeness | Logical dependency between adjacent steps | LLM-as-Judge |
| Traceability | Steps with evidence / total steps | Automated |
| Answer Consistency | Semantic similarity of conclusion to Answer | sentence-transformers |
| PPL | Perplexity of reasoning chain text | Language model |

### Qualitative Evaluation (Case Study)

Select 10-20 representative cases with detailed side-by-side comparison:
- Show original formulaic reasoning vs enhanced structured reasoning
- Highlight specific improvements: evidence grounding, logical completeness, information density

### Human Evaluation

50 samples, blind evaluation:
- Evaluator sees Question + Answer + two anonymous reasoning chains (A/B)
- Rate each on 1-5: persuasiveness, factual accuracy, logical clarity, trustworthiness

### Ablation Study

| Configuration | RAG | Self-Consistency | Reviewer | Expected |
|---------------|-----|-----------------|----------|----------|
| Full Pipeline | Yes | Yes (×3) | Yes (3 phases) | Best |
| w/o Reviewer | Yes | Yes (×3) | No | Slight drop |
| w/o Self-Consistency | Yes | No | Yes | Medium drop |
| w/o RAG | No | Yes (×3) | Yes | Significant drop |
| Baseline (AgThoughts original) | — | — | — | Lowest |

---

## Optimization Strategies

1. **Dynamic Routing (核心优化):** Difficulty Classifier → Easy/Medium/Hard paths, ~47% cost reduction
2. **Feedback Loop:** Evaluator score < 3.0 → send back to Reviewer Phase B + Reviser (max 1 iteration)
3. **Parallelization:** Reviewer Phase A and Phase B can run in parallel
4. **Structured reasoning chain:** 7 typed step sequences for interpretability, evaluability, and editability

---

## Course Project Scale

**Data scale:** Subsample 1,000-2,000 items from AgThoughts (stratified by Question Type)

**Pipeline simplification:**
- Reviewer: 3 phases in 1 module (not 8 separate reviewers)
- Max 1 iteration feedback loop (not 2)
- 50 human-annotated samples for calibration (not 500)

**Estimated cost (1,000 items, with dynamic routing + structured actions):**

| Module | LLM Calls | RAG Calls | GPT-4o-mini Cost |
|--------|-----------|-----------|-----------------|
| Difficulty Classifier | 1,000 | 0 | ~$0.3 |
| Thinker (Easy/Med/Hard) | ~5,400 | ~2,500 | ~$2-3 |
| Reviewer (1/2/3 phases) | ~1,800 | ~1,200 | ~$0.8 |
| Reviser (structured, 40% LLM) | ~200 | ~100 | ~$0.1 |
| Evaluator (Med/Hard only) | ~500 | 0 | ~$0.15 |
| **Total** | **~8,900** | **~3,800** | **~$3-4** |

- Compared to no optimization (18,000 calls, $6-9): **~50% savings**
- Structured actions reduce Reviser LLM calls by 60% (from ~500 to ~200)
- Embedding: < $0.1
- Knowledge base: free (Wikipedia API)
- Local model alternative: $0 (Qwen2.5-7B)

**File structure:**

```
project/
├── data/
│   ├── raw/                     # Wikipedia 原始数据
│   ├── agthoughts/              # 采样后的 AgThoughts 数据
│   │   ├── train.json
│   │   └── sample_1000.json
│   ├── chunks/                  # 分块后的 passages
│   │   └── passages.jsonl       # {id, text, metadata}
│   └── index/
│       ├── faiss.index          # FAISS 向量索引
│       ├── bm25.pkl             # BM25 模型
│       └── metadata.json        # chunk metadata
├── src/
│   ├── rag/
│   │   ├── knowledge_builder.py # 知识库构建 (Wikipedia → chunks → index)
│   │   ├── retriever.py         # Hybrid Retriever (FAISS + BM25 + RRF)
│   │   ├── query_processor.py   # 查询改写 + 意图处理
│   │   └── rag_tool.py          # RAGTool 对外接口
│   ├── thinker/
│   │   ├── react_generator.py   # ReAct 推理循环
│   │   └── consistency.py       # Self-Consistency 采样 + 选择
│   ├── reviewer/
│   │   ├── logic_reviewer.py    # Phase A: 内在逻辑审查
│   │   ├── external_reviewer.py # Phase B: 外部标准审查
│   │   └── integrator.py        # Phase C: 整合 + 优先级排序
│   ├── reviser/
│   │   └── reviser.py           # 按 unified_actions 修改推理链
│   ├── evaluator/
│   │   ├── llm_judge.py         # LLM-as-Judge (faithfulness + completeness)
│   │   ├── auto_metrics.py      # 自动化指标 (structure + density + traceability)
│   │   └── ppl_scorer.py        # PPL 计算
│   ├── pipeline.py              # 整体 Pipeline 编排
│   └── config.py                # 配置 (模型、参数、路径)
├── output/
│   └── enhanced_dataset.json    # 最终输出数据集
├── docs/
│   └── architecture_design.md
└── tests/
```

---

## Academic Positioning

**Title direction:** "ReAct-Guided Multi-Agent Reasoning Augmentation for Agricultural QA"

**Innovation points:**
1. Dynamic routing by difficulty (easy/medium/hard paths, ~47% cost reduction)
2. ReAct-based reasoning generation (interleaving Thought and RAG Action, not one-shot generation)
3. Structured reasoning chains (7 typed step sequences, not free-text CoT)
4. Faithful CoT (every step must cite evidence or derive logically from cited steps)
5. Self-Consistency for reasoning chain selection (multiple paths, score-based selection)
6. 4-stage pipeline with clear role separation: Thinker → Reviewer → Reviser → Evaluator
7. 3-phase unified Reviewer (logic + external standards + integration)
8. Dual-reasoning dataset format (original + enhanced for direct comparison)

**Key Related Work:**
- Chain-of-Thought: Wei et al. (2022), NeurIPS
- ReAct: Yao et al. (2023), ICLR
- Self-Consistency: Wang et al. (2023), ICLR
- Faithful CoT: Lyu et al. (2023)
- CoT Distillation: Hsieh et al. (2023), "Distilling Step-by-Step"
- Multi-Agent Debate: Du et al. (2023)
- Self-Refine: Madaan et al. (2023)
- Socratic Models: Zeng et al. (2022)
- LLM-as-Judge: Zheng et al. (2023)
- Toolformer: Schick et al. (2023)
- Self-RAG: Jiang et al. (2023)
- Orca: Mukherjee et al. (2023)
- WizardMath / MAmmoTH: Li et al. (2023)
