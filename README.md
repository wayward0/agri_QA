# AgriQA — 农业问答推理链增强系统

基于多智能体协作的农业问答推理链自动生成与质量优化系统。输入农业 QA 对，输出带有结构化推理链、证据引用和质量评分的增强数据集。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Orchestrator                        │
│                          (pipeline.py)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌──────────┐
│  │Classifier│──▶│ Thinker  │──▶│ Reviewer │──▶│Reviser │──▶│Evaluator │
│  │ Stage 0  │   │ Stage 1  │   │ Stage 2  │   │Stage 3 │   │ Stage 4  │
│  └──────────┘   └──────────┘   └──────────┘   └────────┘   └──────────┘
│       │              │               │              │            │
│  deepseek-v4    deepseek-v4     deepseek-v4   deepseek-v4  deepseek-v4
│    -flash         -pro            -pro          -flash        -pro
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              RAG Knowledge Base (Graph-Enhanced)              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │  │  FAISS  │  │  BM25   │  │ RRF Fusion│  │  Reranker    │  │   │
│  │  │ (dense) │  │(sparse) │  │  (hybrid) │  │(bge-reranker)│  │   │
│  │  └─────────┘  └─────────┘  └──────────┘  └──────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Knowledge Graph (entities + relations + 1-hop traversal) │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
输入: QA 对 {Question, Answer, Question Type}
  │
  ▼
Stage 0: 难度分类 (Classifier)
  │  单次 LLM 调用, temperature=0.0
  │  输出: EASY / MEDIUM / HARD
  │
  ▼
Stage 1: 推理生成 (Thinker)
  │  ReAct 框架: Thought → Action(RAG) → Observation 循环
  │  RAG 作为按需工具: LLM 自主决定何时 retrieve，无初始强制检索
  │  HARD 题: 自洽性采样 (3 条路径并行, 温度 [0.3, 0.7, 1.0], 择优)
  │  其他: 单路径
  │  苏格拉底自质疑: 生成挑战性问题 → 修正推理链 (含按需 RAG 补证)
  │  输出: ReasoningChain (draft_chain)
  │
  ▼
Stage 2: 多阶段审查 (Reviewer)
  │  Phase A: 逻辑审查 (纯 LLM, 无 RAG)  ─┐ MEDIUM/HARD 并行
  │  Phase B: 事实核查 (RAG 增强)         ─┘
  │  Phase C: 整合为原子操作 (串行, 依赖 A+B)
  │  输出: UnifiedActions + ReviewCritique
  │
  ▼
Stage 3: 修订执行 (Reviser)
  │  按优先级执行原子操作:
  │    remove_step → merge_steps → revise_step
  │    → insert_step → add_evidence → adjust_confidence
  │  ~50% 操作为程序化 (无需 LLM)
  │  输出: ReasoningChain (revised_chain)
  │
  ▼
Stage 4: 质量评估 (Evaluator)
  │  LLM-as-Judge: 忠实度 (1-5), 逻辑完整性 (1-5)
  │  自动指标: 结构得分, 信息密度, 可追溯性
  │  加权总分 = faith*0.25 + structure*0.15 + density*0.10
  │            + logic*0.25 + traceability*0.15 + step_order*0.10
  │  输出: QualityScores
  │
  ▼
质量门控:
  │  HARD: overall < 3.0 → Review + Revise + Evaluate (带扣分诊断反馈)
  │  MEDIUM: faithfulness < 3.0 → 强制修订 (单维强门控)
  │
  ▼
输出: PipelineItem (包含全部中间结果)
```

## 目录结构

```
agri_QA/
├── config.py                   # 全局配置 (环境变量覆盖)
├── models.py                   # 数据模型 (Protocol + dataclass)
├── pipeline.py                 # Pipeline 编排器 (唯一入口)
├── llm_client.py               # LLM 调用抽象 (OpenAI 兼容)
├── rebuild_indices.py          # 重建 FAISS + BM25 索引 + KG (支持 --kg-only)
├── eval_quantitative.py        # 量化评估: 原始推理 vs 增强推理对比
├── fetch_from_dumps.py         # Wikipedia dump 离线解析器
├── run_fetch.py                # Wikipedia API 在线抓取器
├── AgThoughts.json             # 源 QA 数据集 (44,615 条)
├── requirements.txt            # Python 依赖
├── pytest.ini                  # 测试配置
│
├── classifier/                 # Stage 0: 难度分类
│   ├── __init__.py
│   └── difficulty_classifier.py
├── thinker/                    # Stage 1: 推理生成
│   ├── __init__.py
│   ├── react_generator.py      #   ReAct 推理循环
│   ├── self_consistency.py     #   自洽性多路径采样
│   └── socratic_challenger.py  #   苏格拉底自质疑 + 修正
├── reviewer/                   # Stage 2: 多阶段审查
│   ├── __init__.py
│   ├── logic_reviewer.py       #   Phase A: 逻辑审查
│   ├── external_reviewer.py    #   Phase B: 事实核查
│   └── integrator.py           #   Phase C: 整合
├── reviser/                    # Stage 3: 修订执行
│   ├── __init__.py
│   └── reviser.py
├── evaluator/                  # Stage 4: 质量评估
│   ├── __init__.py
│   ├── auto_metrics.py         #   自动指标 (结构/密度/可追溯性)
│   ├── llm_judge.py            #   LLM-as-Judge (忠实度/逻辑完整性)
│   └── ppl_scorer.py           #   困惑度评分 (诊断指标, DeepSeek logprobs)
│
├── rag/                        # RAG 检索增强
│   ├── __init__.py
│   ├── rag_tool.py             #   统一检索接口 (含 Graph 扩展)
│   ├── retriever.py            #   混合检索: FAISS + BM25 + RRF
│   ├── embedding_client.py     #   Embedding API 客户端 (bge-m3)
│   ├── reranker.py             #   Reranker API 客户端 (bge-reranker-v2-m3)
│   ├── knowledge_builder.py    #   知识库构建器 + Wikipedia 抓取
│   ├── kg_builder.py           #   知识图谱构建 (LLM 实体/关系抽取)
│   ├── kg_index.py             #   知识图谱索引 (实体匹配 + 图扩展)
│   └── query_processor.py      #   查询改写 + 实体抽取
│
├── tests/                      # 单元测试
├── test_pipeline_run.py        # Pipeline 端到端测试
├── test_react_debug.py         # ReAct 推理调试
├── test_react_prompt.py        # ReAct prompt 测试
├── test_max_tokens.py          # Token 上限测试
├── docs/                       # 文档
│   └── ...
│
├── data/
│   ├── agthoughts/sample_1000.json   # 分层抽样 1000 条
│   ├── chunks/passages.jsonl         # 分块后的 Wikipedia 段落
│   ├── raw/                          # Wikipedia 原始数据 (dump/抓取)
│   └── index/
│       ├── faiss.index               # FAISS 密集索引 (1024 维)
│       ├── bm25.pkl                  # BM25 稀疏索引
│       ├── metadata.json             # 段落元数据
│       ├── kg_entities.json          # 知识图谱实体 (8 类)
│       ├── kg_relations.json         # 知识图谱关系 (8 类)
│       ├── entity_faiss.index        # 实体名向量索引
│       └── entity_faiss_map.json     # 实体索引映射
│
└── output/
    └── enhanced_dataset.json         # Pipeline 输出 (含 checkpoint)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- `openai` — LLM API 调用
- `faiss-cpu` — 向量检索
- `rank-bm25` — BM25 稀疏检索
- `tqdm` — 进度条

### 2. 配置 API

编辑 `config.py` 或设置环境变量:

```bash
# LLM API
export LLM_API_BASE_URL="https://ai.centos.hk/v1"
export AI_CENTOS_API_KEY="your-key"

# 模型分层
export MODEL_LIGHT="deepseek-v4-flash"      # 分类/修订
export MODEL_STANDARD="deepseek-v4-pro"     # 审查/评估
export MODEL_PREMIUM="deepseek-v4-pro"      # 核心推理

# Embedding API
export EMBEDDING_API_BASE_URL="https://api.moark.com/v1"
export EMBEDDING_API_KEY="your-key"
export EMBEDDING_MODEL="bge-m3"

# Reranker API
export RERANKER_API_URL="https://api.moark.com/v1/rerank"
export RERANKER_API_KEY="your-key"
export RERANKER_MODEL="bge-reranker-v2-m3"
```

### 3. 构建知识库索引

```bash
# 从 Wikipedia 抓取农业文章 (需要网络访问)
python run_fetch.py

# 重建 FAISS + BM25 索引
python rebuild_indices.py
```

### 4. 运行 Pipeline

```bash
# 运行完整 pipeline (异步并发，默认 10 并发)
python pipeline.py

# 或在 Python 中调用
python -c "
from pipeline import run_pipeline
run_pipeline(n_items=100, resume=True)
"
```

输出保存在 `output/enhanced_dataset.json`，每个 item 处理后自动 checkpoint。

## 核心模块详解

### 难度分类器 (classifier/)

单次 LLM 调用，将 QA 对分为 EASY / MEDIUM / HARD 三级。使用正则词边界匹配解析输出，取最后一个匹配结果以处理 LLM "先推理再给答案" 的输出模式。

```python
from classifier.difficulty_classifier import classify_difficulty
result = classify_difficulty(question, answer, llm_call)
# result.difficulty: DifficultyLevel.EASY / MEDIUM / HARD
```

**难度路由策略:**

| 难度 | Thinker | Reviewer | Reviser | 质量门控 |
|------|---------|----------|---------|---------|
| EASY | 单路径 + 苏格拉底质疑 | Phase A 逻辑+对齐审查 | 跳过 | 无 |
| MEDIUM | 单路径 + 苏格拉底质疑 | Phase A + B (含事实核查) | 执行 | faithfulness < 3.0 强制重来 |
| HARD | 3 路径自洽性 + 苏格拉底质疑 | Phase A + B + C | 执行 | overall < 3.0 重来 (带反馈) |

### 推理生成器 (thinker/)

**ReAct 框架** (`react_generator.py`): Thought → Action → Observation 循环

```
Question + Answer 进入 ReAct 循环 (无初始 RAG 检索)

Round 1:
  Thought: "需要了解作物的土壤需求"
  Action:  retrieve: "corn soil requirements sandy loam"
  Observation: [RAG 返回的证据]

Round 2:
  Thought: "已有足够信息构建推理链"
  Action:  FINISH
  → 输出结构化 JSON 推理链
```

**RAG-as-Tool 设计**: RAG 是纯按需工具，不由 pipeline 预设调用时机。LLM 在 Thought 阶段自主判断是否需要检索，通过 `retrieve: <query>` Action 发起调用。ReAct 调用 max_tokens=4096，为 DeepSeek 内部思考 + 可见输出留足空间。

**自洽性采样** (`self_consistency.py`): 对 HARD 题**并行**生成 3 条路径 (温度 0.3/0.7/1.0，ThreadPoolExecutor 并发)，两阶段择优:

```
路径 1 (temp=0.3) ─┐
路径 2 (temp=0.7) ─┤  提取各路径的最终结论
路径 3 (temp=1.0) ─┘
        │
        ▼
语义聚类 (LLM): 哪些结论本质相同?
  → Cluster A: [路径 1, 路径 3] "推荐生物防治"  ← 多数派
  → Cluster B: [路径 2] "推荐化学防治"
        │
        ▼
从多数派聚类中选启发式得分最高者
  → 路径 1 (结构分更高)
```

启发式评分标准 (在聚类之后应用):
- 步骤数 (3-7 步最优, 权重 0.30)
- 证据利用率 (0.35)
- 类型多样性 (0.20)
- 高置信度比例 (0.15)

**苏格拉底自质疑** (`socratic_challenger.py`): 在自洽性选择后执行，聚焦推理链的**内省** — 检查自身完备性和领域严谨性:

```
选出的最优推理链
  │
  ▼
Socratic Challenge: 生成 3-5 个质疑问题 (聚焦内省)
  - missing_edge_case: 遗漏的领域边界条件 (土壤类型、气候带、生长阶段)
  - overgeneralization: 过度泛化的结论 ("所有作物" vs 实际仅部分适用)
  - alternative_ignored: 未考虑的替代方案 (生物防治 vs 化学防治)
  - domain_constraint: 缺失的领域约束 (pH 范围、温度阈值、药剂兼容性)
  │
  ▼
Revision: 基于质疑修正推理链
  - 可按需调用 RAG 补充证据 (retrieve: <query>)
  - 保留未被质疑的步骤
  │
  ▼
输出: 修正后的 ReasoningChain (draft_chain)
```

> **与 Reviewer 的分工**: Socratic 聚焦"内省"（推理链自身的完备性），Reviewer 聚焦"对齐"（与原始 QA 的语义一致性和结构逻辑）。两者不重复检查。

### 审查器 (reviewer/)

三阶段审查，逐级深入:

**Phase A — 逻辑 + 语义对齐审查** (`logic_reviewer.py`):
- 结构逻辑: 检查相邻步骤间的逻辑缺口、矛盾、循环推理
- 语义对齐: 检查推理链结论是否与原始 Answer 发生语义漂移
- 纯 LLM 分析，不调用 RAG

**Phase B — 事实核查** (`external_reviewer.py`):
- 对含事实断言的步骤 (knowledge_application, causal_reasoning, evidence_integration) 检索 RAG 证据
- 检查: 事实正确性、证据可靠性、完整性、术语准确性、可操作性

**Phase C — 整合** (`integrator.py`):
- 将 Phase A/B 的审查意见合并为结构化原子操作
- 操作优先级: P0 (必须修复: 事实错误、矛盾) → P1 (应修复: 缺失步骤) → P2 (可选: 风格)
- MEDIUM/HARD: Phase A 与 Phase B **并行执行** (ThreadPoolExecutor)，Phase C 等待两者完成后串行整合

### 修订器 (reviser/)

按特定顺序执行原子操作，避免索引冲突:

1. `remove_step` — 删除步骤 (程序化)
2. `merge_steps` — 合并步骤 (LLM)
3. `revise_step` — 修改步骤 (LLM)
4. `insert_step` — 插入步骤 (LLM)
5. `add_evidence` — 补充证据 (程序化)
6. `adjust_confidence` — 调整置信度 (程序化)

约 50% 的操作为程序化执行，无需 LLM 调用。

### 评估器 (evaluator/)

**自动指标** (`auto_metrics.py`): 纯函数，无外部依赖
- `structure_score`: context_setup/conclusion 存在、步骤数 3-7、内容非空、conclusion 在末位
- `information_density`: 非公式化词汇占比 (自动检测中英文，分别使用停用词表)
- `traceability`: 有证据引用的步骤占比
- `step_order_score`: context_setup 在前、conclusion 在后且不早于第 3 步

**LLM-as-Judge** (`llm_judge.py`):
- `faithfulness` (1-5): 每个事实断言是否有证据支持
- `logical_completeness` (1-5): 推理链是否覆盖所有必要步骤

**PPL 诊断** (`ppl_scorer.py`):
- 通过 DeepSeek API 的 logprobs 计算平均负对数似然，作为困惑度代理指标
- 使用 `deepseek-v4-flash` (LIGHT 层级) 以降低成本
- 仅作为诊断指标，不参与 overall 加权总分
- PPL 越低表示模型对文本的预测置信度越高 (更流畅)

**加权总分:**

```
overall = faithfulness/5 × 0.25
        + structure × 0.15
        + information_density × 0.10
        + logical_completeness/5 × 0.25
        + traceability × 0.15
        + step_order × 0.10
```

### 质量门控与反馈流

```
评估结果 (scores + step-level notes)
  │
  ├─ HARD: overall < 3.0 ─────────┐
  └─ MEDIUM: faithfulness < 3.0 ──┤
                                   ▼
                    构建扣分诊断报告
                    (哪些步骤 faithfulness 低? 哪些逻辑不完整?)
                                   │
                                   ▼
                    反哺给 Reviewer Phase C (Integrator)
                    Integrator 优先修复 Evaluator 标记的具体步骤
                                   │
                                   ▼
                    Reviser 执行针对性修订
                                   │
                                   ▼
                    Evaluator 重新评估
```

关键: Reviewer 第二次审查时不再是"盲目修订"，而是拿到 Evaluator 的逐步扣分诊断，对症下药。

### 设计原则: RAG-as-Tool

RAG 是所有 Agent 共享的按需工具，而非 pipeline 中的固定阶段。每个 Agent 在需要证据时自主调用 RAG，由 LLM 决定调用时机:

- **Thinker**: 在 ReAct 循环中通过 `retrieve: <query>` 按需检索
- **Socratic Challenger**: 在修正推理链时按需补充证据
- **Reviewer Phase B**: 逐步事实核查时按需检索
- **Evaluator**: 验证断言时按需检索

**批量检索** (`retrieve_batch`): 当多个查询可同时发出时 (如 Reviewer Phase B 对多个步骤的事实核查)，通过单次 embedding API 调用编码所有查询，再逐个执行 FAISS/BM25/Reranker，减少网络往返。

### 混合检索架构

```
Query
  │
  ├─▶ FAISS (密集检索) ──┐
  │   bge-m3, 1024 维     ├─▶ RRF 融合 ─▶ Reranker ─▶ Top-K
  └─▶ BM25 (稀疏检索) ──┘   (k=60)    (bge-reranker)
```

**密集检索**: FAISS IndexFlatIP，使用 bge-m3 API 编码 (1024 维)

**稀疏检索**: BM25Okapi，基于词频的稀疏匹配

**融合**: Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ 1/(k + rank_i(d))  对每个检索器的排名
```

**重排序**: bge-reranker-v2-m3 交叉编码器，对 RRF 融合后的候选集进行精排 (最多 25 条)

### Graph RAG: 知识图谱增强检索

在基础混合检索之上，增加知识图谱层实现关系感知的查询扩展。对所有现有调用点完全透明，无需修改 thinker/reviewer/reviser 代码。

**实体类型 (8 类):** crop, disease, pest, chemical, practice, nutrient, soil, climate

**关系类型 (8 类):** affects, causes, treats, prevents, requires, interacts_with, found_in, applied_to

**离线构建** (`kg_builder.py`): LLM 从 Wikipedia 文章中批量抽取实体和关系，按 article 分批保持上下文。10 个并行 worker (ThreadPoolExecutor)，支持断点续传 (每 50 篇保存 checkpoint)。去重后保存为 JSON，实体名+别名构建 FAISS 索引用于查询时向量匹配。

**查询时图扩展** (`kg_index.py`):

```
Query → 实体匹配 (entity_faiss 向量搜索)
      → 1-hop 图遍历 (收集关联实体的 passage_ids)
      → 加权排序 (match_score × confidence × 0.5 距离衰减)
      → 合并基础检索结果 (graph 结果 +20% 分数提升)
      → Top-K
```

**集成方式**: `RAGTool.retrieve()` 内部自动执行图扩展，对外接口不变:

```python
rag_tool.retrieve("fusarium wilt treatment", intent="background", top_k=5)
# 内部: 基础混合检索 + KG 图扩展 → 合并去重 → 返回
```

**构建 KG:**

```bash
# rebuild_indices.py 已集成 KG 构建步骤
python rebuild_indices.py
# 输出: data/index/kg_entities.json, kg_relations.json, entity_faiss.index
```

### Embedding 客户端

```python
from rag.embedding_client import EmbeddingClient

embedder = EmbeddingClient(
    base_url="https://api.moark.com/v1",
    api_key="your-key",
    model="bge-m3"
)
vectors = embedder.encode(["query text"], normalize_embeddings=True)
# 返回 1024 维向量
```

### Reranker 客户端

```python
from rag.reranker import RerankerClient

reranker = RerankerClient(
    base_url="https://api.moark.com/v1/rerank",
    api_key="your-key",
    model="bge-reranker-v2-m3"
)
results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
# 返回 [{"index": 0, "relevance_score": 0.95}, ...]
```

## 模型分层策略

为平衡成本与质量，不同阶段使用不同模型:

| 阶段 | 模型层级 | 模型 | 理由 |
|------|---------|------|------|
| Classifier | LIGHT | deepseek-v4-flash | 简单分类，快速低成本 |
| Thinker | PREMIUM | deepseek-v4-pro | 核心推理，需最高质量 |
| Reviewer | STANDARD | deepseek-v4-pro | 审查整合，需均衡质量 |
| Reviser | LIGHT | deepseek-v4-flash | 执行结构化操作，快速即可 |
| Evaluator | STANDARD | deepseek-v4-pro | 质量评估，需准确判断 |

所有模型通过 `create_stage_caller()` 绑定默认模型，支持按调用覆盖:

```python
from llm_client import create_llm_caller, create_stage_caller

base = create_llm_caller(api_key="...", model="deepseek-v4-pro")
classifier = create_stage_caller(base, "deepseek-v4-flash")
# classifier 使用 deepseek-v4-flash，可通过 model= 参数覆盖
```

## 数据模型

所有类型定义在 `models.py`，模块间通过类型而非导入耦合:

| 类型 | 说明 | 关键字段 |
|------|------|---------|
| `DifficultyLevel` | 难度枚举 | EASY, MEDIUM, HARD |
| `Evidence` | RAG 检索到的证据 | content, source, relevance_score |
| `ReasoningStep` | 推理链单步 | step, type (7种), content, evidence, confidence |
| `ReasoningChain` | 完整推理链 | steps, react_rounds, self_consistency_selected |
| `QualityScores` | 质量评分 (五维 + PPL 诊断) | faithfulness, structure, density, logic, traceability, overall, ppl |
| `ReviewCritique` | 审查意见 | issues (列表), phase |
| `ReviewAction` | 原子审查操作 | action, target_step, priority, params, reason |
| `UnifiedActions` | 整合后的操作集 | priority_actions, optional_improvements, conflicts_resolved |
| `PipelineItem` | 流经 pipeline 的完整 item | 含全部中间结果和元数据 |
| `LLMCallFn` | LLM 调用协议 | `(prompt, system, temperature, max_tokens, model) -> str` |

**推理步骤类型:** context_setup, knowledge_application, causal_reasoning, comparison, condition_analysis, evidence_integration, conclusion

**审查操作类型:** add_evidence, revise_step, insert_step, remove_step, merge_steps, adjust_confidence

## 配置参数

所有参数在 `config.py` 中定义，支持环境变量覆盖:

### API 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_BASE_URL` | `https://ai.centos.hk/v1` | LLM API 地址 |
| `AI_CENTOS_API_KEY` | (空) | LLM API Key |
| `LLM_MAX_TOKENS` | 4096 | LLM 最大输出 token 数 |

### RAG 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_API_BASE_URL` | `https://api.moark.com/v1` | Embedding API 地址 |
| `EMBEDDING_API_KEY` | (空) | Embedding API Key |
| `EMBEDDING_MODEL_NAME` | `bge-m3` | Embedding 模型名 (环境变量: `EMBEDDING_MODEL`) |
| `EMBEDDING_DIM` | 1024 | Embedding 向量维度 |
| `RERANKER_API_URL` | `https://api.moark.com/v1/rerank` | Reranker API 地址 |
| `RERANKER_API_KEY` | (空) | Reranker API Key |
| `RERANKER_MODEL` | `bge-reranker-v2-m3` | Reranker 模型名 |
| `RAG_TOP_K_BACKGROUND` | 5 | 背景检索返回条数 |
| `RAG_TOP_K_FACT_CHECK` | 3 | 事实核查返回条数 |
| `RAG_TOP_K_GAP_FILL` | 3 | 缺口填充返回条数 |
| `RRF_K` | 60 | RRF 融合常数 |
| `CHUNK_MIN_TOKENS` | 200 | 最小分块长度 |
| `CHUNK_MAX_TOKENS` | 400 | 最大分块长度 |

### 知识图谱参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `KG_TOP_K_ENTITIES` | 5 | 查询时实体匹配数量 |
| `KG_MAX_GRAPH_PASSAGES` | 10 | 图扩展最大返回 passage 数 |

### Thinker 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REACT_MAX_ROUNDS` | 5 | ReAct 最大循环轮数 |
| `SELF_CONSISTENCY_N` | 3 | 自洽性采样路径数 |
| `REACT_TEMPERATURES` | [0.3, 0.7, 1.0] | 采样温度列表 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QUALITY_GATE_THRESHOLD` | 3.0 | HARD 题 overall 门控阈值 |
| `FAITHFULNESS_GATE_THRESHOLD` | 3.0 | MEDIUM 题 faithfulness 强门控阈值 |
| `MAX_REVISION_ITERATIONS` | 1 | 最大修订迭代次数 |
| `EVAL_WEIGHTS` | faith:0.25, struct:0.15, density:0.10, logic:0.25, trace:0.15, step_order:0.10 | 六维权重 |

### Pipeline 控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_CONCURRENCY` | 10 | 异步并发 item 数量 |
| `API_CALL_INTERVAL` | 0.5s | item 间限流间隔 (仅串行模式, 异步模式下未使用) |
| `CONSECUTIVE_FAILURE_LIMIT` | 3 | 连续失败上限 |

## 输出格式

`output/enhanced_dataset.json` 为 JSON 数组，每个元素:

```json
{
  "id": "item_0000",
  "question": "How do I apply phosphorus fertilizer...",
  "answer": "To address your bell peppers' needs...",
  "question_type": "Crop Management Questions",
  "difficulty": "hard",
  "draft_chain": {
    "steps": [
      {
        "step": 1,
        "type": "context_setup",
        "content": "...",
        "evidence": "...",
        "confidence": "high"
      }
    ],
    "react_rounds": 3,
    "self_consistency_selected": true
  },
  "revised_chain": { "steps": [...] },
  "quality_scores": {
    "faithfulness": 3.0,
    "structure": 1.0,
    "information_density": 0.969,
    "logical_completeness": 3.0,
    "traceability": 0.667,
    "overall": 0.745,
    "ppl": 45.2
  },
  "critique_history": [...],
  "metadata": {
    "classification_raw": "hard",
    "n_samples": 3,
    "models": {
      "classifier": "deepseek-v4-flash",
      "thinker": "deepseek-v4-pro",
      "reviewer": "deepseek-v4-pro",
      "reviser": "deepseek-v4-flash",
      "evaluator": "deepseek-v4-pro"
    }
  }
}
```

## 知识库构建

### 方式一: 在线抓取 (需 Wikipedia API 访问)

```bash
python run_fetch.py
```

通过 Wikipedia Category API 发现农业文章标题，REST API 抓取内容，自动进行相关性过滤。支持断点续传: 标题发现阶段保存 checkpoint (已访问分类 + 已发现标题)，中断后重新运行自动跳过已完成的分类。

### 方式二: 离线解析 (从 dump 文件)

```bash
python fetch_from_dumps.py
```

从 Wikipedia multistream dump (.bz2) 中解析农业文章，无需 API 访问但需下载 dump 文件。支持断点续传: 下载阶段跳过已存在的文件 (>10MB)，解析阶段加载已有文章并跳过已处理标题，每处理完一个 dump 文件自动增量保存。

### 重建索引

```bash
# 完整重建 (9 阶段: 分块 → FAISS → BM25 → metadata → KG → entity FAISS)
python rebuild_indices.py

# 仅重建知识图谱 (跳过 FAISS/BM25/metadata，适用于 KG 提取中断后恢复)
python rebuild_indices.py --kg-only
```

对已有文章重新分块、编码、构建 FAISS + BM25 索引 + 知识图谱。切换 embedding 模型后需重建。

KG 提取使用 10 个并行 worker (ThreadPoolExecutor)，支持断点续传: 每处理 50 篇文章自动保存 checkpoint，中断后重新运行自动从上次位置恢复。

## 量化评估

```bash
python eval_quantitative.py
```

对 pipeline 生成的增强推理与原始 AgThoughts 推理进行逐样本量化对比。评估维度:

| 维度 | 说明 |
|------|------|
| specificity | 数字、单位、专有名词密度 |
| evidence | Extension/大学/USDA 等权威来源引用 |
| actionability | 具体操作动词密度 |
| coherence | 过渡词和逻辑连接词 |
| structure | 标题、编号列表、分节 |
| constraint_coverage | 问题约束条件的覆盖比例 |

输出逐样本对比表 + 聚合统计。

## 故障排除

**LLM 返回空响应**: deepseek-v4-pro 内部推理 token 会消耗 max_tokens 预算，确保 max_tokens >= 256。

**Reranker 400 错误**: Reranker API 最多接受 25 条文档，代码已自动限制候选数量。

**分类结果不稳定**: deepseek-v4 即使 temperature=0.0 也可能因内部推理产生不同输出，已通过正则词边界匹配 + 取最后匹配优化。

**Pipeline 卡住**: 检查网络连接，LLM API 超时设为 120 秒，自动重试 3 次。
