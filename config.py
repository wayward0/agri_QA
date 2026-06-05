"""Project configuration constants.

This file imports nothing from the project. Only os and pathlib.
No module should import config directly — pipeline.py reads config and passes values.
"""

import os
from pathlib import Path

# --- API ---
API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://ai.centos.hk/v1")
API_KEY = os.environ.get("AI_CENTOS_API_KEY", "")
LLM_MAX_TOKENS = 4096

# --- Model Tiers (cost vs quality balance) ---
# Tier 1: Cheap & fast — classification, simple extraction, revision execution
MODEL_LIGHT = os.environ.get("MODEL_LIGHT", "deepseek-v4-flash")
# Tier 2: Standard — review, evaluation, integration
MODEL_STANDARD = os.environ.get("MODEL_STANDARD", "deepseek-v4-pro")
# Tier 3: Premium — core reasoning generation (Thinker)
MODEL_PREMIUM = os.environ.get("MODEL_PREMIUM", "deepseek-v4-pro")

# --- Embedding ---
EMBEDDING_API_BASE_URL = os.environ.get("EMBEDDING_API_BASE_URL", "https://api.moark.com/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = 1024

# --- Reranker ---
RERANKER_API_URL = os.environ.get("RERANKER_API_URL", "https://api.moark.com/v1/rerank")
RERANKER_API_KEY = os.environ.get("RERANKER_API_KEY", "")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "bge-reranker-v2-m3")

# --- RAG ---
RAG_TOP_K_BACKGROUND = 5
RAG_TOP_K_FACT_CHECK = 3
RAG_TOP_K_GAP_FILL = 3
RRF_K = 60
CHUNK_MIN_TOKENS = 200
CHUNK_MAX_TOKENS = 400

# --- Thinker ---
REACT_MAX_ROUNDS = 5
SELF_CONSISTENCY_N = 3
REACT_TEMPERATURES = [0.3, 0.7, 1.0]

# --- Evaluator ---
QUALITY_GATE_THRESHOLD = 3.0
FAITHFULNESS_GATE_THRESHOLD = 3.0  # MEDIUM: force retry if faithfulness drops below this
MAX_REVISION_ITERATIONS = 1
EVAL_WEIGHTS = {
    "faithfulness": 0.25,
    "structure": 0.20,
    "information_density": 0.15,
    "logical_completeness": 0.25,
    "traceability": 0.15,
}

# --- Pipeline ---
API_CALL_INTERVAL = 0.5  # seconds between items (legacy, unused in async mode)
CONSECUTIVE_FAILURE_LIMIT = 3
MAX_CONCURRENCY = 10  # max concurrent items in async pipeline

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PATH_AGTHOUGHTS = BASE_DIR / "AgThoughts.json"
PATH_SAMPLE = DATA_DIR / "agthoughts" / "sample_1000.json"
PATH_CHUNKS = DATA_DIR / "chunks" / "passages.jsonl"
PATH_FAISS_INDEX = DATA_DIR / "index" / "faiss.index"
PATH_BM25_INDEX = DATA_DIR / "index" / "bm25.pkl"
PATH_INDEX_METADATA = DATA_DIR / "index" / "metadata.json"
PATH_OUTPUT = OUTPUT_DIR / "enhanced_dataset.json"

# --- Knowledge Graph ---
KG_TOP_K_ENTITIES = 5
KG_MAX_GRAPH_PASSAGES = 10
PATH_KG_ENTITIES = DATA_DIR / "index" / "kg_entities.json"
PATH_KG_RELATIONS = DATA_DIR / "index" / "kg_relations.json"
PATH_KG_ENTITY_FAISS = DATA_DIR / "index" / "entity_faiss.index"
