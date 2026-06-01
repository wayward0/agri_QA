"""Pipeline v2 configuration — all constants in one place."""

# --- LLM API (Gitee AI OpenAI-compatible) ---
API_BASE_URL = "https://ai.gitee.com/v1"
API_KEY = "VEKQW2FENKYIPYBLFD0CXHC1TPAQPOSBGSWIGP1E"

# Model assignments
LLM_MODEL_GENERATE = "DeepSeek-R1"       # Reasoning chain generation
LLM_MODEL_TRANSLATE = "DeepSeek-V3-Flash"  # KG translation (lightweight)
LLM_MODEL_VERIFY = "DeepSeek-R1"         # Post-verification

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# --- FAISS ---
FAISS_TOP_K = 5

# --- Reasoning generation ---
TEMPERATURE_GENERATE = 0.2
TEMPERATURE_SAMPLE = 0.4
MAX_TOKENS = 8192
SELF_CONSISTENCY_N = 3

# --- Pipeline ---
API_CALL_INTERVAL = 1.0  # seconds between API calls
TEST_LIMIT = 3           # set to None for full run

# --- Paths ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

PATH_CROPDP_KG = os.path.join(BASE_DIR, "CropDP-KG.csv")
PATH_CROPDP_KG_EN = os.path.join(DATA_DIR, "CropDP-KG-EN.csv")
PATH_TERM_DICT = os.path.join(DATA_DIR, "term_dict.json")
PATH_AGTHOUGHTS = os.path.join(BASE_DIR, "AgThoughts.json")
PATH_AGRI_SUBSET = os.path.join(DATA_DIR, "agri_subset.json")
PATH_OUTPUT_XML = os.path.join(OUTPUT_DIR, "agri_qa_reasoning.xml")

# --- Domain filtering ---
BIOTIC_CATEGORIES = [
    "Biotic Diseases Questions",
    "Biotic Insects Questions",
    "Biotic Weeds Questions",
]
