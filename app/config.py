import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
ORDERS_FILE = BASE_DIR / "data" / "orders.json"
EVALUATION_FILE = BASE_DIR / "evaluation" / "visible-cases.json"


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_PROVIDER = "gemini"

# Keep the model name you were already using.
LLM_MODEL = "gemini-3.6-flash"


# =========================================================
# RAG
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks returned by RAG
TOP_K = 5

# Minimum semantic similarity required BEFORE
# authority/keyword bonuses are applied.
MIN_RELEVANCE_SCORE = 0.30


# =========================================================
# STARTUP WARNING
# =========================================================

if not GEMINI_API_KEY:
    print(
        "Warning: GEMINI_API_KEY is not configured."
    )