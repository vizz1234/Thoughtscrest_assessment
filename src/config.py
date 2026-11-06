"""Configuration settings for the Agentic Personal Finance Advisor."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env (and fall back to .env.example if .env is not created yet)
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.example")

DATA_DIR = BASE_DIR / "data_store"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRACE_LOG_DIR = BASE_DIR / "logs"
TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "bank_advisor.db"
MEMORY_PATH = DATA_DIR / "long_term_memory.json"

# Gemini LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
LLM_MODE = os.getenv("LLM_MODE", "auto").strip().lower()  # 'auto', 'gemini', or 'fallback'

# Reference current date for consistent mock data analysis
REFERENCE_DATE = "2026-11-06"
CURRENT_MONTH = "2026-10"     # Full recent month for complete MoM analysis
PREVIOUS_MONTH = "2026-09"
TWO_MONTHS_AGO = "2026-08"

# Default Alert Thresholds
BUDGET_WARNING_THRESHOLD = 0.85    # 85% of monthly category budget
BUDGET_CRITICAL_THRESHOLD = 1.00   # 100% of monthly category budget
ANOMALY_Z_SCORE_THRESHOLD = 2.2    # Z-score threshold for unusual transaction spikes

