"""
config.py - one place for every setting the project uses.

Every other file imports from here instead of hard-coding paths or names.
That way, if a file moves or a setting changes, we only edit this file.
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Project folders
# ---------------------------------------------------------------------------

# The folder that contains this config.py file (the project root).
PROJECT_ROOT = Path(__file__).resolve().parent

# Read the .env file in the project root so ANTHROPIC_API_KEY becomes an
# environment variable. The key itself is never written in the code.
load_dotenv(PROJECT_ROOT / ".env")

# Where the CSV files live. Tests set the DATA_DIR environment variable to
# point at a COPY of the data, so the original CSV files are never changed.
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

LOGS_DIR = PROJECT_ROOT / "logs"                  # tool-call log CSV (Phase 4)
TEST_DATA_DIR = PROJECT_ROOT / "test_data"        # CSV copies used by tests

# Generated folders (all in .gitignore). Tests point these at test_data/.
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", PROJECT_ROOT / "vector_store"))  # FAISS index
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", PROJECT_ROOT / "memory_store"))  # long-term memory
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))          # MedlinePlus cache

# ---------------------------------------------------------------------------
# 2. Data file paths (the names match the files already in data/)
# ---------------------------------------------------------------------------

PATIENTS_CSV = DATA_DIR / "patients.csv"
MEDICAL_RECORDS_CSV = DATA_DIR / "medical_records.csv"
DOCTORS_CSV = DATA_DIR / "doctors.csv"
SLOTS_CSV = DATA_DIR / "slots.csv"
APPOINTMENTS_CSV = DATA_DIR / "appointment.csv"
# Name matching (find_patient / identify_patient). First name and surname are
# compared SEPARATELY (difflib SequenceMatcher ratio on lower-case text, 0 to 1).
# A name is shown as "Similar name" only if BOTH minimums are reached. Tune here.
SURNAME_SIMILARITY_MIN = 0.85
FIRST_NAME_SIMILARITY_MIN = 0.78

SCHEDULE_TEMPLATE_CSV = DATA_DIR / "schedule_template.csv"   # weekly slot pattern per doctor
WAITLIST_CSV = DATA_DIR / "waitlist.csv"                     # patients waiting for an opening

# Tests set TOOL_LOG_CSV so their log rows go to test_data/, not logs/.
TOOL_LOG_CSV = Path(os.getenv("TOOL_LOG_CSV", LOGS_DIR / "tool_calls.csv"))

# ---------------------------------------------------------------------------
# 3. "Now" (current date and time)
# ---------------------------------------------------------------------------

# Format used for the APP_NOW override, e.g. "2026-10-21 12:00".
NOW_FORMAT = "%Y-%m-%d %H:%M"


def get_now():
    """Return the current date and time.

    Normally this is the real clock. Tests can pretend it is a fixed moment
    by setting the APP_NOW environment variable (format: "YYYY-MM-DD HH:MM").
    Every tool asks this function for "now" instead of calling
    datetime.now() directly, so one override changes the whole project.
    """
    override = os.getenv("APP_NOW")
    if override:
        return datetime.strptime(override, NOW_FORMAT)
    return datetime.now()


# ---------------------------------------------------------------------------
# 4. LLM settings
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # loaded from .env above

# Change this one line to switch models (for example "claude-opus-5-5").
LLM_MODEL = "claude-sonnet-5-5"
LLM_MAX_TOKENS = 2048   # longest reply we allow from the LLM

# The model that grades answers in the evaluation (Phase 8). Using a different
# (e.g. stronger) model than the agent reduces "grading your own homework" bias.
# Change it here, or for one run:  python -m evaluation.evaluate --judge-model claude-opus-5-5
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

# ---------------------------------------------------------------------------
# 5. Embedding and search settings
# ---------------------------------------------------------------------------

# A small, free embedding model that runs on your own computer.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MedlinePlus web service (free, no API key needed).
# Docs: https://medlineplus.gov/about/developers/webservices/
MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
MEDLINEPLUS_DB = "healthTopics"                     # English health topics
MEDLINEPLUS_TOOL_NAME = "agentic_healthcare_capstone"  # identifies our app to NLM (no spaces)
REQUEST_TIMEOUT_SECONDS = 10
# NLM asks apps to cache results for 12-24 hours. We keep them for 24.
SEARCH_CACHE_HOURS = 24

# How many recent messages the short-term conversation memory keeps.
SHORT_TERM_MEMORY_MESSAGES = 10

# ---------------------------------------------------------------------------
# 6. UI text
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Demo only, not medical advice. All patient data is synthetic. "
    "Always consult a qualified doctor."
)
