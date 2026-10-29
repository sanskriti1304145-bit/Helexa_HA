"""
test_setup.py - checks that Phase 1 setup works.

Run from the project root:
    python -m tests.test_setup

It checks that:
  1. every required package can be imported
  2. tests use a copy of the data, not the originals
  3. every data file path in config.py points to a real file
  4. the API key is loaded from .env (the key itself is never printed)
"""

from tests.test_helpers import use_fresh_test_data, TEST_DATA_DIR

# Switch to the test copy BEFORE importing config.
use_fresh_test_data()

import importlib  # noqa: E402

import config  # noqa: E402


def check_packages():
    # Module names as you would write them in an "import" line.
    packages = [
        "pandas", "dotenv", "requests", "anthropic",
        "langchain", "langchain_core", "langchain_anthropic",
        "langchain_community", "langchain_classic", "langchain_huggingface",
        "faiss", "sentence_transformers", "streamlit",
    ]
    for name in packages:
        importlib.import_module(name)
        print(f"  OK   import {name}")


def check_uses_test_copy():
    assert config.DATA_DIR == TEST_DATA_DIR, "config is NOT using the test copy!"
    print(f"  OK   config.DATA_DIR -> {config.DATA_DIR}")


def check_data_files():
    files = [
        config.PATIENTS_CSV, config.MEDICAL_RECORDS_CSV, config.DOCTORS_CSV,
        config.SLOTS_CSV, config.APPOINTMENTS_CSV,
    ]
    for path in files:
        assert path.exists(), f"Missing data file: {path}"
        print(f"  OK   {path.name} found")


def check_api_key():
    key = config.ANTHROPIC_API_KEY
    assert key, "ANTHROPIC_API_KEY not found. Check your .env file."
    print(f"  OK   ANTHROPIC_API_KEY loaded ({len(key)} characters, not shown)")


if __name__ == "__main__":
    print("1. Packages")
    check_packages()
    print("2. Test data copy")
    check_uses_test_copy()
    print("3. Data files")
    check_data_files()
    print("4. API key")
    check_api_key()
    print("\nAll Phase 1 checks passed.")
