"""
test_helpers.py - shared helper for every test script.

Rule: tests must never touch the original CSV files in data/.
So before a test runs, we copy data/ into test_data/ and point the
DATA_DIR environment variable at that copy. config.py reads DATA_DIR,
so every tool then reads and writes the copy instead of the original.
"""

import hashlib
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEST_LOG_CSV = TEST_DATA_DIR / "logs" / "tool_calls.csv"


def use_fresh_test_data():
    """Copy data/ to test_data/ (replacing any old copy) and switch to it.

    Call this at the very top of a test script, BEFORE importing config
    or any tool, so config.py picks up the new DATA_DIR.
    """
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    shutil.copytree(ORIGINAL_DATA_DIR, TEST_DATA_DIR)
    os.environ["DATA_DIR"] = str(TEST_DATA_DIR)
    # Tool-call logs from tests go to test_data/logs/, never to the real logs/.
    # (A sub-folder, so the log file is not part of fingerprint(TEST_DATA_DIR).)
    os.environ["TOOL_LOG_CSV"] = str(TEST_LOG_CSV)
    # Same for every other generated folder (Phase 6).
    os.environ["VECTOR_STORE_DIR"] = str(TEST_DATA_DIR / "vector_store")
    os.environ["MEMORY_DIR"] = str(TEST_DATA_DIR / "memory_store")
    os.environ["CACHE_DIR"] = str(TEST_DATA_DIR / "cache")
    return TEST_DATA_DIR


# Counts how many checks passed, so a test script can print the total.
passed_count = {"passed": 0}


def check(description, condition, result=None):
    """Print PASS/FAIL for one check. Stop the script on the first failure."""
    if condition:
        passed_count["passed"] += 1
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if result is not None:
            print(f"        got: {result}")
        raise SystemExit(1)


def fingerprint(folder):
    """A hash of every CSV file in a folder. If any file changes even by one
    character, its hash changes, so comparing two fingerprints tells us
    whether anything was modified."""
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).glob("*.csv"))}
