"""
live_scenario.py - run the sample scenario end to end with the REAL LLM and
the REAL MedlinePlus service, on a COPY of the data (data/ is never changed).

Needs internet and the API key. About 3 LLM calls (plan, booking message,
disease summary), plus 11 more if the patient store has not been built yet.
Run from the project root:
    python -m tests.live_scenario
It answers "yes" to each confirmation automatically and prints every turn.
"""

import shutil

from tests.test_helpers import use_fresh_test_data, PROJECT_ROOT, TEST_DATA_DIR

use_fresh_test_data()

from agent.agent import HealthcareAgent  # noqa: E402
from memory.vector_store import build_patient_index  # noqa: E402
from tools.logger import read_logs  # noqa: E402

# Reuse the real patient store if it exists (it was built from the same data).
real_store = PROJECT_ROOT / "vector_store"
if real_store.exists():
    shutil.copytree(real_store, TEST_DATA_DIR / "vector_store")
else:
    print("Building the patient store first (11 LLM calls)...")
    print(build_patient_index()["message"])

SCENARIO = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist "
            "for him. Also, can you summarize latest treatment methods?")

agent = HealthcareAgent()
turns = [SCENARIO, "yes", "yes"]
for number, text in enumerate(turns, start=1):
    print("=" * 78)
    print(f"TURN {number} - USER: {text}")
    print("-" * 78)
    result = agent.handle(text)
    print(result["reply"])
    if result["waiting_for"] is None and number < len(turns):
        print("\n(The agent is not waiting for an answer; stopping here.)")
        break

print("=" * 78)
print("TRACE (one line per step):")
for t in result["trace"]:
    print(f"  {t['step']}. {t['tool']:<22} {t['status']:<20} {t['message'][:90]}")
print("\nLOG ROWS (test_data/logs/tool_calls.csv):")
for row in read_logs():
    print(f"  {row['tool_name']:<26} {row['status']:<20} {row['duration_ms']:>9} ms")
