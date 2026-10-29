"""
test_memory.py - tests for memory/conversation_memory.py

Run from the project root:
    python -m tests.test_memory
"""

import json
import os

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from memory.conversation_memory import (  # noqa: E402
    ShortTermMemory, save_patient_memory, recall_patient_memory,
)


def test_short_term():
    print("short-term memory")
    memory = ShortTermMemory(max_messages=4)
    memory.add_user_message("My father has kidney disease.")
    memory.add_ai_message("Is your father Ramesh Kumar (P001)?")
    check("messages stored in order",
          [m.type for m in memory.get_messages()] == ["human", "ai"])
    check("as_text shows who said what",
          memory.as_text() == "User: My father has kidney disease.\n"
                              "Assistant: Is your father Ramesh Kumar (P001)?", memory.as_text())

    memory.set_context("patient_id", "P001")
    check("context remembers the current patient", memory.get_context("patient_id") == "P001")
    check("missing context -> default", memory.get_context("doctor_id", "none") == "none")

    for i in range(5):
        memory.add_user_message(f"message {i}")
    texts = [m.text for m in memory.get_messages()]
    check("only the last 4 messages are kept",
          texts == ["message 1", "message 2", "message 3", "message 4"], texts)

    memory.clear()
    check("clear() empties messages and context",
          memory.get_messages() == [] and memory.context == {})
    check("default size comes from config",
          ShortTermMemory().max_messages == config.SHORT_TERM_MEMORY_MESSAGES)
    check("two conversations do not share memory",
          ShortTermMemory().get_messages() == [] and ShortTermMemory().context == {})


def test_long_term():
    print("long-term memory")
    check("no notes yet -> no_memory", recall_patient_memory("P001")["status"] == "no_memory")

    r = save_patient_memory("p001", "Booked A011 with Dr. Meera Shah on 2026-10-21 14:00.", "booking")
    check("save -> success", r["status"] == "success", r)
    check("  note has timestamp, type and text",
          r["note"] == {"timestamp": "2026-10-21 12:00:00", "type": "booking",
                        "text": "Booked A011 with Dr. Meera Shah on 2026-10-21 14:00."}, r)
    save_patient_memory("P001", "Looked up MedlinePlus: Chronic Kidney Disease.", "search")
    save_patient_memory("P001", "Family asked about dialysis.")

    path = TEST_DATA_DIR / "memory_store" / "P001.json"
    check("saved as test_data/memory_store/P001.json", path.exists())
    check("  file is readable JSON with 3 notes",
          len(json.loads(path.read_text())["notes"]) == 3)

    r = recall_patient_memory("P001")
    check("recall -> all 3 notes, oldest first",
          [n["type"] for n in r["notes"]] == ["booking", "search", "note"], r)
    check("limit=1 -> only the newest note",
          [n["text"] for n in recall_patient_memory("P001", limit=1)["notes"]]
          == ["Family asked about dialysis."])
    check("filter by type 'booking'",
          [n["type"] for n in recall_patient_memory("P001", note_type="booking")["notes"]] == ["booking"])

    check("other patients are separate", recall_patient_memory("P011")["status"] == "no_memory")
    check("unknown patient -> not_found (save)",
          save_patient_memory("P999", "hello")["status"] == "not_found")
    check("unknown patient -> not_found (recall)", recall_patient_memory("P999")["status"] == "not_found")
    check("empty note -> invalid", save_patient_memory("P001", "   ")["status"] == "invalid")

    # "Long-term" means it survives a restart: read straight from the file again.
    check("notes survive (re-read from disk)", len(recall_patient_memory("P001", limit=None)["notes"]) == 3)


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    data_before = fingerprint(TEST_DATA_DIR)

    test_short_term()
    test_long_term()

    print("safety")
    check("memory notes never change the CSV files", fingerprint(TEST_DATA_DIR) == data_before)
    check("tests used test_data/memory_store",
          config.MEMORY_DIR == TEST_DATA_DIR / "memory_store")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
