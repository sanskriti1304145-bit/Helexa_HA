"""
test_vector_store.py - tests for memory/vector_store.py

Run from the project root:
    python -m tests.test_vector_store

Offline and free: instead of Claude we use LangChain's FakeListChatModel,
which simply returns the summaries listed below, one per call, in order.
The embeddings are the REAL local model, so the search checks test real
meaning-based retrieval. (The model must have been downloaded once.)
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()
import os  # noqa: E402
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from memory.vector_store import (  # noqa: E402
    summarize_patient, build_patient_index, refresh_patient_summary,
    search_patient_summaries, get_patient_summary, _format_for_prompt,
)
from tools.logger import read_logs  # noqa: E402
from tools.medical_records_tool import add_record, get_patient_history  # noqa: E402

from tests.fake_data import FAKE_SUMMARIES  # noqa: E402


def fake_llm(responses):
    return FakeListChatModel(responses=list(responses))


def broken_llm():
    """A stand-in LLM that always fails, like a network or API error."""
    def fail(_):
        raise ConnectionError("simulated API failure")
    return RunnableLambda(fail)


def top_ids(result, n=1):
    return [m["patient_id"] for m in result["matches"][:n]]


# ---------------------------------------------------------------------------

def test_before_index_exists():
    print("before the store is built")
    check("search -> no_index", search_patient_summaries("kidney")["status"] == "no_index")
    check("get summary -> no_index", get_patient_summary("P001")["status"] == "no_index")
    check("refresh -> no_index", refresh_patient_summary("P001", llm=fake_llm(["x"]))["status"] == "no_index")


def test_summarize_patient():
    print("summarize_patient")
    inputs = _format_for_prompt(get_patient_history("P001"))
    check("prompt gets patient name, age and every record",
          "Ramesh Kumar" in inputs["patient"] and "Age: 70" in inputs["patient"]
          and inputs["records"].count("\n- ") == 3 and "eGFR 31" in inputs["records"], inputs)

    r = summarize_patient("P001", llm=fake_llm([FAKE_SUMMARIES[0]]))
    check("P001 -> success, text from the LLM", r["status"] == "success"
          and r["summary"]["summary"] == FAKE_SUMMARIES[0] and r["summary"]["generated_by"] == "llm", r)
    check("  metadata: 4 records, latest visit 2026-04-20",
          r["summary"]["record_count"] == 4 and r["summary"]["latest_visit"] == "2026-04-20", r["summary"])

    r = summarize_patient("P012", llm=broken_llm())   # would fail if the LLM were called
    check("P012 (no records) -> fixed sentence, LLM not called",
          r["status"] == "success" and r["summary"]["generated_by"] == "template"
          and "No medical records" in r["summary"]["summary"], r)

    check("unknown patient -> not_found", summarize_patient("P999", llm=fake_llm(["x"]))["status"] == "not_found")
    r = summarize_patient("P001", llm=broken_llm())
    check("LLM failure -> status error (no crash)", r["status"] == "error" and "simulated" in r["message"], r)
    r = summarize_patient("P001", llm=fake_llm(["   "]))
    check("empty LLM reply -> status error", r["status"] == "error", r)


def test_build_index():
    print("build_patient_index")
    count = len(read_logs())
    r = build_patient_index(llm=fake_llm(FAKE_SUMMARIES))
    check("build -> success with 12 summaries", r["status"] == "success" and r["count"] == 12, r)
    check("  files saved in test_data/vector_store",
          (TEST_DATA_DIR / "vector_store" / "patient_summaries.json").exists()
          and (TEST_DATA_DIR / "vector_store" / "faiss_index" / "index.faiss").exists())
    rows = read_logs()[count:]
    check("  one log row for the whole build (inner calls not logged)",
          [x["tool_name"] for x in rows] == ["build_patient_index"], [x["tool_name"] for x in rows])

    before = fingerprint(TEST_DATA_DIR / "vector_store")
    store_files = sorted(p.name for p in (TEST_DATA_DIR / "vector_store").rglob("*"))
    r = build_patient_index(llm=broken_llm())
    check("build with failing LLM -> error, says store not changed",
          r["status"] == "error" and "not changed" in r["message"], r)
    check("  old store kept exactly",
          fingerprint(TEST_DATA_DIR / "vector_store") == before
          and sorted(p.name for p in (TEST_DATA_DIR / "vector_store").rglob("*")) == store_files)


def test_retrieval():
    print("search_patient_summaries (real local embeddings)")
    r = search_patient_summaries("70-year-old father with chronic kidney disease")
    check("scenario query -> Ramesh Kumar (P001) first", top_ids(r) == ["P001"], r["message"])
    sims = [m["similarity"] for m in r["matches"]]
    check("  results have summary and similarity, highest first",
          all(-1 <= s <= 1 for s in sims) and sims == sorted(sims, reverse=True)
          and all(m["summary"] for m in r["matches"]), sims)
    check("  the right patient is clearly ahead (gap > 0.1)", sims[0] - sims[1] > 0.1, sims)
    check("  3 results by default", len(r["matches"]) == 3)

    check("'panic attacks' -> Michael Johnson (P007)",
          top_ids(search_patient_summaries("panic attacks")) == ["P007"])
    check("'trouble sleeping' -> an insomnia patient (P003 or P010)",
          top_ids(search_patient_summaries("trouble sleeping at night"))[0] in ("P003", "P010"))
    check("'irregular heartbeat on blood thinners' -> Arvind Iyer (P009)",
          top_ids(search_patient_summaries("irregular heartbeat on blood thinners")) == ["P009"])
    check("k=5 returns 5", len(search_patient_summaries("diabetes", k=5)["matches"]) == 5)
    check("empty query -> invalid", search_patient_summaries("  ")["status"] == "invalid")

    r = get_patient_summary("p001")
    check("get_patient_summary('p001') -> saved summary",
          r["status"] == "success" and r["summary"]["summary"] == FAKE_SUMMARIES[0], r)
    check("get_patient_summary unknown -> not_found", get_patient_summary("P999")["status"] == "not_found")


def test_refresh():
    print("refresh_patient_summary")
    add_record("P012", "2026-10-15", "Seasonal allergic rhinitis", "Cetirizine 10mg",
               alerts="No known allergies", confirm=True)
    new_text = ("Daniel Brooks (P012), 25-year-old male, was seen in October 2026 for seasonal "
                "allergic rhinitis and started Cetirizine 10mg. No known allergies.")
    r = refresh_patient_summary("P012", llm=fake_llm([new_text]))
    check("refresh after new record -> success, now written by LLM",
          r["status"] == "success" and r["summary"]["generated_by"] == "llm", r)
    check("  saved summary updated", get_patient_summary("P012")["summary"]["summary"] == new_text)
    check("  other patients unchanged",
          get_patient_summary("P001")["summary"]["summary"] == FAKE_SUMMARIES[0])
    check("  search finds the new information",
          top_ids(search_patient_summaries("hay fever allergy sneezing")) == ["P012"])
    check("refresh unknown patient -> not_found",
          refresh_patient_summary("P999", llm=fake_llm(["x"]))["status"] == "not_found")


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    real_store = config.PROJECT_ROOT / "vector_store"
    real_store_existed = real_store.exists()

    test_before_index_exists()
    test_summarize_patient()
    test_build_index()
    test_retrieval()
    test_refresh()

    print("safety")
    check("tests used test_data/vector_store, not the real vector_store/",
          config.VECTOR_STORE_DIR == TEST_DATA_DIR / "vector_store"
          and real_store.exists() == real_store_existed)
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed (no LLM calls, no internet).")
