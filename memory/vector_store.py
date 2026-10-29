"""
vector_store.py - LLM-written patient summaries stored in FAISS, plus retrieval.

The idea (this is the "R" in RAG for patients):
    1. For each patient, the LLM reads their records and writes ONE paragraph.
    2. Each paragraph is turned into a list of 384 numbers (an "embedding")
       by a small free model that runs on this computer
       (sentence-transformers/all-MiniLM-L6-v2). Texts with similar meaning
       get similar numbers.
    3. FAISS stores those number lists and can quickly find the paragraphs
       closest in meaning to a question, e.g.
           "70-year-old man with chronic kidney disease"  ->  Ramesh Kumar (P001)

Functions in this file (all logged tools):
    summarize_patient(patient_id, llm)        -> one summary (does not save it)
    build_patient_index(llm)                  -> summarize everyone, build + save FAISS
    refresh_patient_summary(patient_id, llm)  -> redo one summary after records change
    search_patient_summaries(query, k)        -> find patients by meaning
    get_patient_summary(patient_id)           -> read one saved summary

Saved files (in vector_store/, which is in .gitignore):
    patient_summaries.json   every summary as plain text (easy to read/show)
    faiss_index/             the FAISS index built from those summaries

Why summaries are short (at most 120 words) and start with the main
conditions: the embedding model only reads about the first 256 word-pieces
of a text, so anything after that would be ignored by the search.

Safety: the LLM is told to use ONLY the records. A patient with no records
gets a fixed sentence and the LLM is not called at all, so nothing can be
invented for them.
"""

import hashlib
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import config
from agent.prompts import PATIENT_SUMMARY_PROMPT
from tools.csv_helpers import read_csv
from tools.logger import log_tool_call
from tools.medical_records_tool import get_patient_history

SUMMARY_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_embeddings = None   # the embedding model, loaded once and then reused


def records_fingerprint(records):
    """A short fingerprint of a patient's records (a list of dicts).

    Saved with each summary. If any record is added OR EDITED later, the
    fingerprint changes, so we know the stored summary is out of date. (Just
    counting the records would miss an edit to an existing record.)
    """
    text = json.dumps(records, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers: file locations, embedding model, saving/loading
# ---------------------------------------------------------------------------

def _summaries_file():
    return config.VECTOR_STORE_DIR / "patient_summaries.json"


def _index_folder():
    return config.VECTOR_STORE_DIR / "faiss_index"


def get_embeddings():
    """Load the local embedding model once (takes a few seconds), then reuse it."""
    global _embeddings
    if _embeddings is None:
        # Unit-length vectors let us turn FAISS distances into cosine similarity.
        settings = {"model_name": config.EMBEDDING_MODEL,
                    "encode_kwargs": {"normalize_embeddings": True}}
        try:
            # Use the copy already downloaded on this computer (no internet).
            _embeddings = HuggingFaceEmbeddings(**settings, model_kwargs={"local_files_only": True})
        except Exception:
            # First run only: download the model (about 90 MB), then it is cached.
            _embeddings = HuggingFaceEmbeddings(**settings)
    return _embeddings


def _load_summaries():
    """Read all saved summaries as a dict {patient_id: {...}} (empty if none)."""
    try:
        return json.loads(_summaries_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _save_summaries_and_index(summaries):
    """Save the summaries JSON and rebuild the FAISS index from it.

    The index is built completely in memory FIRST; files are written only
    after that succeeds, so a failure never leaves a half-built store.
    """
    documents = [
        Document(page_content=s["summary"],
                 metadata={"patient_id": s["patient_id"], "name": s["name"]})
        for s in summaries.values()
    ]
    index = FAISS.from_documents(documents, get_embeddings())

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(_index_folder()))
    temp_path = _summaries_file().with_suffix(".tmp")
    temp_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    os.replace(temp_path, _summaries_file())


def _format_for_prompt(history):
    """Turn a get_patient_history() result into the text the prompt expects."""
    p = history["patient"]
    patient_text = (f"Name: {p['name']}; ID: {p['patient_id']}; Age: {p['age']}; "
                    f"Gender: {p['gender']}; Contact person: {p['contact_person']}")
    record_lines = [
        f"- {r['visit_date']} ({r['record_id']}): Diagnosis: {r['diagnosis']}; "
        f"Medications: {r['medications']}; Lab results: {r['lab_results']}; "
        f"Alerts: {r['alerts']}; Doctor notes: {r['doctor_notes']}"
        for r in history["records"]
    ]
    return {"patient": patient_text, "records": "\n".join(record_lines)}


# ---------------------------------------------------------------------------
# 1. Summarize one patient
# ---------------------------------------------------------------------------

@log_tool_call
def summarize_patient(patient_id, llm=None):
    """Ask the LLM for a one-paragraph summary of one patient's records.

    llm: leave as None to use Claude (agent.llm.get_llm); tests pass a fake.
    Returns {"status": "success", "summary": {...}} but does NOT save it.

    Status values: "success", "not_found", "error".
    """
    history = get_patient_history(patient_id)
    if history["status"] in ("not_found", "error"):
        return {"status": history["status"], "message": history["message"]}

    p = history["patient"]
    summary = {
        "patient_id": p["patient_id"],
        "name": p["name"],
        "age": p["age"],
        "gender": p["gender"],
        "record_count": len(history["records"]),
        "records_hash": records_fingerprint(history["records"]),
        "latest_visit": history["records"][-1]["visit_date"] if history["records"] else "",
        "generated_at": config.get_now().strftime(SUMMARY_TIME_FORMAT),
    }

    if history["status"] == "no_records":
        # No records -> fixed sentence, no LLM call, nothing can be invented.
        summary["summary"] = (f"{p['name']} ({p['patient_id']}), age {p['age']}, "
                              f"{p['gender'].lower()}. No medical records on file yet.")
        summary["generated_by"] = "template"
    else:
        try:
            if llm is None:
                from agent.llm import get_llm   # imported here so tests never need a key
                llm = get_llm()
            chain = PATIENT_SUMMARY_PROMPT | llm          # prompt -> LLM, a LangChain "chain"
            reply = chain.invoke(_format_for_prompt(history))
            text = " ".join(reply.text.split())         # .text skips any non-text parts
        except Exception as error:
            return {"status": "error",
                    "message": f"LLM could not summarize {p['patient_id']}: {error}"}
        if not text:
            return {"status": "error", "message": f"LLM returned an empty summary for {p['patient_id']}."}
        summary["summary"] = text
        summary["generated_by"] = "llm"

    return {"status": "success",
            "message": f"Summary written for {p['name']} ({p['patient_id']}).",
            "summary": summary}


# ---------------------------------------------------------------------------
# 2. Build the whole store
# ---------------------------------------------------------------------------

@log_tool_call
def build_patient_index(llm=None):
    """Summarize every patient, then build and save the FAISS index.

    If ANY summary fails, nothing is saved and the old store stays as it was.

    Status values: "success", "error".
    """
    try:
        patients = read_csv(config.PATIENTS_CSV)
        summaries = {}
        for patient_id in patients["patient_id"]:
            result = summarize_patient(patient_id, llm=llm)
            if result["status"] != "success":
                return {"status": "error",
                        "message": f"Stopped at {patient_id}: {result['message']} "
                                   "The existing store was not changed."}
            summaries[patient_id] = result["summary"]

        _save_summaries_and_index(summaries)
        return {"status": "success",
                "message": f"Built the patient store with {len(summaries)} summaries.",
                "count": len(summaries)}
    except Exception as error:
        return {"status": "error", "message": f"Could not build the patient store: {error}"}


# ---------------------------------------------------------------------------
# 3. Refresh one patient after their records change
# ---------------------------------------------------------------------------

@log_tool_call
def refresh_patient_summary(patient_id, llm=None):
    """Re-summarize one patient and rebuild the index (no other LLM calls).

    Use this after add_record / update_record / add_patient, so the store
    never describes out-of-date records.

    Status values: "success", "not_found", "no_index", "error".
    """
    try:
        summaries = _load_summaries()
        if not summaries:
            return {"status": "no_index",
                    "message": "The patient store has not been built yet. Run build_patient_index first."}
        result = summarize_patient(patient_id, llm=llm)
        if result["status"] != "success":
            return {"status": result["status"], "message": result["message"]}
        new_summary = result["summary"]
        summaries[new_summary["patient_id"]] = new_summary
        _save_summaries_and_index(summaries)
        return {"status": "success",
                "message": f"Summary refreshed for {new_summary['name']} ({new_summary['patient_id']}).",
                "summary": new_summary}
    except Exception as error:
        return {"status": "error", "message": f"Could not refresh summary: {error}"}


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------

@log_tool_call
def search_patient_summaries(query, k=3):
    """Find the k patients whose summaries are closest in meaning to the query.

    Example: search_patient_summaries("70-year-old with chronic kidney disease")
    Each match has patient_id, name, age, summary and similarity.

    similarity is the "cosine similarity" of the query and the summary:
    1 = same meaning, 0 = unrelated, below 0 = opposite. FAISS gives us the
    squared distance d between the two (unit-length) vectors, and for
    unit-length vectors cosine similarity = 1 - d / 2.

    Status values: "success", "no_index", "invalid", "error".
    """
    query = " ".join(str(query or "").split())
    if not query:
        return {"status": "invalid", "message": "Please give something to search for.", "matches": []}
    if not (_index_folder() / "index.faiss").exists():
        return {"status": "no_index",
                "message": "The patient store has not been built yet. Run build_patient_index first.",
                "matches": []}
    try:
        # allow_dangerous_deserialization: FAISS saves part of the index with
        # Python's "pickle" format, which can run code when loaded. That is
        # only a risk for files from strangers; this index is one we built.
        index = FAISS.load_local(str(_index_folder()), get_embeddings(),
                                 allow_dangerous_deserialization=True)
        summaries = _load_summaries()
        hits = index.similarity_search_with_score(query, k=int(k))   # closest first
        matches = []
        for document, squared_distance in hits:
            patient_id = document.metadata["patient_id"]
            matches.append({
                "patient_id": patient_id,
                "name": document.metadata["name"],
                "age": summaries.get(patient_id, {}).get("age", ""),
                "summary": document.page_content,
                "similarity": round(1 - float(squared_distance) / 2, 3),
            })
        names = ", ".join(f"{m['name']} ({m['patient_id']}, {m['similarity']})" for m in matches)
        return {"status": "success",
                "message": f"Closest patient summaries: {names}.",
                "matches": matches}
    except Exception as error:
        return {"status": "error", "message": f"Could not search the patient store: {error}", "matches": []}


@log_tool_call
def get_patient_summary(patient_id):
    """Return the saved summary for one patient (no LLM call).

    Status values: "success", "not_found", "no_index".
    """
    summaries = _load_summaries()
    if not summaries:
        return {"status": "no_index",
                "message": "The patient store has not been built yet. Run build_patient_index first."}
    patient_id = str(patient_id).strip().upper()
    if patient_id not in summaries:
        return {"status": "not_found", "message": f"No saved summary for {patient_id}."}
    summary = summaries[patient_id]
    return {"status": "success",
            "message": f"Summary for {summary['name']} ({patient_id}), written {summary['generated_at']}.",
            "summary": summary}


# ---------------------------------------------------------------------------
# Run this file directly to (re)build the real store with Claude:
#     python -m memory.vector_store
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Summarizing every patient with the LLM and building FAISS...")
    result = build_patient_index()
    print(result["message"])
    if result["status"] == "success":
        for s in _load_summaries().values():
            print(f"\n[{s['patient_id']}] ({s['generated_by']}) {s['summary']}")
        print("\nTest search: '70-year-old father with chronic kidney disease'")
        print(search_patient_summaries("70-year-old father with chronic kidney disease")["message"])
