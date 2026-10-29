"""
disease_rag.py - search MedlinePlus and summarize the results (a RAG pipeline).

RAG = Retrieval-Augmented Generation. Instead of letting the LLM answer from
its own memory (which could be wrong or invented), we:
    1. RETRIEVE: search MedlinePlus (search_medlineplus, with 24-hour cache)
    2. SPLIT:    cut each result's text into small passages (chunks)
    3. RANK:     embed the chunks in a temporary FAISS index and keep the ones
                 closest in meaning to the question (e.g. "treatment options")
    4. GENERATE: the LLM summarizes ONLY those passages, citing [1], [2]...
    5. CHECK:    our code verifies the citations and adds the source links
                 itself, so the links are always correct and always shown.

If MedlinePlus finds nothing, is down, or the LLM's answer has no valid
citations, we never fall back to the LLM's own knowledge.
"""

import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.prompts import DISEASE_SUMMARY_PROMPT
from memory.conversation_memory import save_patient_memory
from memory.vector_store import get_embeddings, get_patient_summary
from tools.disease_search_tool import search_medlineplus
from tools.logger import log_tool_call

# Words that describe WHAT the user wants to know, not WHICH disease. MedlinePlus
# is a health-topic search, so they make results worse ("chronic kidney disease
# latest treatments" finds Heart Diseases). We remove them from the search term;
# the full question is still used to rank the passages.
FILLER_WORDS = {"latest", "newest", "new", "recent", "current", "modern", "treatment", "treatments",
                "therapy", "therapies", "options", "option", "methods", "method", "management",
                "information", "info", "about", "summary", "summarize", "for", "of", "on", "the", "and",
                "what", "are", "is", "how", "to", "treat", "treated", "cure", "cures"}

CHUNK_SIZE = 600        # characters per passage
CHUNK_OVERLAP = 80      # a little overlap so sentences are not cut in half
PASSAGES_TO_USE = 5     # how many passages the LLM gets


def clean_search_topic(topic):
    """'chronic kidney disease latest treatments' -> 'chronic kidney disease'.
    If nothing would be left, the original topic is kept."""
    words = [w.strip("?,.!") for w in str(topic).split()]
    kept = [w for w in words if w and w.lower() not in FILLER_WORDS]
    return " ".join(kept) if kept else " ".join(w for w in words if w)


def _split_into_passages(results):
    """Turn search results into small Documents, each remembering its source number."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    passages = []
    for number, result in enumerate(results, start=1):
        for chunk in splitter.split_text(result["text"]):
            passages.append(Document(page_content=chunk, metadata={
                "source_number": number, "title": result["title"], "link": result["link"]}))
    return passages


def _sources_block(results, numbers):
    lines = [f"[{n}] {results[n - 1]['title']} - {results[n - 1]['link']}" for n in sorted(numbers)]
    return "Sources (MedlinePlus, U.S. National Library of Medicine):\n" + "\n".join(lines)


@log_tool_call
def summarize_disease_info(topic, question, patient_id=None, llm=None):
    """Answer a question about a disease using only MedlinePlus text.

    topic:      what to search for, e.g. "chronic kidney disease"
    question:   what the user wants to know, e.g. "latest treatment methods"
    patient_id: optional. If given, the patient's stored summary is looked up
                (memory) and added to the prompt as context, and a note about
                this search is saved to the patient's long-term memory.

    Status values: "success", "no_results", "timeout", "no_connection",
                   "invalid", "error".
    Returns "answer" (text with citations + sources), "sources", "passages".
    """
    # --- 1. Retrieve (search for the disease name only) ---
    topic = clean_search_topic(topic)
    search = search_medlineplus(topic, max_results=3)
    if search["status"] != "success":
        return {"status": search["status"],
                "message": f"{search['message']} I will not summarize treatment information "
                           "without a trusted source.",
                "answer": "", "sources": [], "passages": []}
    results = search["results"]

    # --- 2 + 3. Split and rank ---
    passages = _split_into_passages(results)
    index = FAISS.from_documents(passages, get_embeddings())
    best = index.similarity_search(f"{topic}: {question}", k=min(PASSAGES_TO_USE, len(passages)))
    best.sort(key=lambda d: d.metadata["source_number"])   # group passages by source
    passages_text = "\n\n".join(f"[{d.metadata['source_number']}] ({d.metadata['title']}) {d.page_content}"
                                for d in best)
    allowed_numbers = {d.metadata["source_number"] for d in best}
    used_passages = [{"source_number": d.metadata["source_number"], "title": d.metadata["title"],
                      "link": d.metadata["link"], "text": d.page_content} for d in best]

    # --- 4. Generate (with the patient's stored summary as context) ---
    patient_context = ""
    if patient_id:
        stored = get_patient_summary(patient_id)
        patient_context = stored["summary"]["summary"] if stored["status"] == "success" else ""
    try:
        if llm is None:
            from agent.llm import get_llm
            llm = get_llm()
        reply = (DISEASE_SUMMARY_PROMPT | llm).invoke({
            "question": question,
            "patient_context": patient_context or "(none)",
            "passages": passages_text,
        })
        answer = reply.text.strip()
    except Exception as error:
        return {"status": "error", "message": f"The LLM could not summarize the sources: {error}",
                "answer": "", "sources": [], "passages": used_passages}

    # --- 5. Check citations and add the source links ourselves ---
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if not answer or not cited or not cited <= allowed_numbers:
        # The LLM did not cite properly: show the source passages word for word instead.
        quoted = "\n\n".join(f"[{p['source_number']}] {p['text']}" for p in used_passages)
        answer = ("I could not produce a properly cited summary, so here are the most relevant "
                  "passages from MedlinePlus, word for word:\n\n" + quoted)
        cited = allowed_numbers
        summary_type = "quoted"
    else:
        summary_type = "llm_summary"

    sources = [{"number": n, "title": results[n - 1]["title"], "link": results[n - 1]["link"]}
               for n in sorted(cited)]
    full_answer = answer + "\n\n" + _sources_block(results, cited)
    if patient_id:
        save_patient_memory(patient_id, "Looked up MedlinePlus: "
                            + ", ".join(s["title"] for s in sources) + ".", note_type="search")
    return {"status": "success",
            "message": f"Summarized {len(best)} MedlinePlus passage(s) from {len(cited)} source(s)"
                       + (" (from cache)" if search.get("from_cache") else "") + ".",
            "answer": full_answer,
            "summary_type": summary_type,
            "sources": sources,
            "passages": used_passages}
