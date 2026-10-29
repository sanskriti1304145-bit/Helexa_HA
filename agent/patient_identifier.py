"""
patient_identifier.py - work out which patient a request is about.

Never guesses, and ALWAYS asks the user to confirm before using a patient.

A) The user typed a name or ID  -> find_patient(name), then:
     - EXACT full-name matches (ignoring case) come first. There can be several
       (two patients with the same name), each shown with age, gender and ID.
     - SIMILAR names come after them, labeled "Similar name", only if the
       surname AND the first name are alike enough (the two minimums are in
       config.py; they are compared separately, see name_similarity()).
       "Ankit Prasad" is never offered for "Sanskriti Prasad".
     - No exact match -> the similar names are offered as "Did you mean...?".
     - Nothing at all -> "needs_name": the patient was not found.
     - Age/gender from the request only NARROW the similar names (and add a
       note when a patient does not fit). They never add a patient.
     - exactly one patient in the end -> "needs_confirmation": a confirmation
       card (name, age, gender, patient ID, contact person; no phone number and
       no medical information) and "Is this the patient you're looking for?"
     - several -> "needs_choice": a dropdown list; the list always gets
       "None of these" (added by the agent).

B) No name given -> search_patient_summaries(request) (FAISS), keep the
   candidates whose age and gender fit. Exactly one left -> confirmation card.
   Otherwise "needs_name": asks only "What is the patient's full name or
   patient ID?" and shows NO other patients (privacy).
"""

import config
from memory.vector_store import search_patient_summaries
from tools.csv_helpers import read_csv
from tools.logger import log_tool_call
from tools.medical_records_tool import find_patient

SEARCH_CANDIDATES = 5   # how many FAISS matches to check against the facts
ASK_NAME_QUESTION = "What is the patient's full name or patient ID?"


def _describe(patient):
    return f"{patient['name']} ({patient['patient_id']}, age {patient['age']}, {patient['gender'].lower()})"


CONFIRM_QUESTION = "Is this the patient you're looking for?"


def patient_card(patient):
    """The confirmation card. Only basic identity details: never the phone
    number and never anything medical."""
    return (f"Name: {patient['name']}\nAge: {patient['age']}\nGender: {patient['gender']}\n"
            f"Patient ID: {patient['patient_id']}\nContact person: {patient['contact_person']}")


def _mismatch_note(patient, facts):
    """'' if the patient fits the age/gender from the request, otherwise a short note."""
    parts = []
    if facts.get("age") is not None and int(patient["age"]) != facts["age"]:
        parts.append(f"age {facts['age']}")
    if facts.get("gender") and patient["gender"].lower() != facts["gender"]:
        parts.append(facts["gender"])
    return f"does not match {' and '.join(parts)}" if parts else ""


def _matches_facts(patient, facts):
    """True if the patient's age and gender agree with every known fact."""
    if facts.get("age") is not None and int(patient["age"]) != facts["age"]:
        return False
    if facts.get("gender") and patient["gender"].lower() != facts["gender"]:
        return False
    return True


@log_tool_call
def identify_patient(request, name=None, facts=None):
    """Find the one patient the request is about, or ask for the name.

    request: the user's words, used for the FAISS search
    name:    a name or patient ID, if known
    facts:   {"age": ..., "gender": ...} from planner.extract_patient_facts

    Status values: "needs_confirmation" (with "patient"), "needs_choice"
                   (with "candidates"), "needs_name" (only a "candidate_count"), "error".
    """
    facts = facts or {}
    name = (name or "").strip()
    patients = read_csv(config.PATIENTS_CSV).set_index("patient_id", drop=False)
    used = ", ".join(f"{k}={v}" for k, v in facts.items() if v is not None) or "none"

    if name:
        return _identify_by_name(name, facts, patients, used)

    # --- B) no name: description search, then the hard facts from patients.csv ---
    found = search_patient_summaries(request, k=SEARCH_CANDIDATES)
    if found["status"] != "success":
        return {"status": "error", "message": found["message"]}
    candidates = [patients.loc[m["patient_id"]].to_dict() for m in found["matches"]
                  if m["patient_id"] in patients.index]
    filtered = [p for p in candidates if _matches_facts(p, facts)]
    if len(filtered) == 1:
        return _confirmation(filtered[0], f"Found one match (patient summary search, filters: {used}): "
                                          f"{_describe(filtered[0])}.")
    # PRIVACY: never reveal other patients here. The message (shown in logs/traces)
    # gives only a count, and the question is fixed.
    return {"status": "needs_name",
            "message": f"{len(filtered)} possible patient(s) (patient summary search, filters: {used}); "
                       "asked for the name.",
            "candidate_count": len(filtered),
            "question": ASK_NAME_QUESTION}


def _confirmation(patient, message, intro="", note=""):
    """One patient: the card plus 'Is this the patient you're looking for?'."""
    blocks = [intro, patient_card(patient), f"Note: this record {note}." if note else "", CONFIRM_QUESTION]
    return {"status": "needs_confirmation", "message": message, "patient": patient,
            "question": "\n\n".join(block for block in blocks if block)}


def _identify_by_name(name, facts, patients, used):
    found = find_patient(name)
    if found["status"] == "error":
        return {"status": "error", "message": found["message"]}

    exact, similar = [], []
    for m in found.get("matches", []):
        patient = patients.loc[m["patient_id"]].to_dict()
        if m["match_type"] in ("exact_name", "exact_id"):
            exact.append(patient)
        elif m["match_type"] == "similar_name" or (m["match_type"] == "partial_name" and m.get("is_similar")):
            patient["match_label"] = "Similar name"
            similar.append(patient)

    # Age/gender only narrow the similar names (never remove an exact match, never add anyone).
    narrowed = [p for p in similar if _matches_facts(p, facts)]
    if narrowed:
        similar = narrowed
    elif any(_matches_facts(p, facts) for p in exact):
        similar = []          # an exact match fits the facts: no need for look-alikes
    candidates = exact + similar
    for p in candidates:
        p["note"] = _mismatch_note(p, facts)

    if not candidates:
        return {"status": "needs_name",
                "message": f"No patient found for '{name}' (filters: {used}).",
                "candidate_count": 0,
                "question": f"I could not find a patient named '{name}'. " + ASK_NAME_QUESTION}

    if len(candidates) == 1:
        patient = candidates[0]
        intro = ""
        if not exact:
            intro = f"I couldn't find '{name}' exactly. Did you mean this patient (the name is spelled differently)?"
        return _confirmation(patient, f"Found one match for '{name}' (filters: {used}): {_describe(patient)}.",
                             intro=intro, note=patient["note"])

    if exact:
        question = f"Several patients match '{name}'. Which one do you mean?"
        if any(p["note"] for p in candidates):
            question = (f"'{name}' matches more than one patient, and some details differ from your "
                        f"request ({used}). Which one do you mean?")
    else:
        question = f"I couldn't find '{name}' exactly. Did you mean one of these?"
    return {"status": "needs_choice",
            "message": f"{len(candidates)} patients match '{name}' ({len(exact)} exact, "
                       f"{len(similar)} similar; filters: {used}).",
            "candidates": candidates,
            "question": question}
