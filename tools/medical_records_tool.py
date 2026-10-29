"""
medical_records_tool.py - find patients, read/add/update medical records.

Functions in this file:
    find_patient(query)                       -> search by patient ID or name
    get_patient_history(patient_id)           -> all records, sorted by date
    add_record(patient_id, ..., confirm)      -> add a new medical record
    update_record(record_id, updates, confirm)-> change fields of a record
    add_patient(name, age, ..., confirm)      -> register a new patient

Every function returns a dict that always has:
    "status"  : a short word the code can check, e.g. "success", "not_found"
    "message" : a sentence a person can read

Safety rule for writes (add_record, update_record, add_patient):
    Called with confirm=False (the default), the function writes NOTHING.
    It returns status "needs_confirmation" plus a preview of exactly what
    would be written. The agent or UI shows that preview to the user, and
    only if the user agrees does it call the same function again with
    confirm=True to actually save.
"""

import re
from datetime import datetime
from difflib import SequenceMatcher

import config
from tools.csv_helpers import read_csv, write_csv, next_id
from tools.logger import log_tool_call


# Columns a user may change in update_record. patient_id and record_id are
# deliberately NOT here: they must never change.
EDITABLE_RECORD_FIELDS = [
    "visit_date", "diagnosis", "medications",
    "lab_results", "alerts", "doctor_notes",
]
PROTECTED_RECORD_FIELDS = ["record_id", "patient_id"]

DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clean(text):
    """Lower-case a string and squash extra spaces, for fair comparisons."""
    return " ".join(str(text).lower().split())


def _ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def name_similarity(query, name):
    """Are two names alike? Returns (is_similar, score).

    The FIRST NAME and the SURNAME are compared separately (never the whole
    name together), with SequenceMatcher on lower-case text. Both must reach
    their minimum from config.py (FIRST_NAME_SIMILARITY_MIN, SURNAME_SIMILARITY_MIN):
    "Sanskriti Prasad" is NOT similar to "Ankit Prasad" (same surname, but the
    first names are too different). A single word is compared with the first
    name or the surname. score (0 to 1) is only used to sort the matches.
    """
    q, n = _clean(query).split(), _clean(name).split()
    if not q or not n:
        return False, 0.0
    if len(q) == 1:
        first, last = _ratio(q[0], n[0]), _ratio(q[0], n[-1])
        return (first >= config.FIRST_NAME_SIMILARITY_MIN or last >= config.SURNAME_SIMILARITY_MIN), max(first, last)
    first, last = _ratio(q[0], n[0]), _ratio(q[-1], n[-1])
    return (first >= config.FIRST_NAME_SIMILARITY_MIN and last >= config.SURNAME_SIMILARITY_MIN), (first + last) / 2


def _check_date(date_text):
    """Return an error message if date_text is not a valid past/today date,
    otherwise return None."""
    try:
        date = datetime.strptime(date_text, DATE_FORMAT).date()
    except ValueError:
        return f"'{date_text}' is not a valid date. Use YYYY-MM-DD."
    if date > config.get_now().date():
        return f"Visit date {date_text} is in the future."
    return None


# ---------------------------------------------------------------------------
# 1. Find a patient
# ---------------------------------------------------------------------------

@log_tool_call
def find_patient(query):
    """Find patients by ID (e.g. "P001") or by full/partial name.

    Name search returns ALL possible matches: exact, partial ("kumar"),
    and similar-looking names ("Ramesh Kumar" also returns "Rakesh Kumar"),
    so the agent can ask the user which person they mean instead of
    silently picking the wrong patient.

    Status values: "success" (exactly one match), "multiple_matches",
                   "not_found", "invalid", "error".
    """
    try:
        query = str(query or "").strip()
        if not query:
            return {"status": "invalid", "message": "Please give a patient ID or name."}

        patients = read_csv(config.PATIENTS_CSV)

        # --- Case A: the query looks like a patient ID, e.g. "P001" or "p1" ---
        if re.fullmatch(r"[Pp]\d+", query):
            patient_id = f"P{int(query[1:]):03d}"   # "p1" -> "P001"
            rows = patients[patients["patient_id"] == patient_id]
            if rows.empty:
                return {"status": "not_found",
                        "message": f"No patient with ID {patient_id}.",
                        "matches": []}
            match = rows.iloc[0].to_dict()
            match["match_type"] = "exact_id"
            return {"status": "success",
                    "message": f"Found patient {match['name']} ({patient_id}).",
                    "matches": [match]}

        # --- Case B: search by name ---
        matches = []
        for _, row in patients.iterrows():
            name = row["name"]
            similar, score = name_similarity(query, name)
            if _clean(query) == _clean(name):
                match_type = "exact_name"
            elif _clean(query) in _clean(name):
                match_type = "partial_name"
            elif similar:
                match_type = "similar_name"
            else:
                continue  # not a match, skip this patient
            match = row.to_dict()
            match["match_type"] = match_type
            match["similarity"] = round(score, 2)
            match["is_similar"] = similar
            matches.append(match)

        # Best matches first: exact, then partial, then similar (by score).
        order = {"exact_name": 0, "partial_name": 1, "similar_name": 2}
        matches.sort(key=lambda m: (order[m["match_type"]], -m["similarity"]))

        if not matches:
            return {"status": "not_found",
                    "message": f"No patient found matching '{query}'.",
                    "matches": []}
        if len(matches) == 1:
            m = matches[0]
            return {"status": "success",
                    "message": f"Found patient {m['name']} ({m['patient_id']}).",
                    "matches": matches}
        names = ", ".join(f"{m['name']} ({m['patient_id']}, age {m['age']})" for m in matches)
        return {"status": "multiple_matches",
                "message": f"{len(matches)} patients match '{query}': {names}. "
                           "Please confirm which patient you mean.",
                "matches": matches}

    except Exception as error:
        return {"status": "error", "message": f"Could not search patients: {error}"}


# ---------------------------------------------------------------------------
# 2. Get a patient's history
# ---------------------------------------------------------------------------

@log_tool_call
def get_patient_history(patient_id):
    """Return all medical records for one patient, oldest visit first.

    Status values: "success", "no_records" (patient exists but has none),
                   "not_found" (no such patient), "error".
    """
    try:
        patient_id = str(patient_id).strip().upper()
        patients = read_csv(config.PATIENTS_CSV)
        patient_rows = patients[patients["patient_id"] == patient_id]
        if patient_rows.empty:
            return {"status": "not_found",
                    "message": f"No patient with ID {patient_id}."}
        patient = patient_rows.iloc[0].to_dict()

        records = read_csv(config.MEDICAL_RECORDS_CSV)
        mine = records[records["patient_id"] == patient_id]
        # Dates are stored as YYYY-MM-DD text, which sorts correctly as text.
        mine = mine.sort_values("visit_date")

        if mine.empty:
            return {"status": "no_records",
                    "message": f"{patient['name']} ({patient_id}) has no medical records yet.",
                    "patient": patient,
                    "records": []}

        return {"status": "success",
                "message": f"Found {len(mine)} record(s) for {patient['name']} ({patient_id}).",
                "patient": patient,
                "records": mine.to_dict(orient="records")}

    except Exception as error:
        return {"status": "error", "message": f"Could not read history: {error}"}


# ---------------------------------------------------------------------------
# 3. Add a medical record
# ---------------------------------------------------------------------------

@log_tool_call
def add_record(patient_id, visit_date, diagnosis, medications="", lab_results="",
               alerts="", doctor_notes="", confirm=False):
    """Add a new medical record for an existing patient.

    With confirm=False: returns "needs_confirmation" and a preview, writes nothing.
    With confirm=True : saves the record and returns "success".

    Status values: "needs_confirmation", "success", "not_found", "invalid", "error".
    """
    try:
        patient_id = str(patient_id).strip().upper()
        patients = read_csv(config.PATIENTS_CSV)
        if patient_id not in patients["patient_id"].values:
            return {"status": "not_found", "message": f"No patient with ID {patient_id}."}

        date_error = _check_date(str(visit_date).strip())
        if date_error:
            return {"status": "invalid", "message": date_error}
        if not str(diagnosis).strip():
            return {"status": "invalid", "message": "Diagnosis cannot be empty."}

        records = read_csv(config.MEDICAL_RECORDS_CSV)
        new_record = {
            "record_id": next_id(records["record_id"], "R"),
            "patient_id": patient_id,
            "visit_date": str(visit_date).strip(),
            "diagnosis": str(diagnosis).strip(),
            "medications": str(medications).strip(),
            "lab_results": str(lab_results).strip(),
            "alerts": str(alerts).strip(),
            "doctor_notes": str(doctor_notes).strip(),
        }

        if not confirm:
            return {"status": "needs_confirmation",
                    "message": f"About to add record {new_record['record_id']} for "
                               f"{patient_id}. Please confirm.",
                    "will_write": new_record}

        records.loc[len(records)] = new_record   # append one row at the end
        write_csv(records, config.MEDICAL_RECORDS_CSV)
        return {"status": "success",
                "message": f"Record {new_record['record_id']} added for {patient_id}.",
                "record": new_record}

    except Exception as error:
        return {"status": "error", "message": f"Could not add record: {error}"}


# ---------------------------------------------------------------------------
# 4. Update a medical record
# ---------------------------------------------------------------------------

@log_tool_call
def update_record(record_id, updates, confirm=False):
    """Change one or more fields of an existing record.

    `updates` is a dict, e.g. {"medications": "Losartan 100mg"}.
    record_id and patient_id can never be changed.

    With confirm=False: returns "needs_confirmation" with old -> new values.
    With confirm=True : saves the change.

    Status values: "needs_confirmation", "success", "not_found", "invalid",
                   "no_change", "error".
    """
    try:
        record_id = str(record_id).strip().upper()
        if not isinstance(updates, dict) or not updates:
            return {"status": "invalid", "message": "Give the fields to update as a dict."}

        # Rule: never allow changing the IDs.
        blocked = [f for f in updates if f in PROTECTED_RECORD_FIELDS]
        if blocked:
            return {"status": "invalid",
                    "message": f"Not allowed to change {', '.join(blocked)}."}

        unknown = [f for f in updates if f not in EDITABLE_RECORD_FIELDS]
        if unknown:
            return {"status": "invalid",
                    "message": f"Unknown field(s): {', '.join(unknown)}. "
                               f"Allowed: {', '.join(EDITABLE_RECORD_FIELDS)}."}

        if "visit_date" in updates:
            date_error = _check_date(str(updates["visit_date"]).strip())
            if date_error:
                return {"status": "invalid", "message": date_error}
        if "diagnosis" in updates and not str(updates["diagnosis"]).strip():
            return {"status": "invalid", "message": "Diagnosis cannot be empty."}

        records = read_csv(config.MEDICAL_RECORDS_CSV)
        row_positions = records.index[records["record_id"] == record_id]
        if len(row_positions) == 0:
            return {"status": "not_found", "message": f"No record with ID {record_id}."}
        row = row_positions[0]

        # Build an "old -> new" list, skipping fields whose value is unchanged.
        changes = {}
        for field, new_value in updates.items():
            new_value = str(new_value).strip()
            old_value = records.at[row, field]
            if new_value != old_value:
                changes[field] = {"old": old_value, "new": new_value}

        if not changes:
            return {"status": "no_change",
                    "message": f"Record {record_id} already has these values."}

        if not confirm:
            return {"status": "needs_confirmation",
                    "message": f"About to update record {record_id}. Please confirm.",
                    "will_write": changes}

        for field, change in changes.items():
            records.at[row, field] = change["new"]
        write_csv(records, config.MEDICAL_RECORDS_CSV)
        return {"status": "success",
                "message": f"Record {record_id} updated ({', '.join(changes)}).",
                "changes": changes}

    except Exception as error:
        return {"status": "error", "message": f"Could not update record: {error}"}


# ---------------------------------------------------------------------------
# 5. Add a patient
# ---------------------------------------------------------------------------

PHONE_PROBLEM = ("That doesn't look like a phone number. Use digits only, 7 to 15 digits, "
                 "optionally starting with + and the country code (for example +15550100123).")


def clean_phone(text):
    """Check a phone number. Spaces, dashes, dots and brackets are ignored.

    Returns (number, None) or (None, problem). Valid: optional "+", then 7 to 15 digits.
    An empty value is allowed here and means "no number" ("", None).
    """
    text = str(text or "").strip()
    if not text:
        return "", None
    number = re.sub(r"[\s\-.()]", "", text)
    if re.fullmatch(r"\+?\d{7,15}", number):
        return number, None
    return None, PHONE_PROBLEM


@log_tool_call
def add_patient(name, age, gender, contact_person="self", contact_phone="", confirm=False):
    """Register a new patient.

    Rejects an exact duplicate: same name (ignoring capitals/extra spaces)
    AND same age. A same-name patient with a different age is allowed, but
    the preview warns about it.

    With confirm=False: returns "needs_confirmation" with a preview.
    With confirm=True : saves the patient.

    Status values: "needs_confirmation", "success", "duplicate", "invalid", "error".
    """
    try:
        name = " ".join(str(name).split())   # tidy spaces: " ramesh  kumar " -> "ramesh kumar"
        if not name:
            return {"status": "invalid", "message": "Name cannot be empty."}
        try:
            age = int(age)
        except (TypeError, ValueError):
            return {"status": "invalid", "message": f"Age '{age}' is not a whole number."}
        if not 0 <= age <= 120:
            return {"status": "invalid", "message": "Age must be between 0 and 120."}
        if not str(gender).strip():
            return {"status": "invalid", "message": "Gender cannot be empty."}
        phone, problem = clean_phone(contact_phone)
        if problem:
            return {"status": "invalid", "message": problem}

        patients = read_csv(config.PATIENTS_CSV)
        if "contact_phone" not in patients.columns:        # an older file: add the blank column
            patients.insert(list(patients.columns).index("contact_person") + 1, "contact_phone", "")

        # Exact duplicate check: same cleaned name AND same age.
        for _, row in patients.iterrows():
            if _clean(row["name"]) == _clean(name) and int(row["age"]) == age:
                return {"status": "duplicate",
                        "message": f"{row['name']} (age {age}) already exists as "
                                   f"{row['patient_id']}."}

        new_patient = {
            "patient_id": next_id(patients["patient_id"], "P"),
            "name": name,
            "age": str(age),
            "gender": str(gender).strip(),
            "contact_person": str(contact_person).strip(),
            "contact_phone": phone,
            "created_date": config.get_now().strftime(DATE_FORMAT),
        }

        if not confirm:
            # Warn (but do not block) if similar names already exist.
            similar = [f"{r['name']} ({r['patient_id']}, age {r['age']})"
                       for _, r in patients.iterrows()
                       if name_similarity(name, r["name"])[0]]
            message = f"About to add patient {name} as {new_patient['patient_id']}."
            if similar:
                message += f" Note: similar existing patient(s): {', '.join(similar)}."
            return {"status": "needs_confirmation",
                    "message": message + " Please confirm.",
                    "will_write": new_patient}

        patients.loc[len(patients)] = new_patient
        write_csv(patients, config.PATIENTS_CSV)
        return {"status": "success",
                "message": f"Patient {name} added as {new_patient['patient_id']}.",
                "patient": new_patient}

    except Exception as error:
        return {"status": "error", "message": f"Could not add patient: {error}"}
