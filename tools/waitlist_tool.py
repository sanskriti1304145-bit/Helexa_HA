"""
waitlist_tool.py - the waiting list for specialties with no open slots.

data/waitlist.csv columns:
    waitlist_id, patient_id, specialty, doctor_id, reason, status, created_at, recheck_on

recheck_on is the date (YYYY-MM-DD) when the app should remind staff to look for
openings again: created date + 3, 7 or 14 days. Older rows may have it blank.

status is "waiting" (the only value this tool writes). doctor_id is the
doctor the patient waits for, or empty when ANY doctor of the specialty will do.

Two tools (both logged like the others):
    add_to_waitlist(patient_id, specialty, reason, doctor_id, confirm)
        confirm=False -> preview only, nothing is written
        confirm=True  -> saves the entry
        recheck_days: remind in this many days (default 7)
        A patient can only wait ONCE per specialty at a time (a duplicate of a
        still-waiting entry is rejected).
    list_waitlist(patient_id, specialty, status)
"""

from datetime import timedelta

import config
from tools.appointment_tool import find_doctors
from tools.csv_helpers import next_id, read_csv, write_csv
from tools.logger import log_tool_call

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M"
DATE_FORMAT = "%Y-%m-%d"
COLUMNS = ["waitlist_id", "patient_id", "specialty", "doctor_id", "reason", "status", "created_at",
           "recheck_on"]
MAX_RECHECK_DAYS = 90


@log_tool_call
def add_to_waitlist(patient_id, specialty, reason="", doctor_id="", recheck_days=7, confirm=False):
    """Put a patient on the waiting list for a specialty.

    Status values: "needs_confirmation" (preview, with "will_write"), "success",
                   "duplicate", "not_found" (patient or specialty), "invalid", "error".
    """
    try:
        try:
            recheck_days = int(recheck_days)
        except (TypeError, ValueError):
            return {"status": "invalid", "message": f"Recheck days '{recheck_days}' is not a whole number."}
        if not 1 <= recheck_days <= MAX_RECHECK_DAYS:
            return {"status": "invalid", "message": f"Recheck days must be between 1 and {MAX_RECHECK_DAYS}."}
        patient_id = str(patient_id or "").strip().upper()
        patients = read_csv(config.PATIENTS_CSV)
        patient = patients[patients["patient_id"] == patient_id]
        if patient.empty:
            return {"status": "not_found", "message": f"No patient with ID {patient_id or '(empty)'}."}
        patient_name = patient.iloc[0]["name"]

        found = find_doctors(specialty)
        if found["status"] != "success":
            return {"status": found["status"], "message": found["message"]}
        specialties = {d["specialty"] for d in found["doctors"]}
        if len(specialties) != 1:
            return {"status": "invalid",
                    "message": f"'{specialty}' matches several specialties; please be more specific."}
        specialty = specialties.pop()

        doctor_id = str(doctor_id or "").strip().upper()
        if doctor_id and doctor_id not in [d["doctor_id"] for d in found["doctors"]]:
            return {"status": "invalid", "message": f"{doctor_id} is not a {specialty} doctor."}

        waitlist = read_csv(config.WAITLIST_CSV)
        if "recheck_on" not in waitlist.columns:            # an older file: add the blank column
            waitlist["recheck_on"] = ""
        same = waitlist[(waitlist["patient_id"] == patient_id) & (waitlist["specialty"] == specialty)
                        & (waitlist["status"] == "waiting")]
        if not same.empty:
            return {"status": "duplicate",
                    "message": f"{patient_name} is already on the waiting list for {specialty} "
                               f"({same.iloc[0]['waitlist_id']}).",
                    "entry": same.iloc[0].to_dict()}

        entry = {"waitlist_id": next_id(waitlist["waitlist_id"], "W"), "patient_id": patient_id,
                 "specialty": specialty, "doctor_id": doctor_id,
                 "reason": " ".join(str(reason or "").split()) or f"{specialty} appointment",
                 "status": "waiting", "created_at": config.get_now().strftime(DATE_TIME_FORMAT),
                 "recheck_on": (config.get_now() + timedelta(days=recheck_days)).strftime(DATE_FORMAT)}

        if not confirm:
            return {"status": "needs_confirmation",
                    "message": f"About to add {patient_name} to the waiting list for {specialty} "
                               f"as {entry['waitlist_id']} (recheck on {entry['recheck_on']}). Please confirm.",
                    "will_write": entry}

        waitlist.loc[len(waitlist)] = entry
        write_csv(waitlist[COLUMNS], config.WAITLIST_CSV)
        return {"status": "success",
                "message": f"{patient_name} is on the waiting list for {specialty} ({entry['waitlist_id']}). "
                           f"Recheck on {entry['recheck_on']}.",
                "entry": entry}

    except Exception as error:
        return {"status": "error", "message": f"Could not add to the waiting list: {error}"}


@log_tool_call
def list_waitlist(patient_id=None, specialty=None, status=None):
    """List waiting-list entries (oldest first), optionally filtered.

    Status values: "success" (entries may be empty), "error".
    """
    try:
        entries = read_csv(config.WAITLIST_CSV)
        if patient_id:
            entries = entries[entries["patient_id"] == str(patient_id).strip().upper()]
        if specialty:
            entries = entries[entries["specialty"].str.lower() == str(specialty).strip().lower()]
        if status:
            entries = entries[entries["status"] == status]
        entries = entries.sort_values("created_at", kind="stable")
        return {"status": "success", "message": f"Found {len(entries)} waiting-list entr(ies).",
                "entries": entries.to_dict(orient="records")}
    except Exception as error:
        return {"status": "error", "message": f"Could not read the waiting list: {error}"}
