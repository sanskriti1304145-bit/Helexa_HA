"""
test_appointment_tool.py - tests for tools/appointment_tool.py

Run from the project root:
    python -m tests.test_appointment_tool

All tests run on a fresh COPY of data/ (test_data/), with "now" frozen at
2026-10-21 12:00. With that date:
  - D01 (nephrology) slots S001 (10-19) and S003 (10-20) are in the past
  - S004 (10-21 14:00) is later today, so it is still bookable
  - every D10 (psychiatry) slot is booked
"""

import os

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count,
    ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from tools import appointment_tool  # noqa: E402
from tools.appointment_tool import (  # noqa: E402
    find_doctors, list_available_slots, book_appointment,
    cancel_appointment, list_appointments,
)
from tools.csv_helpers import read_csv, write_csv  # noqa: E402


def slot_status(slot_id):
    return read_csv(config.SLOTS_CSV).set_index("slot_id").at[slot_id, "status"]


def appt_status(appointment_id):
    return read_csv(config.APPOINTMENTS_CSV).set_index("appointment_id").at[appointment_id, "status"]


def slot_ids(result):
    return [s["slot_id"] for s in result.get("slots", [])]


# ---------------------------------------------------------------------------

def test_find_doctors():
    print("find_doctors")
    r = find_doctors("Nephrology")
    check("'Nephrology' -> Dr. Meera Shah", r["status"] == "success"
          and [d["doctor_id"] for d in r["doctors"]] == ["D01"], r)
    check("'nephrologist' (everyday word) -> D01",
          [d["doctor_id"] for d in find_doctors("  Nephrologist ")["doctors"]] == ["D01"])
    check("partial 'cardio' -> D02",
          [d["doctor_id"] for d in find_doctors("cardio")["doctors"]] == ["D02"])
    r = find_doctors("Oncology")
    check("unknown specialty -> not_found (lists known ones)",
          r["status"] == "not_found" and "Nephrology" in r["message"], r)
    check("empty -> invalid", find_doctors("")["status"] == "invalid")


def test_list_available_slots():
    print("list_available_slots")
    r = list_available_slots(doctor_id="D01")
    check("D01 -> only future, available slots, soonest first",
          slot_ids(r) == ["S004", "S005", "S006", "S007", "S008"], r)
    check("  past slots S001/S003 excluded, booked S002 excluded",
          not {"S001", "S002", "S003"} & set(slot_ids(r)))
    check("  each slot includes doctor name and clinic",
          r["slots"][0]["doctor_name"] == "Dr. Meera Shah" and r["slots"][0]["clinic"])

    check("by specialty 'nephrologist' gives the same slots",
          slot_ids(list_available_slots(specialty="nephrologist")) == slot_ids(r))

    r = list_available_slots(specialty="psychiatrist")
    check("psychiatrist fully booked -> no_slots with clear message",
          r["status"] == "no_slots" and r["message"] == "Dr. Kavita Rao (Psychiatry) has no openings in the next 2 weeks.", r)

    from tools.appointment_tool import no_openings_message
    doctors = read_csv(config.DOCTORS_CSV)
    check("message for several doctors uses 'have'", no_openings_message(doctors.iloc[:2]) ==
          "Dr. Meera Shah (Nephrology) and Dr. Arjun Mehta (Cardiology) have no openings in the next 2 weeks.")
    check("'All upcoming slots are booked' is gone", "booked" not in r["message"])

    check("unknown doctor -> not_found", list_available_slots(doctor_id="D99")["status"] == "not_found")
    check("unknown specialty -> not_found",
          list_available_slots(specialty="oncology")["status"] == "not_found")
    check("no arguments -> invalid", list_available_slots()["status"] == "invalid")


def test_book_appointment():
    print("book_appointment")
    before = fingerprint(TEST_DATA_DIR)

    r = book_appointment("P001", "S004", "CKD stage 3b review")
    check("without confirm -> needs_confirmation", r["status"] == "needs_confirmation", r)
    check("  preview shows A011 with Dr. Meera Shah",
          r["will_write"]["appointment"]["appointment_id"] == "A011"
          and "Dr. Meera Shah" in r["message"], r)
    check("  nothing written yet", fingerprint(TEST_DATA_DIR) == before)

    r = book_appointment("P001", "S004", "CKD stage 3b review", confirm=True)
    check("with confirm=True -> success", r["status"] == "success", r)
    check("  slot S004 is now booked", slot_status("S004") == "booked")
    check("  appointment A011 saved as confirmed", appt_status("A011") == "confirmed")
    check("  created_at uses 'now'", r["appointment"]["created_at"] == "2026-10-21 12:00", r)
    check("  S004 no longer listed as open", "S004" not in slot_ids(list_available_slots(doctor_id="D01")))

    # Each of these must fail AND change nothing.
    unchanged = fingerprint(TEST_DATA_DIR)
    cases = [
        ("same slot again -> slot_unavailable", ("P004", "S004", "Review"), "slot_unavailable"),
        ("already booked S002 -> slot_unavailable", ("P004", "S002", "Review"), "slot_unavailable"),
        ("past slot S003 -> slot_in_past", ("P004", "S003", "Review"), "slot_in_past"),
        ("psychiatry slot (booked) -> slot_unavailable", ("P012", "S073", "Anxiety"), "slot_unavailable"),
        ("unknown slot -> not_found", ("P001", "S999", "Review"), "not_found"),
        ("unknown patient -> not_found", ("P999", "S005", "Review"), "not_found"),
        ("empty reason -> invalid", ("P001", "S005", "   "), "invalid"),
    ]
    for description, args, expected in cases:
        r = book_appointment(*args, confirm=True)
        check(description, r["status"] == expected, r)
    check("  none of the failed bookings changed any file", fingerprint(TEST_DATA_DIR) == unchanged)

    # Same patient, two doctors, same date and time (S007 D01 and S023 D03: 10-26 09:30).
    book_appointment("P001", "S007", "CKD follow-up", confirm=True)
    r = book_appointment("P001", "S023", "General check", confirm=True)
    check("patient double-booked at same time -> conflict", r["status"] == "conflict", r)


def test_slot_taken_between_preview_and_confirm():
    print("re-check before booking")
    r = book_appointment("P004", "S005", "Kidney function review")
    check("preview for S005 is fine", r["status"] == "needs_confirmation", r)

    # Someone else books S005 in the meantime (simulated by editing the copy).
    slots = read_csv(config.SLOTS_CSV)
    slots.loc[slots["slot_id"] == "S005", "status"] = "booked"
    write_csv(slots, config.SLOTS_CSV)
    count_before = len(read_csv(config.APPOINTMENTS_CSV))

    r = book_appointment("P004", "S005", "Kidney function review", confirm=True)
    check("confirm after slot was taken -> slot_unavailable", r["status"] == "slot_unavailable", r)
    check("  no appointment was added", len(read_csv(config.APPOINTMENTS_CSV)) == count_before)

    # Put S005 back for later tests.
    slots.loc[slots["slot_id"] == "S005", "status"] = "available"
    write_csv(slots, config.SLOTS_CSV)


def test_cancel_appointment():
    print("cancel_appointment")
    r = cancel_appointment("A011")
    check("without confirm -> needs_confirmation", r["status"] == "needs_confirmation", r)
    check("  nothing written yet", appt_status("A011") == "confirmed" and slot_status("S004") == "booked")

    r = cancel_appointment("A011", confirm=True)
    check("with confirm=True -> success", r["status"] == "success", r)
    check("  A011 is cancelled", appt_status("A011") == "cancelled")
    check("  slot S004 is available again", slot_status("S004") == "available")

    unchanged = fingerprint(TEST_DATA_DIR)
    r = cancel_appointment("A011", confirm=True)
    check("cancel A011 again -> already_cancelled", r["status"] == "already_cancelled", r)
    r = cancel_appointment("A002", confirm=True)
    check("cancel A002 (cancelled in original data) -> already_cancelled",
          r["status"] == "already_cancelled", r)
    r = cancel_appointment("A999", confirm=True)
    check("cancel unknown A999 -> not_found", r["status"] == "not_found", r)
    r = cancel_appointment("A001", confirm=True)
    check("cancel A001 (slot already passed) -> invalid", r["status"] == "invalid", r)
    check("  all of these said 'Nothing was changed'",
          all("Nothing was changed" in cancel_appointment(a, confirm=True)["message"]
              for a in ["A011", "A002", "A999", "A001"]))
    check("  and really changed nothing", fingerprint(TEST_DATA_DIR) == unchanged)


def test_rollback():
    print("rollback when the second write fails")
    real_write_csv = appointment_tool.write_csv

    def failing_write(fail_on_path):
        """A fake write_csv that crashes for one file and works for others."""
        def fake(df, path):
            if path == fail_on_path:
                raise OSError("simulated disk error")
            real_write_csv(df, path)
        return fake

    # Booking writes Slots first, then Appointments. Make Appointments fail.
    before = fingerprint(TEST_DATA_DIR)
    appointment_tool.write_csv = failing_write(config.APPOINTMENTS_CSV)
    r = book_appointment("P004", "S005", "Kidney function review", confirm=True)
    appointment_tool.write_csv = real_write_csv
    check("booking: second write fails -> error", r["status"] == "error", r)
    check("  message says the change was undone", "undone" in r["message"], r)
    check("  slot S005 rolled back to available", slot_status("S005") == "available")
    check("  both files exactly as before", fingerprint(TEST_DATA_DIR) == before)

    # Cancelling writes Appointments first, then Slots. Make Slots fail.
    book_appointment("P004", "S005", "Kidney function review", confirm=True)
    new_id = list_appointments(patient_id="P004", status="confirmed")["appointments"][-1]["appointment_id"]
    before = fingerprint(TEST_DATA_DIR)
    appointment_tool.write_csv = failing_write(config.SLOTS_CSV)
    r = cancel_appointment(new_id, confirm=True)
    appointment_tool.write_csv = real_write_csv
    check("cancelling: second write fails -> error", r["status"] == "error", r)
    check(f"  {new_id} rolled back to confirmed", appt_status(new_id) == "confirmed")
    check("  both files exactly as before", fingerprint(TEST_DATA_DIR) == before)


def test_list_appointments():
    print("list_appointments")
    r = list_appointments(patient_id="P001")
    check("P001 appointments include date, time and doctor name",
          r["status"] == "success" and all(a["date"] and a["doctor_name"] for a in r["appointments"]), r)
    dates = [(a["date"], a["time"]) for a in r["appointments"]]
    check("  sorted soonest first", dates == sorted(dates), dates)
    r = list_appointments(doctor_id="D10", status="confirmed")
    check("D10 has 8 confirmed appointments", len(r["appointments"]) == 8, r["message"])
    check("patient with none -> no_appointments",
          list_appointments(patient_id="P011")["status"] == "no_appointments")


def test_files_agree():
    print("consistency")
    slots = read_csv(config.SLOTS_CSV).set_index("slot_id")["status"]
    appts = read_csv(config.APPOINTMENTS_CSV)
    confirmed_slots = set(appts[appts["status"] == "confirmed"]["slot_id"])
    booked_slots = set(slots[slots == "booked"].index)
    check("every booked slot has exactly one confirmed appointment, and vice versa",
          confirmed_slots == booked_slots
          and appts[appts["status"] == "confirmed"]["slot_id"].is_unique,
          (confirmed_slots ^ booked_slots))


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)

    test_find_doctors()
    test_list_available_slots()
    test_book_appointment()
    test_slot_taken_between_preview_and_confirm()
    test_cancel_appointment()
    test_rollback()
    test_list_appointments()
    test_files_agree()

    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
