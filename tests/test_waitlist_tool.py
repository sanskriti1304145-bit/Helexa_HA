"""
test_waitlist_tool.py - tests for tools/waitlist_tool.py (the waiting list)

Run from the project root:
    python -m tests.test_waitlist_tool

Runs on a COPY of the data (test_data/); the real data/ is never touched.
"""

import os

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from tools.csv_helpers import read_csv, write_csv  # noqa: E402
from tools.logger import read_logs  # noqa: E402
from tools.waitlist_tool import COLUMNS, add_to_waitlist, list_waitlist  # noqa: E402


def rows():
    return read_csv(config.WAITLIST_CSV)


def test_file_and_preview():
    print("waitlist.csv and the preview")
    check("original data/waitlist.csv has the right columns and no rows",
          list(read_csv(ORIGINAL_DATA_DIR / "waitlist.csv").columns) == COLUMNS == [
              "waitlist_id", "patient_id", "specialty", "doctor_id", "reason", "status", "created_at", "recheck_on"]
          and len(read_csv(ORIGINAL_DATA_DIR / "waitlist.csv")) == 0)
    before = fingerprint(config.DATA_DIR)
    r = add_to_waitlist("P010", "psychiatrist", "anxiety")
    check("preview -> needs_confirmation with will_write", r["status"] == "needs_confirmation"
          and r["will_write"]["waitlist_id"] == "W001" and r["will_write"]["specialty"] == "Psychiatry"
          and r["will_write"]["status"] == "waiting" and r["will_write"]["created_at"] == "2026-10-21 12:00", r)
    check("  recheck_on defaults to created date + 7 days", r["will_write"]["recheck_on"] == "2026-10-28", r)
    for days, expected in [(3, "2026-10-24"), (14, "2026-11-04")]:
        check(f"  recheck_days={days} -> {expected}", add_to_waitlist("P010", "Psychiatry", recheck_days=days)
              ["will_write"]["recheck_on"] == expected)
    for bad in (0, -1, 91, "soon"):
        check(f"  recheck_days={bad!r} -> invalid", add_to_waitlist("P010", "Psychiatry", recheck_days=bad)["status"] == "invalid")
    check("  a preview writes nothing", fingerprint(config.DATA_DIR) == before and len(rows()) == 0)


def test_confirm_and_duplicates():
    print("confirm, duplicates")
    r = add_to_waitlist("P010", "Psychiatry", "anxiety", confirm=True)
    check("confirm -> saved as W001", r["status"] == "success" and len(rows()) == 1
          and rows().iloc[0]["waitlist_id"] == "W001" and rows().iloc[0]["reason"] == "anxiety", r)
    for confirm in (False, True):
        r = add_to_waitlist("p010", "psychiatrist", "again", confirm=confirm)
        check(f"same patient + specialty still waiting -> duplicate (confirm={confirm})",
              r["status"] == "duplicate" and "already on the waiting list" in r["message"] and len(rows()) == 1, r)
    r = add_to_waitlist("P010", "Cardiology", confirm=True)
    check("same patient, other specialty -> allowed (W002)", r["status"] == "success"
          and r["entry"]["waitlist_id"] == "W002" and r["entry"]["reason"] == "Cardiology appointment", r)
    r = add_to_waitlist("P001", "Psychiatry", confirm=True)
    check("other patient, same specialty -> allowed", r["status"] == "success" and len(rows()) == 3)
    waitlist = rows()
    waitlist.loc[waitlist["waitlist_id"] == "W001", "status"] = "booked"
    write_csv(waitlist, config.WAITLIST_CSV)
    r = add_to_waitlist("P010", "Psychiatry", confirm=True)
    check("an entry that is no longer 'waiting' does not block a new one", r["status"] == "success", r)


def test_bad_input():
    print("bad input")
    before = fingerprint(config.DATA_DIR)
    check("unknown patient -> not_found", add_to_waitlist("P999", "Psychiatry", confirm=True)["status"] == "not_found")
    check("unknown specialty -> not_found", add_to_waitlist("P001", "Oncology", confirm=True)["status"] == "not_found")
    check("empty specialty -> invalid", add_to_waitlist("P001", "", confirm=True)["status"] == "invalid")
    check("doctor from another specialty -> invalid",
          add_to_waitlist("P002", "Psychiatry", doctor_id="D01", confirm=True)["status"] == "invalid")
    check("none of these wrote anything", fingerprint(config.DATA_DIR) == before)
    r = add_to_waitlist("P002", "Psychiatry", doctor_id="d10", confirm=True)
    check("a doctor of the specialty is accepted and stored", r["status"] == "success"
          and r["entry"]["doctor_id"] == "D10", r)


def test_old_file_without_recheck_column():
    print("an older waitlist.csv without recheck_on still works")
    path = config.WAITLIST_CSV
    saved = path.read_text()
    old = read_csv(path).drop(columns=["recheck_on"])
    write_csv(old, path)
    r = add_to_waitlist("P004", "Psychiatry", confirm=True, recheck_days=3)
    check("adds the column and the entry", r["status"] == "success" and "recheck_on" in read_csv(path).columns
          and read_csv(path).iloc[-1]["recheck_on"] == "2026-10-24", r)
    check("  old rows keep a blank recheck_on", (read_csv(path).iloc[:-1]["recheck_on"] == "").all())
    path.write_text(saved)


def test_list_and_log():
    print("listing and logging")
    everyone = list_waitlist()
    check("list_waitlist returns every entry", everyone["status"] == "success" and len(everyone["entries"]) == len(rows()))
    check("filter by patient", {e["patient_id"] for e in list_waitlist(patient_id="p010")["entries"]} == {"P010"})
    check("filter by specialty", {e["specialty"] for e in list_waitlist(specialty="psychiatry")["entries"]} == {"Psychiatry"})
    check("filter by status", {e["status"] for e in list_waitlist(status="booked")["entries"]} == {"booked"})
    logged = [r for r in read_logs() if r["tool_name"] in ("add_to_waitlist", "list_waitlist")]
    check("both tools are logged like the others", {r["tool_name"] for r in logged} == {"add_to_waitlist", "list_waitlist"}
          and any(r["status"] == "needs_confirmation" for r in logged) and any(r["status"] == "duplicate" for r in logged))


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    test_file_and_preview()
    test_confirm_and_duplicates()
    test_bad_input()
    test_old_file_without_recheck_column()
    test_list_and_log()
    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
