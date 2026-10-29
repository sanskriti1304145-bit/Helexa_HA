"""
test_medical_records_tool.py - tests for tools/medical_records_tool.py

Run from the project root:
    python -m tests.test_medical_records_tool

All tests run on a fresh COPY of data/ (test_data/). At the end we check
that the original files in data/ were not changed at all.
"""

import os

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR,
)

# Switch to the test copy and freeze "now" BEFORE importing config/tools.
use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from tools.csv_helpers import read_csv  # noqa: E402
from tools.medical_records_tool import (  # noqa: E402
    find_patient, get_patient_history, add_record, update_record, add_patient,
)

def ids(result):
    return [m["patient_id"] for m in result.get("matches", [])]


# ---------------------------------------------------------------------------

def test_find_patient():
    print("find_patient")
    r = find_patient("P001")
    check("find by ID P001 -> Ramesh Kumar", r["status"] == "success" and ids(r) == ["P001"], r)

    r = find_patient("p1")
    check("lower-case short ID 'p1' -> P001", ids(r) == ["P001"], r)

    r = find_patient("P999")
    check("unknown ID -> not_found", r["status"] == "not_found", r)

    r = find_patient("kumar")
    check("partial 'kumar' -> both Kumars", r["status"] == "multiple_matches"
          and set(ids(r)) == {"P001", "P011"}, r)

    r = find_patient("Ramesh Kumar")
    check("'Ramesh Kumar' -> also returns similar 'Rakesh Kumar'",
          r["status"] == "multiple_matches" and ids(r) == ["P001", "P011"], r)
    check("  exact match is listed first", r["matches"][0]["match_type"] == "exact_name", r)

    r = find_patient("  ANITA  ")
    check("case/space-insensitive 'ANITA' -> Anita Sharma only", ids(r) == ["P002"], r)

    r = find_patient("Zzzz Nobody")
    check("unknown name -> not_found", r["status"] == "not_found", r)

    r = find_patient("")
    check("empty query -> invalid", r["status"] == "invalid", r)


def test_get_patient_history():
    print("get_patient_history")
    r = get_patient_history("P001")
    dates = [rec["visit_date"] for rec in r["records"]]
    check("P001 has 4 records", r["status"] == "success" and len(dates) == 4, r)
    check("  records sorted oldest -> newest", dates == sorted(dates), dates)
    check("  'None' text kept as text (not lost)",
          get_patient_history("P002")["records"][0]["lab_results"] == "None")

    r = get_patient_history("P012")
    check("P012 (Daniel Brooks) -> no_records", r["status"] == "no_records"
          and r["records"] == [], r)

    r = get_patient_history("P999")
    check("unknown patient -> not_found", r["status"] == "not_found", r)


def test_add_record():
    print("add_record")
    before = len(read_csv(config.MEDICAL_RECORDS_CSV))

    r = add_record("P012", "2026-10-15", "Seasonal allergies", "Cetirizine 10mg")
    check("without confirm -> needs_confirmation", r["status"] == "needs_confirmation", r)
    check("  preview shows new ID R024", r["will_write"]["record_id"] == "R024", r)
    check("  nothing written yet", len(read_csv(config.MEDICAL_RECORDS_CSV)) == before)

    r = add_record("P012", "2026-10-15", "Seasonal allergies", "Cetirizine 10mg", confirm=True)
    check("with confirm=True -> success", r["status"] == "success", r)
    check("  one row added", len(read_csv(config.MEDICAL_RECORDS_CSV)) == before + 1)
    check("  P012 now has history", get_patient_history("P012")["status"] == "success")

    check("unknown patient -> not_found",
          add_record("P999", "2026-10-15", "Flu")["status"] == "not_found")
    check("bad date -> invalid",
          add_record("P001", "15/10/2026", "Flu")["status"] == "invalid")
    check("future date -> invalid",
          add_record("P001", "2027-01-01", "Flu")["status"] == "invalid")
    check("empty diagnosis -> invalid",
          add_record("P001", "2026-10-15", "  ")["status"] == "invalid")


def test_update_record():
    print("update_record")
    r = update_record("R004", {"patient_id": "P002"})
    check("changing patient_id is rejected", r["status"] == "invalid", r)
    r = update_record("R004", {"record_id": "R999"})
    check("changing record_id is rejected", r["status"] == "invalid", r)
    r = update_record("R004", {"medications": "x", "patient_id": "P002"}, confirm=True)
    check("mixed update incl. patient_id is rejected entirely", r["status"] == "invalid", r)
    check("  record unchanged",
          read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").at["R004", "patient_id"] == "P001")

    check("unknown field -> invalid",
          update_record("R004", {"blood_type": "O+"})["status"] == "invalid")
    check("unknown record -> not_found",
          update_record("R999", {"medications": "x"})["status"] == "not_found")

    r = update_record("R004", {"medications": "Losartan 100mg; sodium bicarbonate"})
    check("same value -> no_change", r["status"] == "no_change", r)

    r = update_record("R004", {"doctor_notes": "Updated note."})
    check("without confirm -> needs_confirmation with old/new",
          r["status"] == "needs_confirmation" and r["will_write"]["doctor_notes"]["new"] == "Updated note.", r)
    check("  nothing written yet",
          read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").at["R004", "doctor_notes"] != "Updated note.")

    r = update_record("R004", {"doctor_notes": "Updated note."}, confirm=True)
    check("with confirm=True -> success", r["status"] == "success", r)
    check("  value saved",
          read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").at["R004", "doctor_notes"] == "Updated note.")


def test_add_patient():
    print("add_patient")
    r = add_patient("Ramesh Kumar", 70, "Male", "son")
    check("exact duplicate (same name + age) -> duplicate", r["status"] == "duplicate", r)
    r = add_patient("  ramesh   KUMAR ", "70", "Male")
    check("duplicate ignoring case/spaces -> duplicate", r["status"] == "duplicate", r)

    r = add_patient("Ramesh Kumar", 45, "Male")
    check("same name, different age -> allowed (needs_confirmation)",
          r["status"] == "needs_confirmation", r)
    check("  preview warns about similar names", "similar existing" in r["message"], r)
    check("  preview new ID P013 and today's date",
          r["will_write"]["patient_id"] == "P013" and r["will_write"]["created_date"] == "2026-10-21", r)

    check("bad age -> invalid", add_patient("New Person", "seventy", "Male")["status"] == "invalid")
    check("age out of range -> invalid", add_patient("New Person", 150, "Male")["status"] == "invalid")
    check("empty name -> invalid", add_patient("  ", 30, "Male")["status"] == "invalid")

    r = add_patient("Phone Tester", 33, "Female", "self", "+1 555-010-0199")
    check("phone: spaces/dashes are ignored, shown in the preview's will_write",
          r["status"] == "needs_confirmation" and r["will_write"]["contact_phone"] == "+15550100199", r)
    check("phone: the preview message does not repeat the number", "5550100199" not in r["message"])
    for bad in ["12345", "abc1234567", "+1555", "55501001234567890", "555-01x-0199"]:
        r = add_patient("Phone Tester", 33, "Female", "self", bad)
        check(f"phone {bad!r} -> invalid with a clear message", r["status"] == "invalid"
              and "7 to 15 digits" in r["message"], r)
    saved_file = config.PATIENTS_CSV.read_text()            # restored below: later tests count patients
    r = add_patient("Phone Tester", 33, "Female", "self", "5550100199", confirm=True)
    saved = read_csv(config.PATIENTS_CSV).set_index("patient_id").loc[r["patient"]["patient_id"]]
    check("phone saved in contact_phone", r["status"] == "success" and saved["contact_phone"] == "5550100199", saved.to_dict())
    check("existing patients keep a blank phone", (read_csv(config.PATIENTS_CSV).iloc[:12]["contact_phone"] == "").all())
    config.PATIENTS_CSV.write_text(saved_file)
    r = add_patient("Grace Hopper", 40, "Female", confirm=True)
    check("with confirm=True -> success", r["status"] == "success", r)
    check("  new patient is findable", ids(find_patient("Grace Hopper")) == ["P013"])


def test_every_result_has_status_and_message():
    print("return format")
    results = [
        find_patient("P001"), find_patient(""), get_patient_history("P012"),
        add_record("P999", "2026-01-01", "x"), update_record("R001", {"record_id": "x"}),
        add_patient("Ramesh Kumar", 70, "Male"),
    ]
    check("every result has 'status' and 'message'",
          all("status" in r and "message" in r for r in results))


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)

    test_find_patient()
    test_get_patient_history()
    test_add_record()
    test_update_record()
    test_add_patient()
    test_every_result_has_status_and_message()

    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
