"""
test_reset_data.py - tests for data_seed/ and reset_data.py

Run from the project root:
    python -m tests.test_reset_data

Only reads data_seed/ and works in temporary copies under test_data/; data/ is not touched.
"""

import shutil

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR, PROJECT_ROOT,
)

use_fresh_test_data()

import reset_data  # noqa: E402
from app import dashboard_data  # noqa: E402
from evaluation import evaluate  # noqa: E402

SEED = PROJECT_ROOT / "data_seed"


def rows(name, folder=SEED):
    return reset_data.read(folder, name)


def test_seed_contents():
    print("data_seed/ is the clean baseline")
    counts = reset_data.row_counts(SEED)
    check("row counts: 12 patients, 23 records, 10 doctors, 80 slots, 10 appointments, 65 template rows, empty waitlist",
          counts == {"patients.csv": 12, "medical_records.csv": 23, "doctors.csv": 10, "slots.csv": 80,
                     "appointment.csv": 10, "schedule_template.csv": 65, "waitlist.csv": 0}, counts)
    patients = rows("patients.csv")
    check("patients P001..P012, new schema, every contact_phone blank",
          patients["patient_id"].tolist() == [f"P{i:03d}" for i in range(1, 13)]
          and "contact_phone" in patients.columns and (patients["contact_phone"] == "").all())
    check("records R001..R023", rows("medical_records.csv")["record_id"].tolist() == [f"R{i:03d}" for i in range(1, 24)])
    check("appointments A001..A010 only", rows("appointment.csv")["appointment_id"].tolist()
          == [f"A{i:03d}" for i in range(1, 11)])
    check("slots S001..S080", rows("slots.csv")["slot_id"].tolist() == [f"S{i:03d}" for i in range(1, 81)])
    check("waitlist has its header (incl. recheck_on) and no rows",
          list(rows("waitlist.csv").columns) == ["waitlist_id", "patient_id", "specialty", "doctor_id", "reason",
                                                 "status", "created_at", "recheck_on"])
    check("no test patients (Sangeeta, Sanskriti, Ankit, Ramya ...)",
          not any(x in " ".join(patients["name"]) for x in ("Sangeeta", "Sanskriti", "Ankit", "Ramya", "Grace")))
    check("schedule template has no rows for Dr. Kavita Rao (D10)",
          (rows("schedule_template.csv")["doctor_id"] != "D10").all())
    check("seed has no consistency problems", reset_data.consistency_problems(SEED) == [],
          reset_data.consistency_problems(SEED))


def test_consistency_checker():
    print("the consistency check finds planted problems")
    copy = TEST_DATA_DIR / "seed_copy"
    shutil.copytree(SEED, copy)
    check("a clean copy has no problems", reset_data.consistency_problems(copy) == [])
    slots = rows("slots.csv", copy)
    slots.loc[slots["slot_id"] == "S010", "status"] = "booked"          # booked, but no appointment
    slots.to_csv(copy / "slots.csv", index=False)
    problems = reset_data.consistency_problems(copy)
    check("a booked slot without a confirmed appointment is reported",
          any("S010" in p and "0 confirmed" in p for p in problems), problems)
    appts = rows("appointment.csv", copy)
    appts.loc[len(appts)] = {"appointment_id": "A011", "patient_id": "P001", "doctor_id": "D01", "slot_id": "S011",
                             "reason": "x", "status": "confirmed", "created_at": "2026-10-01 10:00"}
    appts.to_csv(copy / "appointment.csv", index=False)
    problems = reset_data.consistency_problems(copy)
    check("a confirmed appointment whose slot is not booked is reported",
          any("A011" in p and "S011" in p for p in problems), problems)
    appts.loc[len(appts)] = {"appointment_id": "A012", "patient_id": "P001", "doctor_id": "D10", "slot_id": "S073",
                             "reason": "x", "status": "confirmed", "created_at": "2026-10-01 10:00"}
    appts.to_csv(copy / "appointment.csv", index=False)
    check("two confirmed appointments for one slot are reported",
          any("S073" in p and "2 confirmed" in p for p in reset_data.consistency_problems(copy)))


def test_restore():
    print("reset_data.restore_from_seed")
    target = TEST_DATA_DIR / "restore_target"
    target.mkdir()
    (target / "patients.csv").write_text("junk\n")
    (target / "appointment.csv").write_text("junk\n")
    reset_data.restore_from_seed(data_dir=target)
    check("every CSV in the target equals the seed",
          all((target / n).read_bytes() == (SEED / n).read_bytes() for n in reset_data.FILES))

    incomplete = TEST_DATA_DIR / "incomplete_seed"
    shutil.copytree(SEED, incomplete)
    (incomplete / "slots.csv").unlink()
    (target / "patients.csv").write_text("junk\n")
    try:
        reset_data.restore_from_seed(data_dir=target, seed_dir=incomplete)
        refused = False
    except SystemExit:
        refused = True
    check("an incomplete seed is refused and nothing is changed", refused and (target / "patients.csv").read_text() == "junk\n")


def test_sources():
    print("the demo reset and the evaluation use data_seed/")
    check("dashboard_data copies from data_seed/", dashboard_data.SEED_DATA_DIR == SEED)
    check("evaluation copies from data_seed/", evaluate.SEED_DATA_DIR == SEED)
    dashboard_data.reset_demo_data()
    import config
    check("'Reset demo data' gives the seed's files",
          all((config.DATA_DIR / n).read_bytes() == (SEED / n).read_bytes() for n in reset_data.FILES))
    evaluate.reset_workspace()
    check("the evaluation workspace equals the seed",
          all((config.DATA_DIR / n).read_bytes() == (SEED / n).read_bytes() for n in reset_data.FILES))


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    seed_before = fingerprint(SEED)
    test_seed_contents()
    test_consistency_checker()
    test_restore()
    test_sources()
    print("safety")
    check("data/ and data_seed/ are unchanged by these tests",
          fingerprint(ORIGINAL_DATA_DIR) == originals_before and fingerprint(SEED) == seed_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
