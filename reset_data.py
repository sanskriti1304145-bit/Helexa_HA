"""
reset_data.py - put data/ back to the clean baseline stored in data_seed/.

Run from the project root:
    python reset_data.py            # asks "yes/no", then restores data/ from data_seed/
    python reset_data.py --yes      # no question
    python reset_data.py --check    # only print the report for data/ (changes nothing)
    python reset_data.py --check-seed

What it does:
  - copies every CSV in data_seed/ over the same file in data/
  - then prints a report: rows per file and any consistency problems
It does NOT touch vector_store/, memory_store/, logs/ or the demo/test/eval copies.
(The "Reset demo data" button and the evaluation use data_seed/ the same way.)

data_seed/ is the master copy: edit it only on purpose, e.g. when the schema changes.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SEED_DIR = PROJECT_ROOT / "data_seed"
DATA_DIR = PROJECT_ROOT / "data"

FILES = ["patients.csv", "medical_records.csv", "doctors.csv", "slots.csv", "appointment.csv",
         "schedule_template.csv", "waitlist.csv"]


def read(folder, name):
    return pd.read_csv(Path(folder) / name, dtype=str, keep_default_na=False)


def row_counts(folder):
    return {name: len(read(folder, name)) for name in FILES}


def consistency_problems(folder):
    """A list of plain-English problems found in one data folder (empty = all fine)."""
    problems = []
    patients, records = read(folder, "patients.csv"), read(folder, "medical_records.csv")
    doctors, slots = read(folder, "doctors.csv"), read(folder, "slots.csv")
    appts, waitlist = read(folder, "appointment.csv"), read(folder, "waitlist.csv")

    for name, frame, column in [("patients", patients, "patient_id"), ("medical_records", records, "record_id"),
                                ("doctors", doctors, "doctor_id"), ("slots", slots, "slot_id"),
                                ("appointment", appts, "appointment_id"), ("waitlist", waitlist, "waitlist_id")]:
        duplicated = frame[column][frame[column].duplicated()].tolist()
        if duplicated:
            problems.append(f"{name}.csv: duplicate {column}: {duplicated}")

    confirmed = appts[appts["status"] == "confirmed"]
    booked = slots[slots["status"] == "booked"]
    for slot_id in booked["slot_id"]:
        count = int((confirmed["slot_id"] == slot_id).sum())
        if count != 1:
            problems.append(f"slot {slot_id} is booked but has {count} confirmed appointment(s) (needs exactly 1)")
    slot_status = dict(zip(slots["slot_id"], slots["status"]))
    for _, a in confirmed.iterrows():
        if slot_status.get(a["slot_id"]) != "booked":
            problems.append(f"appointment {a['appointment_id']} is confirmed but slot {a['slot_id']} is "
                            f"'{slot_status.get(a['slot_id'], 'missing')}', not booked")
    slot_doctor = dict(zip(slots["slot_id"], slots["doctor_id"]))
    for _, a in appts.iterrows():
        if a["slot_id"] in slot_doctor and slot_doctor[a["slot_id"]] != a["doctor_id"]:
            problems.append(f"appointment {a['appointment_id']}: doctor {a['doctor_id']} differs from the slot's doctor")
    for _, a in appts.iterrows():
        if a["patient_id"] not in set(patients["patient_id"]):
            problems.append(f"appointment {a['appointment_id']}: unknown patient {a['patient_id']}")
    for _, r in records.iterrows():
        if r["patient_id"] not in set(patients["patient_id"]):
            problems.append(f"record {r['record_id']}: unknown patient {r['patient_id']}")
    for _, w in waitlist.iterrows():
        if w["patient_id"] not in set(patients["patient_id"]):
            problems.append(f"waitlist {w['waitlist_id']}: unknown patient {w['patient_id']}")
    if not set(slots["doctor_id"]) <= set(doctors["doctor_id"]):
        problems.append("slots.csv uses a doctor_id that is not in doctors.csv")
    return problems


def print_report(folder):
    folder = Path(folder)
    print(f"\nReport for {folder.relative_to(PROJECT_ROOT)}/")
    for name, count in row_counts(folder).items():
        print(f"  {name:<24}{count:>4} rows")
    slots, appts = read(folder, "slots.csv"), read(folder, "appointment.csv")
    print(f"  slots: {slots['status'].value_counts().to_dict()} · "
          f"appointments: {appts['status'].value_counts().to_dict()}")
    problems = consistency_problems(folder)
    print("  consistency problems: " + ("none" if not problems else ""))
    for problem in problems:
        print(f"    - {problem}")
    return problems


def restore_from_seed(data_dir=DATA_DIR, seed_dir=SEED_DIR):
    """Copy every seed CSV over data_dir. Returns the list of copied file names."""
    seed_dir, data_dir = Path(seed_dir), Path(data_dir)
    missing = [name for name in FILES if not (seed_dir / name).exists()]
    if missing:
        raise SystemExit(f"data_seed is incomplete, nothing was changed. Missing: {missing}")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy(seed_dir / name, data_dir / name)
    return FILES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore data/ from data_seed/.")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--check", action="store_true", help="only report on data/")
    parser.add_argument("--check-seed", action="store_true", help="only report on data_seed/")
    args = parser.parse_args()

    if args.check_seed:
        sys.exit(1 if print_report(SEED_DIR) else 0)
    if args.check:
        sys.exit(1 if print_report(DATA_DIR) else 0)
    if not args.yes and input("Replace the CSVs in data/ with data_seed/? Type yes: ").strip().lower() != "yes":
        raise SystemExit("Cancelled. Nothing was changed.")
    restore_from_seed()
    print("Restored data/ from data_seed/.")
    sys.exit(1 if print_report(DATA_DIR) else 0)
