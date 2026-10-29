"""
dashboard_data.py - read-only helpers that load tables for the dashboard.

Why a separate file?
  - The dashboard only needs to READ files to show them. These helpers read
    the CSV/JSON files directly instead of calling the logged tools, so just
    looking at a page never adds rows to the tool-call log.
  - They contain no Streamlit code, so they are easy to test on their own.

The only functions that WRITE are setup_demo_workspace() and reset_demo_data(),
and they only touch the demo copy (never data/).
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from evaluation.metrics import compute_metrics, read_log
from tools.csv_helpers import add_missing_columns, read_csv

PROJECT_ROOT = config.PROJECT_ROOT
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data"        # the real folder (the dashboard never writes here)
SEED_DATA_DIR = PROJECT_ROOT / "data_seed"       # clean baseline: the demo copy and its reset start from here
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "test_questions.json"


# ---------------------------------------------------------------------------
# Demo workspace
# ---------------------------------------------------------------------------

def using_original_data():
    return config.DATA_DIR.resolve() == ORIGINAL_DATA_DIR.resolve()


def setup_demo_workspace():
    """First run: copy the CSVs (from data_seed/) and the patient store into the demo folders."""
    if using_original_data():
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for csv_file in SEED_DATA_DIR.glob("*.csv"):
        target = config.DATA_DIR / csv_file.name
        if not target.exists():
            shutil.copy(csv_file, target)
    upgrade_demo_files()
    real_store = PROJECT_ROOT / "vector_store"
    if not config.VECTOR_STORE_DIR.exists() and real_store.exists():
        shutil.copytree(real_store, config.VECTOR_STORE_DIR)


def upgrade_demo_files():
    """An older demo copy may lack newer columns: add them blank (nothing else changes)."""
    for path, columns in [(config.PATIENTS_CSV, {"contact_phone": "contact_person"}),
                          (config.WAITLIST_CSV, {"recheck_on": "created_at"})]:
        if path.exists():
            add_missing_columns(path, columns)


def reset_demo_data():
    """Put the demo CSVs back to the clean baseline (data_seed/) and clear long-term memory.
    Refuses to run if the app is pointed at the original data/ folder."""
    if using_original_data():
        return False
    for csv_file in SEED_DATA_DIR.glob("*.csv"):
        shutil.copy(csv_file, config.DATA_DIR / csv_file.name)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Patients, doctors, slots, appointments
# ---------------------------------------------------------------------------

def patients():
    return read_csv(config.PATIENTS_CSV)


def doctors():
    return read_csv(config.DOCTORS_CSV)


def patient_records(patient_id):
    records = read_csv(config.MEDICAL_RECORDS_CSV)
    return records[records["patient_id"] == patient_id].sort_values("visit_date")


def patient_summary(patient_id):
    """The stored LLM summary from the FAISS store's JSON file (or None)."""
    path = config.VECTOR_STORE_DIR / "patient_summaries.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(patient_id)


def patient_notes(patient_id):
    """Long-term memory notes for one patient (list, newest last)."""
    path = config.MEMORY_DIR / f"{patient_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["notes"]


def slots_with_state():
    """All slots with a readable state: available, booked, or past (unbookable)."""
    slots = read_csv(config.SLOTS_CSV).merge(
        doctors()[["doctor_id", "name", "specialty"]], on="doctor_id")
    now = config.get_now()
    when = pd.to_datetime(slots["date"] + " " + slots["time"])
    slots["state"] = slots["status"]
    slots.loc[(slots["status"] == "available") & (when <= now), "state"] = "past"
    return slots.rename(columns={"name": "doctor_name"}).sort_values(["date", "time"])


def appointments_table():
    """Every appointment with date, time, doctor and patient names, soonest first."""
    appts = read_csv(config.APPOINTMENTS_CSV)
    if appts.empty:
        return appts
    table = (appts.merge(read_csv(config.SLOTS_CSV)[["slot_id", "date", "time"]], on="slot_id", how="left")
                  .merge(doctors()[["doctor_id", "name", "specialty"]], on="doctor_id", how="left")
                  .rename(columns={"name": "doctor_name"})
                  .merge(patients()[["patient_id", "name"]], on="patient_id", how="left")
                  .rename(columns={"name": "patient_name"}))
    now = config.get_now()
    when = pd.to_datetime(table["date"] + " " + table["time"])
    table["when"] = "upcoming"
    table.loc[when <= now, "when"] = "past"
    columns = ["appointment_id", "date", "time", "when", "status", "patient_id", "patient_name",
               "doctor_name", "specialty", "slot_id", "reason", "created_at"]
    return table[columns].sort_values(["date", "time"])


def waitlist_table():
    """Waiting-list entries with the patient's and doctor's names, oldest first."""
    entries = read_csv(config.WAITLIST_CSV) if config.WAITLIST_CSV.exists() else pd.DataFrame(
        columns=["waitlist_id", "patient_id", "specialty", "doctor_id", "reason", "status", "created_at",
                 "recheck_on"])
    table = (entries.merge(patients()[["patient_id", "name"]], on="patient_id", how="left")
                    .rename(columns={"name": "patient_name"})
                    .merge(doctors()[["doctor_id", "name"]], on="doctor_id", how="left")
                    .rename(columns={"name": "doctor_name"}))
    table["doctor_name"] = table["doctor_name"].fillna("any doctor")
    if "recheck_on" not in table.columns:          # an older file
        table["recheck_on"] = ""
    columns = ["waitlist_id", "created_at", "recheck_on", "status", "patient_id", "patient_name",
               "specialty", "doctor_name", "reason"]
    return table[columns].sort_values("created_at", kind="stable")


def due_rechecks():
    """Waiting-list entries still waiting whose recheck_on date is today or earlier.

    Each item: {"waitlist_id", "patient_name", "specialty", "doctor_id", "doctor_label", "recheck_on"}.
    doctor_label is the doctor's name, or "<Specialty> doctors" when any doctor will do.
    """
    table = waitlist_table()
    if table.empty:
        return []
    today = config.get_now().strftime("%Y-%m-%d")
    due = table[(table["status"] == "waiting") & (table["recheck_on"] != "") & (table["recheck_on"] <= today)]
    people = doctors().set_index("doctor_id")["name"].to_dict()
    raw = read_csv(config.WAITLIST_CSV).set_index("waitlist_id")["doctor_id"].to_dict()
    items = []
    for _, row in due.iterrows():
        doctor_id = raw.get(row["waitlist_id"], "")
        items.append({"waitlist_id": row["waitlist_id"], "patient_name": row["patient_name"],
                      "specialty": row["specialty"], "doctor_id": doctor_id,
                      "doctor_label": people.get(doctor_id) or f"{row['specialty']} doctors",
                      "recheck_on": row["recheck_on"]})
    return items


# ---------------------------------------------------------------------------
# Medical information (MedlinePlus cache)
# ---------------------------------------------------------------------------

def recent_searches(limit=10):
    """Recent MedlinePlus searches from the 24-hour cache, newest first."""
    folder = config.CACHE_DIR / "medlineplus"
    items = []
    for path in folder.glob("*.json") if folder.exists() else []:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        items.append({"saved_at": saved["saved_at"], "query": saved["query"],
                      "status": saved["result"]["status"],
                      "results": saved["result"].get("results", [])})
    return sorted(items, key=lambda i: i["saved_at"], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Logs and evaluation
# ---------------------------------------------------------------------------

def log_table():
    """The app's tool-call log, newest first."""
    rows = read_log(config.TOOL_LOG_CSV)
    return pd.DataFrame(rows[::-1]) if rows else pd.DataFrame(
        columns=["timestamp", "tool_name", "status", "success", "duration_ms", "error_message"])


def live_metrics():
    """Metrics computed from the app's own log (the same code as the evaluation)."""
    return compute_metrics(read_log(config.TOOL_LOG_CSV))


def latest_evaluation():
    """(summary dict, results DataFrame) of the latest evaluation run, or (None, None)."""
    summary_file, results_file = RESULTS_DIR / "latest_summary.json", RESULTS_DIR / "latest_results.csv"
    if not (summary_file.exists() and results_file.exists()):
        return None, None
    return (json.loads(summary_file.read_text(encoding="utf-8")),
            pd.read_csv(results_file, dtype=str, keep_default_na=False))


def sample_scenarios():
    """The first message of each evaluation question, as ready-made scenarios."""
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return [q["turns"][0] for q in data["questions"]]


def format_time(text):
    """'2026-10-21 14:09:42' -> '21 Oct 14:09' for compact tables."""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return text
