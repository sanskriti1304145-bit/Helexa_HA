"""
test_schedule_tool.py - tests for tools/schedule_tool.py (automatic future slots)

Run from the project root:
    python -m tests.test_schedule_tool

Runs on a COPY of the data. ensure_future_slots() is switched off while
APP_NOW is set, so most tests unset APP_NOW and pass a fixed `now` instead.
"""

import inspect
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR, PROJECT_ROOT,
)

use_fresh_test_data()
os.environ.pop("APP_NOW", None)

import config  # noqa: E402
from tools import appointment_tool, disease_search_tool  # noqa: E402
from tools.csv_helpers import read_csv  # noqa: E402
from tools.schedule_tool import ensure_future_slots  # noqa: E402

NOW = datetime(2026, 10, 27, 8, 0)   # a Tuesday morning, before any slot that day


def test_template():
    print("schedule_template.csv")
    template = read_csv(config.SCHEDULE_TEMPLATE_CSV)
    check("columns are doctor_id, weekday, time", list(template.columns) == ["doctor_id", "weekday", "time"])
    check("Dr. Kavita Rao (D10, psychiatry) has NO entries", "D10" not in set(template["doctor_id"]))
    check("every other doctor (D01-D09) has a weekly pattern",
          set(template["doctor_id"]) == {f"D0{i}" for i in range(1, 10)}, set(template["doctor_id"]))
    check("weekdays only (Mon-Fri)", set(template["weekday"]) <= {"Mon", "Tue", "Wed", "Thu", "Fri"})
    check("no duplicate rows", not template.duplicated().any())
    slots = read_csv(ORIGINAL_DATA_DIR / "slots.csv")
    derived = {(r["doctor_id"], datetime.strptime(r["date"], "%Y-%m-%d").strftime("%a"), r["time"])
               for _, r in slots.iterrows() if r["doctor_id"] != "D10"}
    check("derived exactly from the times in slots.csv",
          derived == set(zip(template["doctor_id"], template["weekday"], template["time"])))


def test_skipped_when_clock_frozen():
    print("frozen clock (APP_NOW) -> does nothing")
    before = fingerprint(TEST_DATA_DIR)
    reference_now = json.loads((PROJECT_ROOT / "evaluation" / "test_questions.json").read_text())["reference_now"]
    os.environ["APP_NOW"] = reference_now          # exactly what the evaluation does
    r = ensure_future_slots()
    os.environ.pop("APP_NOW")
    check("with the evaluation's APP_NOW -> status skipped, 0 added",
          r["status"] == "skipped" and r["added"] == 0, r)
    check("  no file changed", fingerprint(TEST_DATA_DIR) == before)


def test_adds_missing_slots():
    print("ensure_future_slots(days_ahead=14)")
    slots_before = read_csv(config.SLOTS_CSV)
    appts_before = fingerprint(TEST_DATA_DIR)["appointment.csv"]
    r = ensure_future_slots(days_ahead=14, now=NOW)
    after = read_csv(config.SLOTS_CSV)
    new = after.iloc[len(slots_before):]
    check("success, slots were added", r["status"] == "success" and r["added"] > 0, r["message"])
    check("  existing 80 rows are exactly unchanged (same order, same values)",
          after.iloc[:len(slots_before)].equals(slots_before))
    check("  appointment.csv is not touched", fingerprint(TEST_DATA_DIR)["appointment.csv"] == appts_before)
    check("  new IDs continue from S081, one after another",
          list(new["slot_id"]) == [f"S{n:03d}" for n in range(81, 81 + len(new))], list(new["slot_id"])[:3])
    check("  all new slots are 'available'", set(new["status"]) == {"available"})

    dates = {datetime.strptime(d, "%Y-%m-%d") for d in new["date"]}
    check("  all inside the 14-day window",
          all(NOW.date() <= d.date() < NOW.date() + timedelta(days=14) for d in dates))
    check("  weekends skipped (no Sat 31 Oct / Sun 1 Nov / Sat 7 Nov / Sun 8 Nov)",
          all(d.weekday() < 5 for d in dates) and not {"2026-10-31", "2026-11-01"} & set(new["date"]))
    check("  psychiatrist D10 gets no new slots", "D10" not in set(new["doctor_id"]))
    check("  no slot in the past",
          all(datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M") > NOW for d, t in zip(new["date"], new["time"])))
    check("  no duplicates (doctor, date, time) in the whole file",
          not after.duplicated(subset=["doctor_id", "date", "time"]).any())
    check("  an existing slot is not added again (D01 Wed 28 Oct 11:30 is S008 only)",
          list(after[(after["doctor_id"] == "D01") & (after["date"] == "2026-10-28")
                     & (after["time"] == "11:30")]["slot_id"]) == ["S008"])
    check("  follows the template: D01 on Mon 2 Nov gets 09:00, 09:30, 10:30",
          sorted(new[(new["doctor_id"] == "D01") & (new["date"] == "2026-11-02")]["time"])
          == ["09:00", "09:30", "10:30"])

    before_second = fingerprint(TEST_DATA_DIR)
    r2 = ensure_future_slots(days_ahead=14, now=NOW)
    check("running it twice -> adds nothing the second time",
          r2["status"] == "success" and r2["added"] == 0 and fingerprint(TEST_DATA_DIR) == before_second, r2)
    check("days_ahead=0 -> adds nothing", ensure_future_slots(days_ahead=0, now=NOW)["added"] == 0)

    psych = read_csv(config.SLOTS_CSV)
    psych = psych[psych["doctor_id"] == "D10"]
    check("psychiatrist still has no available slot at all", set(psych["status"]) == {"booked"})


def test_bookable_after_real_days_pass():
    print("the new slots can be booked")
    os.environ["APP_NOW"] = NOW.strftime("%Y-%m-%d %H:%M")   # book at the same moment
    r = appointment_tool.list_available_slots(specialty="nephrologist")
    upcoming = [s["slot_id"] for s in r["slots"]]
    check("nephrology has new upcoming slots (not only the old S00x ones)",
          r["status"] == "success" and any(int(s[1:]) > 80 for s in upcoming), upcoming[:5])
    booked = appointment_tool.book_appointment("P001", upcoming[-1], "CKD follow-up", confirm=True)
    check("one of them books normally", booked["status"] == "success", booked["message"])
    psych = appointment_tool.list_available_slots(specialty="psychiatrist")
    check("psychiatrist -> still no_slots", psych["status"] == "no_slots")
    os.environ.pop("APP_NOW")


def test_not_called_by_tools():
    print("only the dashboard calls it (not the booking/search tools, not the evaluation)")
    for module in (appointment_tool, disease_search_tool):
        check(f"{module.__name__} does not call ensure_future_slots",
              "ensure_future_slots" not in inspect.getsource(module))
    evaluate_source = (PROJECT_ROOT / "evaluation" / "evaluate.py").read_text()
    check("evaluation/evaluate.py does not call it", "ensure_future_slots" not in evaluate_source)
    app_source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text()
    check("the dashboard calls it at start-up and after 'Reset demo data'",
          app_source.count("ensure_future_slots()") == 2)


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)

    test_template()
    test_skipped_when_clock_frozen()
    test_adds_missing_slots()
    test_bookable_after_real_days_pass()
    test_not_called_by_tools()

    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
