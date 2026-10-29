"""
test_logger.py - tests for tools/logger.py

Run from the project root:
    python -m tests.test_logger

Checks that the logger records every call correctly AND that wrapping a
function does not change what it returns or how it fails.
"""

import os
import threading
from pathlib import Path

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count,
    ORIGINAL_DATA_DIR, TEST_LOG_CSV, PROJECT_ROOT,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from tools import medical_records_tool, appointment_tool  # noqa: E402
from tools.logger import log_tool_call, read_logs, LOG_COLUMNS  # noqa: E402


def last_log():
    return read_logs()[-1]


# --- Small fake tools, so we control exactly what happens ---

def plain_ok(x):
    """Docstring of plain_ok."""
    return {"status": "success", "message": "fine", "value": x}


@log_tool_call
def fake_ok(x):
    """Docstring of plain_ok."""
    return {"status": "success", "message": "fine", "value": x}


@log_tool_call
def fake_no_answer():
    return {"status": "no_slots", "message": "No available slots."}


@log_tool_call
def fake_error():
    return {"status": "error", "message": "Could not read file"}


@log_tool_call
def fake_crash():
    raise ValueError("something broke")


@log_tool_call
def fake_not_a_dict():
    return 42


@log_tool_call
def fake_outer():
    """A logged tool that calls two other logged tools inside it."""
    inner = fake_ok(1)
    fake_no_answer()
    return {"status": "success", "message": "outer done", "inner": inner}


@log_tool_call
def fake_outer_inner_crashes():
    """A logged tool whose inner logged tool crashes (and is caught)."""
    try:
        fake_crash()
    except ValueError:
        pass
    return {"status": "success", "message": "recovered"}


@log_tool_call
def fake_outer_with_thread():
    """A logged tool that starts a separate thread which calls a tool.
    The thread is a different 'session', so its call is top-level there."""
    worker = threading.Thread(target=fake_ok, args=(9,))
    worker.start()
    worker.join()
    return {"status": "success", "message": "thread done"}


# ---------------------------------------------------------------------------

def test_behaviour_unchanged():
    print("wrapping does not change behaviour")
    check("same result as the unwrapped version", fake_ok(5) == plain_ok(5))
    check("function name kept", fake_ok.__name__ == "fake_ok")
    check("docstring kept", fake_ok.__doc__ == plain_ok.__doc__)
    check("non-dict results returned untouched", fake_not_a_dict() == 42)
    try:
        fake_crash()
        crashed = False
    except ValueError as error:
        crashed = str(error) == "something broke"
    check("an exception is still raised, unchanged", crashed)


def test_log_rows():
    print("log rows")
    fake_ok(1)
    row = last_log()
    check("log file has the expected columns", list(row.keys()) == LOG_COLUMNS, row)
    check("success call -> status success, success True, no error",
          row["tool_name"] == "fake_ok" and row["status"] == "success"
          and row["success"] == "True" and row["error_message"] == "", row)
    check("  duration_ms is a number >= 0", float(row["duration_ms"]) >= 0, row)

    fake_no_answer()
    row = last_log()
    check("clear 'no' answer -> status no_slots but success True",
          row["status"] == "no_slots" and row["success"] == "True", row)

    fake_error()
    row = last_log()
    check("status error -> success False with the error message",
          row["success"] == "False" and row["error_message"] == "Could not read file", row)

    try:
        fake_crash()
    except ValueError:
        pass
    row = last_log()
    check("crash -> status exception, success False, error type + text",
          row["status"] == "exception" and row["success"] == "False"
          and row["error_message"] == "ValueError: something broke", row)

    count = len(read_logs())
    fake_ok(1)
    fake_ok(2)
    check("one row per call", len(read_logs()) == count + 2)


def test_only_top_level_calls_logged():
    print("only top-level calls are logged")
    count = len(read_logs())
    result = fake_outer()
    rows = read_logs()[count:]
    check("outer tool calling 2 inner tools -> exactly 1 row",
          len(rows) == 1 and rows[0]["tool_name"] == "fake_outer", rows)
    check("  inner results still returned normally",
          result["inner"] == plain_ok(1), result)

    count = len(read_logs())
    fake_outer_inner_crashes()
    rows = read_logs()[count:]
    check("inner crash caught by outer -> 1 row for the outer only",
          len(rows) == 1 and rows[0]["tool_name"] == "fake_outer_inner_crashes"
          and rows[0]["status"] == "success", rows)

    count = len(read_logs())
    try:
        fake_crash()
    except ValueError:
        pass
    fake_ok(3)
    rows = read_logs()[count:]
    check("after a top-level crash, the next call is logged again",
          [r["tool_name"] for r in rows] == ["fake_crash", "fake_ok"], rows)

    count = len(read_logs())
    fake_outer_with_thread()
    names = sorted(r["tool_name"] for r in read_logs()[count:])
    check("a call in another thread is still logged (separate sessions)",
          names == ["fake_ok", "fake_outer_with_thread"], names)

    # The real case: list_available_slots(specialty=...) calls find_doctors.
    count = len(read_logs())
    r = appointment_tool.list_available_slots(specialty="nephrologist")
    rows = read_logs()[count:]
    check("list_available_slots(specialty=...) -> exactly 1 row",
          len(rows) == 1 and rows[0]["tool_name"] == "list_available_slots", rows)
    check("  and still returns the nephrology slots",
          r["status"] == "success" and r["slots"][0]["doctor_id"] == "D01", r)

    count = len(read_logs())
    appointment_tool.find_doctors("nephrologist")
    rows = read_logs()[count:]
    check("find_doctors called directly is still logged",
          len(rows) == 1 and rows[0]["tool_name"] == "find_doctors", rows)


def test_broken_log_does_not_break_tool():
    print("a broken log never breaks a tool")
    real_path = config.TOOL_LOG_CSV
    config.TOOL_LOG_CSV = TEST_LOG_CSV.parent   # a folder, so writing to it fails
    result = fake_ok(7)
    config.TOOL_LOG_CSV = real_path
    check("tool still returns its normal result", result == plain_ok(7), result)


def test_real_tools_are_logged():
    print("real tools")
    tool_functions = [
        medical_records_tool.find_patient, medical_records_tool.get_patient_history,
        medical_records_tool.add_record, medical_records_tool.update_record,
        medical_records_tool.add_patient,
        appointment_tool.find_doctors, appointment_tool.list_available_slots,
        appointment_tool.book_appointment, appointment_tool.cancel_appointment,
        appointment_tool.list_appointments,
    ]
    check("all 10 tool functions are wrapped",
          all(hasattr(f, "__wrapped__") for f in tool_functions))

    medical_records_tool.find_patient("P001")
    check("find_patient call logged", last_log()["tool_name"] == "find_patient")

    appointment_tool.book_appointment("P001", "S004", "CKD review")
    row = last_log()
    check("booking preview logged as needs_confirmation",
          row["tool_name"] == "book_appointment" and row["status"] == "needs_confirmation", row)

    appointment_tool.book_appointment("P001", "S004", "CKD review", confirm=True)
    check("confirmed booking logged as success", last_log()["status"] == "success")

    appointment_tool.book_appointment("P002", "S073", "Anxiety", confirm=True)
    row = last_log()
    check("booking a taken slot logged as slot_unavailable (success True)",
          row["status"] == "slot_unavailable" and row["success"] == "True", row)


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    real_log = PROJECT_ROOT / "logs" / "tool_calls.csv"
    real_log_before = real_log.read_bytes() if real_log.exists() else None

    test_behaviour_unchanged()
    test_log_rows()
    test_only_top_level_calls_logged()
    test_broken_log_does_not_break_tool()
    test_real_tools_are_logged()

    print("safety")
    check("tests logged to test_data/logs/, not logs/",
          Path(config.TOOL_LOG_CSV) == TEST_LOG_CSV and TEST_LOG_CSV.exists())
    check("real logs/tool_calls.csv untouched",
          (real_log.read_bytes() if real_log.exists() else None) == real_log_before)
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed.")
