"""
logger.py - shared logger that records every tool call to a CSV file.

How to use it: put @log_tool_call on the line above a tool function.

    @log_tool_call
    def find_patient(query):
        ...

This is called a "decorator". Every time find_patient runs, the decorator:
    1. starts a timer
    2. runs the real function
    3. writes one row to the log CSV
    4. returns the real function's result, untouched

So the tool behaves exactly as before; it just leaves a record behind.

Only TOP-LEVEL calls are logged. If a logged tool calls another logged tool
internally (list_available_slots calls find_doctors), the inner call runs
normally but writes no row. So one agent step = exactly one log row.

Log columns (one row per call):
    timestamp      when the call happened (real clock)
    tool_name      e.g. "book_appointment"
    status         the "status" the tool returned, e.g. "success", "no_slots"
    success        True if the tool did its job, False if it hit an error
    duration_ms    how long the call took, in milliseconds
    error_message  the error text when success is False, otherwise empty

"success" vs "status":
    success=False only when something went WRONG: the tool returned
    status "error" or crashed. A clear "no" answer such as "no_slots",
    "not_found" or "slot_unavailable" means the tool worked correctly, so
    success=True and the status column tells you what the answer was.
"""

import csv
import functools
import threading
import time
from datetime import datetime

import config

LOG_COLUMNS = ["timestamp", "tool_name", "status", "success", "duration_ms", "error_message"]

# Remembers whether we are already inside a logged tool call.
# threading.local() gives each thread its own copy of this flag. Streamlit
# runs each browser session in its own thread, so one user's tool call
# never stops another user's call from being logged.
_state = threading.local()


def _write_log_row(row):
    """Append one row to the log CSV, adding the header if the file is new.

    If logging itself fails (e.g. the disk is full), we print a warning but
    do NOT stop the tool: a broken log must never break a booking.
    """
    log_path = config.TOOL_LOG_CSV
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as error:
        print(f"WARNING: could not write tool log to {log_path}: {error}")


def _run_and_log(tool_function, args, kwargs):
    """Run the tool, time it, write one log row, and return its result."""
    start = time.perf_counter()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_name": tool_function.__name__,
    }
    try:
        result = tool_function(*args, **kwargs)
    except Exception as error:
        # The tool crashed. Log it, then let the error continue exactly
        # as it would have without the logger.
        row.update(status="exception", success=False,
                   duration_ms=round((time.perf_counter() - start) * 1000, 2),
                   error_message=f"{type(error).__name__}: {error}")
        _write_log_row(row)
        raise

    # Normal case: the tool returned a result.
    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
    success = status != "error"
    message = result.get("message", "") if isinstance(result, dict) else ""
    row.update(status=status, success=success,
               duration_ms=round((time.perf_counter() - start) * 1000, 2),
               error_message="" if success else message)
    _write_log_row(row)
    return result


def log_tool_call(tool_function):
    """Decorator: wrap a tool function so every top-level call is logged."""

    # functools.wraps keeps the original name and docstring, so the wrapped
    # function still looks like "find_patient" to the rest of the code.
    @functools.wraps(tool_function)
    def wrapper(*args, **kwargs):
        # Inner call made by another logged tool: just run it, no log row.
        if getattr(_state, "inside_tool", False):
            return tool_function(*args, **kwargs)

        # Top-level call: mark that we are inside a tool, run and log it.
        _state.inside_tool = True
        try:
            return _run_and_log(tool_function, args, kwargs)
        finally:
            # Always clear the flag, even if the tool crashed, so the next
            # top-level call is logged normally.
            _state.inside_tool = False

    return wrapper


def read_logs():
    """Return all logged tool calls as a list of dicts (empty list if none yet)."""
    log_path = config.TOOL_LOG_CSV
    if not log_path.exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
