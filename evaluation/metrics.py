"""
metrics.py - turn the tool-call log into performance numbers.

Only READS files, so the dashboard (Phase 9) can import it safely.

Every row in the log is a TOP-LEVEL call: the logger never writes rows for
tools called from inside other tools (see tools/logger.py), so one agent
step = one row and nothing is counted twice.

The two headline metrics:

  Booking success rate
      = bookings actually saved / bookings the user confirmed
      Counted from "confirm_booking" rows (the step that runs after the user
      says "yes"). A row with status "success" means both CSV files were
      updated. "No slots" or a declined booking never reaches this step, so
      they do not count as failures: the agent was right not to book.

  Correct refusals (separate figure)
      = booking requests that ended without a booking for a good reason:
      doctor fully booked, specialty does not exist, requested slot refused
      (e.g. in the past), or the user declined. Counted per reason.

  Answer precision
      = answers the judge graded CORRECT / answers the judge graded
      Counted from "judge_answer" rows, which the evaluation writes to the
      same log (status "correct" or "incorrect"). Rows with status "error"
      (judge reply unreadable) are left out and reported separately.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_log(path):
    """Read a tool-call log CSV into a list of dicts (empty list if missing)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rate(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else None


def booking_success_rate(rows):
    bookings = [r for r in rows if r["tool_name"] == "confirm_booking"]
    saved = sum(r["status"] == "success" for r in bookings)
    return {"rate": _rate(saved, len(bookings)), "saved": saved, "confirmed": len(bookings)}


# Booking outcomes where NOT booking was the right result, and why.
REFUSAL_REASONS = {
    ("prepare_booking", "no_slots"): "doctor fully booked",
    ("prepare_booking", "not_found"): "specialty does not exist",
    ("prepare_booking", "invalid"): "requested slot refused (past, taken or wrong specialty)",
    ("decline_booking", "declined"): "user declined the offer",
}


def booking_refusals(rows):
    """Count booking requests that correctly ended WITHOUT a booking.

    These are reported separately from the booking success rate: they are
    not failures. (Whether each refusal was really correct is checked by
    the evaluation's booking check against the actual files.)
    """
    by_reason = Counter(REFUSAL_REASONS[(r["tool_name"], r["status"])] for r in rows
                        if (r["tool_name"], r["status"]) in REFUSAL_REASONS)
    return {"total": sum(by_reason.values()), "by_reason": dict(by_reason)}


def answer_precision(rows):
    graded = [r for r in rows if r["tool_name"] == "judge_answer" and r["status"] in ("correct", "incorrect")]
    correct = sum(r["status"] == "correct" for r in graded)
    unreadable = sum(r["tool_name"] == "judge_answer" and r["status"] == "error" for r in rows)
    return {"precision": _rate(correct, len(graded)), "correct": correct,
            "graded": len(graded), "judge_errors": unreadable}


def per_tool_summary(rows):
    """For each tool: number of calls, share without errors, average time, statuses."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["tool_name"]].append(r)
    summary = {}
    for tool, calls in sorted(groups.items()):
        ok = sum(c["success"] == "True" for c in calls)
        summary[tool] = {
            "calls": len(calls),
            "no_error_rate": _rate(ok, len(calls)),
            "avg_ms": round(sum(float(c["duration_ms"]) for c in calls) / len(calls), 1),
            "statuses": dict(Counter(c["status"] for c in calls)),
        }
    return summary


def compute_metrics(rows):
    """All metrics in one dict."""
    return {
        "booking_success_rate": booking_success_rate(rows),
        "booking_refusals": booking_refusals(rows),
        "answer_precision": answer_precision(rows),
        "per_tool": per_tool_summary(rows),
        "total_logged_calls": len(rows),
    }
