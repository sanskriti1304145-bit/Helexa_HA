"""
schedule_tool.py - automatically create future appointment slots.

The problem: slots.csv only has slots for a fixed set of dates. As real days
pass, they all end up in the past and nothing can be booked any more.

The fix: data/schedule_template.csv describes each doctor's WEEKLY pattern,
one row per (doctor_id, weekday, time), e.g. "D01,Mon,09:00". It was derived
from the times already used in slots.csv. Dr. Kavita Rao (D10, psychiatry)
has no rows on purpose, so she stays fully booked.

ensure_future_slots(days_ahead=14) looks at the next 14 days and, for every
weekday (Mon-Fri) in that window, adds each template slot that is missing.

Safety rules:
  - It only ADDS rows. Existing slots and appointments are never changed or deleted.
  - Running it twice adds nothing the second time (no duplicates): a slot is
    "missing" only if no row exists yet for that doctor, date and time.
  - New slot IDs continue from the highest existing one (S081, S082, ...).
  - It is NOT called by the booking or search tools. The dashboard calls it
    when it starts and after "Reset demo data".
  - When the clock is frozen with APP_NOW (tests, the Phase 8 evaluation) it
    does nothing, so those results stay the same every time.
"""

import os
from datetime import datetime, timedelta

import pandas as pd

import config
from tools.csv_helpers import read_csv, write_csv
from tools.logger import log_tool_call

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]   # Saturday and Sunday are skipped


@log_tool_call
def ensure_future_slots(days_ahead=14, now=None):
    """Add any missing available slots for the next `days_ahead` days.

    now: leave as None to use the real clock. Tests pass a fixed datetime
         here (they cannot use APP_NOW, because APP_NOW switches this off).

    Status values: "success", "skipped", "error".
    """
    if now is None:
        if os.getenv("APP_NOW"):
            return {"status": "skipped", "added": 0,
                    "message": "The clock is frozen (APP_NOW), so no slots were added. "
                               "This keeps tests and the evaluation repeatable."}
        now = datetime.now()

    try:
        template = read_csv(config.SCHEDULE_TEMPLATE_CSV)
        slots = read_csv(config.SLOTS_CSV)
        known_doctors = set(read_csv(config.DOCTORS_CSV)["doctor_id"])

        # Every (doctor, date, time) that already has a row, whatever its status.
        existing = set(zip(slots["doctor_id"], slots["date"], slots["time"]))
        next_number = max((int(s[1:]) for s in slots["slot_id"] if s[1:].isdigit()), default=0) + 1

        new_rows = []
        for offset in range(int(days_ahead)):
            day = now.date() + timedelta(days=offset)
            weekday = day.strftime("%a")
            if weekday not in WEEKDAYS:
                continue                                   # weekend
            date_text = day.strftime("%Y-%m-%d")
            for _, row in template[template["weekday"] == weekday].iterrows():
                key = (row["doctor_id"], date_text, row["time"])
                if row["doctor_id"] not in known_doctors or key in existing:
                    continue                               # unknown doctor, or already there
                if datetime.strptime(f"{date_text} {row['time']}", config.NOW_FORMAT) <= now:
                    continue                               # earlier today: already past
                new_rows.append({"slot_id": f"S{next_number:03d}", "doctor_id": row["doctor_id"],
                                 "date": date_text, "time": row["time"], "status": "available"})
                existing.add(key)
                next_number += 1

        if new_rows:
            # Append at the end; the existing rows are written back unchanged.
            write_csv(pd.concat([slots, pd.DataFrame(new_rows)], ignore_index=True), config.SLOTS_CSV)
        return {"status": "success", "added": len(new_rows),
                "slot_ids": [r["slot_id"] for r in new_rows],
                "message": f"Added {len(new_rows)} new slot(s) for the next {days_ahead} day(s)."}
    except Exception as error:
        return {"status": "error", "added": 0, "message": f"Could not create future slots: {error}"}
