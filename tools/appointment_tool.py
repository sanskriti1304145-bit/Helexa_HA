"""
appointment_tool.py - find doctors, list open slots, book and cancel appointments.

Functions in this file:
    find_doctors(specialty)                          -> doctors for a specialty
    list_available_slots(doctor_id=, specialty=)     -> open, future slots
    book_appointment(patient_id, slot_id, reason, confirm)
    cancel_appointment(appointment_id, confirm)
    list_appointments(patient_id=, doctor_id=, status=)  -> appointment tracking

Every function returns a dict with "status" and "message".

Two files must always agree with each other:
    slots.csv        slot status "available" or "booked"
    appointment.csv  appointment status "confirmed" or "cancelled"
A confirmed appointment means its slot is "booked". A cancelled appointment
frees its slot again ("available").

So booking and cancelling each change BOTH files. To keep them in step:
    1. All checks run first, and both new tables are built in memory.
    2. Then the first file is written, then the second.
    3. If the second write fails, the first file is put back the way it was
       (a "rollback"), so the two files never disagree.

Like the medical records tool, booking and cancelling write nothing unless
called with confirm=True. Without it they return "needs_confirmation" and
a preview of what would be saved.
"""

from datetime import datetime

import config
from tools.csv_helpers import read_csv, write_csv, next_id
from tools.logger import log_tool_call

# Everyday words for doctors -> the specialty name used in doctors.csv.
SPECIALIST_WORDS = {
    "nephrologist": "Nephrology",
    "cardiologist": "Cardiology",
    "general physician": "General Medicine",
    "gp": "General Medicine",
    "endocrinologist": "Endocrinology",
    "neurologist": "Neurology",
    "dermatologist": "Dermatology",
    "orthopedist": "Orthopedics",
    "orthopedic surgeon": "Orthopedics",
    "pulmonologist": "Pulmonology",
    "gastroenterologist": "Gastroenterology",
    "psychiatrist": "Psychiatry",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clean(text):
    """Lower-case a string and squash extra spaces."""
    return " ".join(str(text).lower().split())


def _slot_datetime(slot):
    """Turn a slot's date and time text into one datetime object."""
    return datetime.strptime(f"{slot['date']} {slot['time']}", config.NOW_FORMAT)


def _is_in_past(slot):
    """True if the slot's date and time is not after "now"."""
    return _slot_datetime(slot) <= config.get_now()


def _write_two_files(first_path, first_new, first_original, second_path, second_new):
    """Write two CSV files as one unit: both change, or neither does.

    first_original is the first file's content BEFORE our change. If writing
    the second file fails, we write first_original back (rollback).
    """
    # If this first write fails, write_csv leaves the old file untouched,
    # so nothing has changed and we simply let the error go up.
    write_csv(first_new, first_path)
    try:
        write_csv(second_new, second_path)
    except Exception as second_error:
        try:
            write_csv(first_original, first_path)   # undo the first change
        except Exception as rollback_error:
            raise RuntimeError(
                f"Saving {second_path.name} failed ({second_error}) and undoing "
                f"{first_path.name} also failed ({rollback_error}). "
                "The files may now disagree; please check them."
            )
        raise RuntimeError(
            f"Saving {second_path.name} failed ({second_error}). "
            f"The change to {first_path.name} was undone, so nothing was changed."
        )


# ---------------------------------------------------------------------------
# 1. Find doctors
# ---------------------------------------------------------------------------

@log_tool_call
def find_doctors(specialty):
    """Find doctors by specialty, e.g. "Nephrology", "nephrologist" or "cardio".

    Status values: "success", "not_found", "invalid", "error".
    """
    try:
        query = _clean(specialty or "")
        if not query:
            return {"status": "invalid", "message": "Please give a specialty."}

        # "nephrologist" -> "nephrology"
        query = _clean(SPECIALIST_WORDS.get(query, query))

        doctors = read_csv(config.DOCTORS_CSV)
        matches = doctors[doctors["specialty"].apply(lambda s: query in _clean(s))]

        if matches.empty:
            known = ", ".join(sorted(doctors["specialty"].unique()))
            return {"status": "not_found",
                    "message": f"No doctors found for '{specialty}'. "
                               f"Available specialties: {known}.",
                    "doctors": []}

        names = ", ".join(f"{d['name']} ({d['specialty']})" for _, d in matches.iterrows())
        return {"status": "success",
                "message": f"Found {len(matches)} doctor(s): {names}.",
                "doctors": matches.to_dict(orient="records")}

    except Exception as error:
        return {"status": "error", "message": f"Could not search doctors: {error}"}


# ---------------------------------------------------------------------------
# 2. List open slots
# ---------------------------------------------------------------------------

def no_openings_message(doctors):
    """One sentence for the given doctors: "Dr. A (Cardiology) has no openings in the next 2 weeks." """
    people = [f"{d['name']} ({d['specialty']})" for _, d in doctors.iterrows()]
    who = people[0] if len(people) == 1 else ", ".join(people[:-1]) + " and " + people[-1]
    return f"{who} {'has' if len(people) == 1 else 'have'} no openings in the next 2 weeks."


@log_tool_call
def list_available_slots(doctor_id=None, specialty=None):
    """List slots that are "available" AND in the future, soonest first.

    Give either doctor_id (e.g. "D01") or specialty (e.g. "nephrologist").

    Status values: "success", "no_slots" (doctor exists but has no open
                   slots), "not_found", "invalid", "error".
    """
    try:
        doctors = read_csv(config.DOCTORS_CSV)

        # Work out which doctors we are looking at.
        if doctor_id:
            doctor_id = str(doctor_id).strip().upper()
            chosen = doctors[doctors["doctor_id"] == doctor_id]
            if chosen.empty:
                return {"status": "not_found", "message": f"No doctor with ID {doctor_id}."}
        elif specialty:
            found = find_doctors(specialty)
            if found["status"] != "success":
                return found
            chosen = doctors[doctors["doctor_id"].isin([d["doctor_id"] for d in found["doctors"]])]
        else:
            return {"status": "invalid", "message": "Give a doctor_id or a specialty."}

        slots = read_csv(config.SLOTS_CSV)
        slots = slots[slots["doctor_id"].isin(chosen["doctor_id"])]
        slots = slots[slots["status"] == "available"]
        slots = slots[~slots.apply(_is_in_past, axis=1)] if not slots.empty else slots

        doctor_names = ", ".join(chosen["name"])
        if slots.empty:
            # Only when there is NOT ONE future open slot, so "no openings in the
            # next 2 weeks" is always true (we never hide slots further away).
            return {"status": "no_slots", "message": no_openings_message(chosen), "slots": []}

        # Add the doctor's name and clinic to each slot, then sort by date/time.
        slots = slots.merge(chosen[["doctor_id", "name", "specialty", "clinic"]], on="doctor_id")
        slots = slots.rename(columns={"name": "doctor_name"})
        slots = slots.sort_values(["date", "time"])

        return {"status": "success",
                "message": f"Found {len(slots)} available slot(s) for {doctor_names}.",
                "slots": slots.to_dict(orient="records")}

    except Exception as error:
        return {"status": "error", "message": f"Could not list slots: {error}"}


# ---------------------------------------------------------------------------
# 3. Book an appointment
# ---------------------------------------------------------------------------

@log_tool_call
def book_appointment(patient_id, slot_id, reason, confirm=False):
    """Book a slot for a patient.

    Every call (including the confirm=True call) re-reads both files and
    re-checks that the slot is still available, so a slot taken by someone
    else between preview and confirmation is never double-booked.

    Status values: "needs_confirmation", "success", "not_found", "invalid",
                   "slot_unavailable", "slot_in_past", "conflict", "error".
    """
    try:
        patient_id = str(patient_id).strip().upper()
        slot_id = str(slot_id).strip().upper()
        reason = str(reason or "").strip()
        if not reason:
            return {"status": "invalid", "message": "Please give a reason for the visit."}

        # --- Read fresh copies of all the files we need ---
        patients = read_csv(config.PATIENTS_CSV)
        doctors = read_csv(config.DOCTORS_CSV)
        slots = read_csv(config.SLOTS_CSV)
        appointments = read_csv(config.APPOINTMENTS_CSV)

        # --- Validate everything before writing anything ---
        if patient_id not in patients["patient_id"].values:
            return {"status": "not_found", "message": f"No patient with ID {patient_id}."}

        slot_rows = slots.index[slots["slot_id"] == slot_id]
        if len(slot_rows) == 0:
            return {"status": "not_found", "message": f"No slot with ID {slot_id}."}
        slot_row = slot_rows[0]
        slot = slots.loc[slot_row]

        if slot["status"] != "available":
            return {"status": "slot_unavailable",
                    "message": f"Slot {slot_id} ({slot['date']} {slot['time']}) is already booked."}
        if _is_in_past(slot):
            return {"status": "slot_in_past",
                    "message": f"Slot {slot_id} ({slot['date']} {slot['time']}) is in the past."}

        doctor_rows = doctors[doctors["doctor_id"] == slot["doctor_id"]]
        if doctor_rows.empty:
            return {"status": "error", "message": f"Slot {slot_id} has an unknown doctor."}
        doctor = doctor_rows.iloc[0]

        # Extra safety: no other confirmed appointment may use this slot.
        confirmed = appointments[appointments["status"] == "confirmed"]
        if slot_id in confirmed["slot_id"].values:
            return {"status": "slot_unavailable",
                    "message": f"Slot {slot_id} already has a confirmed appointment."}

        # The patient must not already have an appointment at the same date and time.
        patient_slots = slots[slots["slot_id"].isin(
            confirmed[confirmed["patient_id"] == patient_id]["slot_id"])]
        clash = patient_slots[(patient_slots["date"] == slot["date"]) &
                              (patient_slots["time"] == slot["time"])]
        if not clash.empty:
            return {"status": "conflict",
                    "message": f"{patient_id} already has an appointment on "
                               f"{slot['date']} at {slot['time']}."}

        new_appointment = {
            "appointment_id": next_id(appointments["appointment_id"], "A"),
            "patient_id": patient_id,
            "doctor_id": slot["doctor_id"],
            "slot_id": slot_id,
            "reason": reason,
            "status": "confirmed",
            "created_at": config.get_now().strftime(config.NOW_FORMAT),
        }
        details = (f"{new_appointment['appointment_id']}: {patient_id} with "
                   f"{doctor['name']} ({doctor['specialty']}, {doctor['clinic']}) on "
                   f"{slot['date']} at {slot['time']}")

        if not confirm:
            return {"status": "needs_confirmation",
                    "message": f"About to book {details}. Please confirm.",
                    "will_write": {"appointment": new_appointment,
                                   "slot": {"slot_id": slot_id, "status": "available -> booked"}}}

        # --- Build both new tables in memory ---
        new_slots = slots.copy()
        new_slots.at[slot_row, "status"] = "booked"
        new_appointments = appointments.copy()
        new_appointments.loc[len(new_appointments)] = new_appointment

        # --- Write: slot first (claim it), then the appointment ---
        _write_two_files(config.SLOTS_CSV, new_slots, slots,
                         config.APPOINTMENTS_CSV, new_appointments)

        return {"status": "success",
                "message": f"Booked {details}.",
                "appointment": new_appointment}

    except Exception as error:
        return {"status": "error", "message": f"Could not book appointment: {error}"}


# ---------------------------------------------------------------------------
# 4. Cancel an appointment
# ---------------------------------------------------------------------------

@log_tool_call
def cancel_appointment(appointment_id, confirm=False):
    """Cancel a confirmed, upcoming appointment and free its slot.

    Cancelling a missing, already-cancelled or past appointment returns an
    error and changes nothing.

    Status values: "needs_confirmation", "success", "not_found",
                   "already_cancelled", "invalid", "error".
    """
    try:
        appointment_id = str(appointment_id).strip().upper()
        slots = read_csv(config.SLOTS_CSV)
        appointments = read_csv(config.APPOINTMENTS_CSV)

        # --- Validate everything before writing anything ---
        appt_rows = appointments.index[appointments["appointment_id"] == appointment_id]
        if len(appt_rows) == 0:
            return {"status": "not_found",
                    "message": f"No appointment with ID {appointment_id}. Nothing was changed."}
        appt_row = appt_rows[0]
        appointment = appointments.loc[appt_row]

        if appointment["status"] == "cancelled":
            return {"status": "already_cancelled",
                    "message": f"Appointment {appointment_id} is already cancelled. "
                               "Nothing was changed."}

        slot_rows = slots.index[slots["slot_id"] == appointment["slot_id"]]
        if len(slot_rows) == 0:
            return {"status": "error",
                    "message": f"Appointment {appointment_id} points to unknown slot "
                               f"{appointment['slot_id']}. Nothing was changed."}
        slot_row = slot_rows[0]
        slot = slots.loc[slot_row]

        if _is_in_past(slot):
            return {"status": "invalid",
                    "message": f"Appointment {appointment_id} was on {slot['date']} at "
                               f"{slot['time']}, which has already passed. Nothing was changed."}

        details = (f"{appointment_id} ({appointment['patient_id']} with "
                   f"{appointment['doctor_id']} on {slot['date']} at {slot['time']})")

        if not confirm:
            return {"status": "needs_confirmation",
                    "message": f"About to cancel {details}. Please confirm.",
                    "will_write": {"appointment": {"appointment_id": appointment_id,
                                                   "status": "confirmed -> cancelled"},
                                   "slot": {"slot_id": slot["slot_id"],
                                            "status": f"{slot['status']} -> available"}}}

        # --- Build both new tables in memory ---
        new_appointments = appointments.copy()
        new_appointments.at[appt_row, "status"] = "cancelled"
        new_slots = slots.copy()
        new_slots.at[slot_row, "status"] = "available"

        # --- Write: appointment first, then free the slot ---
        _write_two_files(config.APPOINTMENTS_CSV, new_appointments, appointments,
                         config.SLOTS_CSV, new_slots)

        return {"status": "success", "message": f"Cancelled {details}. The slot is free again."}

    except Exception as error:
        return {"status": "error", "message": f"Could not cancel appointment: {error}"}


# ---------------------------------------------------------------------------
# 5. List appointments (for tracking and the dashboard)
# ---------------------------------------------------------------------------

@log_tool_call
def list_appointments(patient_id=None, doctor_id=None, status=None):
    """List appointments with their date, time and doctor name, soonest first.

    All filters are optional, e.g. list_appointments(patient_id="P001").

    Status values: "success", "no_appointments", "error".
    """
    try:
        appointments = read_csv(config.APPOINTMENTS_CSV)
        slots = read_csv(config.SLOTS_CSV)[["slot_id", "date", "time"]]
        doctors = read_csv(config.DOCTORS_CSV)[["doctor_id", "name", "specialty"]]

        if patient_id:
            appointments = appointments[appointments["patient_id"] == str(patient_id).strip().upper()]
        if doctor_id:
            appointments = appointments[appointments["doctor_id"] == str(doctor_id).strip().upper()]
        if status:
            appointments = appointments[appointments["status"] == str(status).strip().lower()]

        if appointments.empty:
            return {"status": "no_appointments", "message": "No matching appointments.",
                    "appointments": []}

        # Add date/time (from Slots) and doctor name (from Doctors).
        table = appointments.merge(slots, on="slot_id", how="left")
        table = table.merge(doctors, on="doctor_id", how="left")
        table = table.rename(columns={"name": "doctor_name"}).sort_values(["date", "time"])

        return {"status": "success",
                "message": f"Found {len(table)} appointment(s).",
                "appointments": table.to_dict(orient="records")}

    except Exception as error:
        return {"status": "error", "message": f"Could not list appointments: {error}"}
