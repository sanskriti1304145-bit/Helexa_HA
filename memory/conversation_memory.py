"""
conversation_memory.py - short-term (this chat) and long-term (per patient) memory.

SHORT-TERM memory: ShortTermMemory
    Remembers the current conversation so the agent understands follow-ups
    like "book the earliest one for him". Lives only in the computer's
    memory (RAM) and is gone when the program or Streamlit session ends.
    Keeps only the last N messages (config.SHORT_TERM_MEMORY_MESSAGES) so
    prompts never grow too long. Also holds a small "context" dict for facts
    such as which patient we are talking about.

LONG-TERM memory: save_patient_memory / recall_patient_memory
    Notes about a patient that should survive restarts, e.g.
        "Booked A011 with Dr. Meera Shah on 2026-10-21 14:00."
        "Looked up MedlinePlus: Chronic Kidney Disease."
    Saved as one JSON file per patient in memory_store/ (in .gitignore).
    These are the agent's own activity notes, NOT medical records: they never
    change patients.csv or medical_records.csv. The agent only saves a note
    AFTER the user has confirmed the action it describes.
"""

import json
import os

from langchain_core.messages import AIMessage, HumanMessage

import config
from tools.csv_helpers import read_csv
from tools.logger import log_tool_call

NOTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# 1. Short-term memory (one conversation)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """The recent messages of one conversation, plus a small context dict."""

    def __init__(self, max_messages=None):
        self.max_messages = max_messages or config.SHORT_TERM_MEMORY_MESSAGES
        # A list of LangChain message objects: HumanMessage (user) and
        # AIMessage (assistant). These can be put straight into a prompt.
        self.messages = []
        self.context = {}   # e.g. {"patient_id": "P001"}

    def add_user_message(self, text):
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai_message(self, text):
        self.messages.append(AIMessage(content=text))
        self._trim()

    def _trim(self):
        """Forget the oldest messages once we have more than max_messages."""
        self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        """A copy of the messages, ready to put into a prompt."""
        return list(self.messages)

    def as_text(self):
        """The conversation as plain text, e.g. for a prompt or the dashboard."""
        speaker = {"human": "User", "ai": "Assistant"}
        return "\n".join(f"{speaker[m.type]}: {m.text}" for m in self.messages)

    def set_context(self, key, value):
        self.context[key] = value

    def get_context(self, key, default=None):
        return self.context.get(key, default)

    def clear(self):
        self.messages = []
        self.context = {}


# ---------------------------------------------------------------------------
# 2. Long-term memory (per patient, saved to disk)
# ---------------------------------------------------------------------------

def _memory_file(patient_id):
    return config.MEMORY_DIR / f"{patient_id}.json"


def _patient_exists(patient_id):
    return patient_id in read_csv(config.PATIENTS_CSV)["patient_id"].values


def _load_notes(patient_id):
    try:
        return json.loads(_memory_file(patient_id).read_text(encoding="utf-8"))["notes"]
    except FileNotFoundError:
        return []


@log_tool_call
def save_patient_memory(patient_id, text, note_type="note"):
    """Add one note to a patient's long-term memory.

    note_type is a short label such as "booking", "search", "summary", "note".

    Status values: "success", "not_found", "invalid", "error".
    """
    try:
        patient_id = str(patient_id).strip().upper()
        text = " ".join(str(text or "").split())
        note_type = str(note_type or "note").strip().lower()
        if not text:
            return {"status": "invalid", "message": "The note is empty."}
        if not _patient_exists(patient_id):
            return {"status": "not_found", "message": f"No patient with ID {patient_id}."}

        notes = _load_notes(patient_id)
        note = {"timestamp": config.get_now().strftime(NOTE_TIME_FORMAT),
                "type": note_type, "text": text}
        notes.append(note)

        # Write to a temp file, then swap it in (no half-written files).
        config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        path = _memory_file(patient_id)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"patient_id": patient_id, "notes": notes}, indent=2),
                             encoding="utf-8")
        os.replace(temp_path, path)
        return {"status": "success", "message": f"Saved a {note_type} note for {patient_id}.",
                "note": note}
    except Exception as error:
        return {"status": "error", "message": f"Could not save memory: {error}"}


@log_tool_call
def recall_patient_memory(patient_id, limit=5, note_type=None):
    """Return a patient's most recent notes (oldest of those first).

    limit: how many recent notes to return (None = all).
    note_type: only return notes of this type, e.g. "booking".

    Status values: "success", "no_memory", "not_found", "error".
    """
    try:
        patient_id = str(patient_id).strip().upper()
        if not _patient_exists(patient_id):
            return {"status": "not_found", "message": f"No patient with ID {patient_id}.", "notes": []}
        notes = _load_notes(patient_id)
        if note_type:
            notes = [n for n in notes if n["type"] == note_type]
        if limit:
            notes = notes[-int(limit):]
        if not notes:
            return {"status": "no_memory", "message": f"No saved notes for {patient_id}.", "notes": []}
        return {"status": "success", "message": f"Found {len(notes)} note(s) for {patient_id}.",
                "notes": notes}
    except Exception as error:
        return {"status": "error", "message": f"Could not read memory: {error}", "notes": []}
