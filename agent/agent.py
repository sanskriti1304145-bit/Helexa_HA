"""
agent.py - runs the plan step by step, calling tools and the LLM.

How a conversation works (sample scenario):

  Turn 1  User: "My 70-year-old father has chronic kidney disease. I want to book a
                 nephrologist for him. Also, can you summarize latest treatment methods?"
          Agent: makes a plan  -> 1. identify_patient  2. get_patient_history
                                  3. book_appointment  4. search_disease_info
                 runs step 1     -> finds Ramesh Kumar (70, male) and PAUSES:
                 "Is the patient Ramesh Kumar (P001, age 70, male)?"
  Turn 2  User: "yes"
          Agent: step 2 -> shows his history summary
                 step 3 -> finds the earliest nephrology slot and PAUSES:
                 "Shall I book Dr. Meera Shah on ... ? (yes / slot ID / no)"
  Turn 3  User: "yes"
          Agent: books it (writes appointment.csv + slots.csv), saves a memory note,
                 step 4 -> MedlinePlus RAG summary with sources, then the disclaimer.

Every step is one logged function, so each step writes exactly one log row.
"Yes"/"no" replies are read with simple rules (not the LLM), because they
decide whether files are written.
"""

import re
from datetime import datetime

import config
from agent.disease_rag import summarize_disease_info
from agent.patient_identifier import ASK_NAME_QUESTION, identify_patient
from agent.planner import extract_patient_facts, make_plan
from agent.prompts import BOOKING_CONFIRMATION_PROMPT
from memory.conversation_memory import ShortTermMemory, save_patient_memory
from memory.vector_store import (get_patient_summary, records_fingerprint, refresh_patient_summary,
                                 summarize_patient)
from tools.appointment_tool import (SPECIALIST_WORDS, book_appointment, find_doctors, list_available_slots,
                                    no_openings_message)
from tools.csv_helpers import read_csv
from tools.logger import log_tool_call
from tools.medical_records_tool import add_patient, clean_phone, get_patient_history
from tools.waitlist_tool import add_to_waitlist

YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "confirm",
             "confirmed", "right", "proceed"}
NO_WORDS = {"no", "n", "nope", "not", "wrong", "dont", "don't", "cancel", "stop", "skip", "incorrect"}
STOP_WORDS = {"cancel", "stop", "quit", "exit", "abort"}

# Words that show a reply is a request or a sentence, not a person's name.
REQUEST_WORDS = {"book", "show", "create", "make", "add", "find", "search", "what", "how", "why",
                 "when", "can", "could", "would", "please", "summarize", "summary", "tell", "history",
                 "appointment", "appointments", "log", "logs", "new", "register", "list", "help",
                 "need", "want", "my", "the", "for", "is", "are", "i", "a", "an", "of", "with",
                 "doctor", "treatment", "has", "have", "about", "slot"}

# Greetings and small talk ("Hi", "good morning", "thanks", "what can you do?").
# A message made ONLY of these words gets the friendly capabilities reply,
# without calling the LLM or making a plan.
SMALL_TALK_WORDS = {"hi", "hello", "hey", "hiya", "howdy", "good", "morning", "afternoon", "evening",
                    "greetings", "there", "everyone", "again", "thanks", "thank", "you", "thx", "ok",
                    "okay", "help", "me", "menu", "start", "what", "can", "do", "who", "are", "how",
                    "assistant", "bot", "please", "is", "this", "your", "name"}

# SCOPE GUARD: fixed reply templates (never written by the LLM) for everything
# that is not an in-scope request. The option bubbles under each reply come from MENU.
GREETING_REPLY = "Hello! I'm Helexa. What would you like to do?"
OFF_TOPIC_REPLY = ("Sorry, that's outside what I can help with. I can help with patient records, "
                   "appointments and disease information.")
ADVICE_REPLY = ("I can't give personal medical advice. I can show general information from "
                "MedlinePlus, or help with an appointment.")
UNCLEAR_REPLY = "I didn't catch that. What would you like to do?"
CANCELLED_REPLY = "I've cancelled this request."
SORRY_REPLY = "I'm sorry to hear that."
PICK_ONE_REPLY = "Pick one to continue"                       # used instead of repeating a message
PICK_ONE_AGAIN = "Choose an option, or type your request."    # if even that would repeat
STILL_WAITING = "I'm still waiting for your answer above."    # a repeat when there are no bubbles

# The option bubbles: (label shown, text sent when clicked).
MENU = [("Show a patient's history", "Show a patient's medical history"),
        ("Book an appointment", "Book an appointment"),
        ("Register a new patient", "Register a new patient"),
        ("Learn about a disease", "Learn about a disease")]
ASK_DISEASE = "Learn about a disease"

# Cancel / drop: "cancel", "never mind", "too late", "the patient is dead".
# Phrases only count in SHORT messages, so a long request that happens to
# contain "too late" is not dropped by mistake.
CANCEL_PHRASES = ["never mind", "nevermind", "forget it", "forget that", "too late", "drop it",
                  "leave it", "not needed", "no longer need", "dont need", "don't need",
                  "changed my mind", "change of plan"]
DEATH_PATTERN = r"\b(dead|died|dies|deceased|passed away|passed on|no longer alive)\b"
SHORT_MESSAGE_WORDS = 8

# Personal medical advice (diagnosis, dosing): refused with a fixed reply.
ADVICE_PATTERNS = [r"\b(dosage|dose|doses|overdose)\b", r"\bshould i (take|use|stop|skip|eat|drink)\b",
                   r"\bdo i have\b", r"\bdiagnose (me|my)\b", r"\bam i (sick|ill|dying|having)\b",
                   r"\bwhat('s| is) wrong with me\b"]

# No open slots: the bubbles (label shown, text sent when clicked) and the
# "Remind me in ..." dropdown. Joining the waiting list sets the recheck date.
NO_SLOTS_OPTIONS = [("Join the waiting list", "join the waiting list"),
                    ("Choose a different specialty", "choose a different specialty"),
                    ("Back to start", "back to start")]
REMIND_DAYS = [3, 7, 14]
DEFAULT_RECHECK_DAYS = 7
NO_SLOTS_QUESTION = "What would you like to do?"
BACK_TO_START_REPLY = "Okay, back to the start. What would you like to do?"


def match_no_slots_choice(text):
    """Read a typed answer to the no-slots question.

    Returns (choice, days): choice is "waitlist", "specialty", "start" or None;
    days is how many days until the recheck (only for "waitlist")."""
    lowered = str(text).lower()
    if "remind" in lowered:
        days = re.search(r"\b(3|7|14)\b", lowered)
        return "waitlist", int(days.group(1)) if days else DEFAULT_RECHECK_DAYS
    if re.search(r"wait|join", lowered):
        return "waitlist", DEFAULT_RECHECK_DAYS
    if re.search(r"different|another|specialty|other", lowered):
        return "specialty", None
    if re.search(r"back|start|menu|home", lowered):
        return "start", None
    return None, None


# Closing the chat. "bye", "thanks, that's all", "no more" end it (short messages only).
CLOSING_OPTIONS = [("Anything else?", "anything else"), ("No, that's all", "no, that's all")]
ANYTHING_ELSE_REPLY = "Sure! What would you like to do?"
FAREWELL_REPLY = "Thank you! Visit again."
NEW_CONVERSATION = ("Start a new conversation", "start a new conversation")
GOODBYE_PATTERN = (r"\b(bye|goodbye|good bye|see you|that's all|that is all|that'll be all|no more|"
                   r"nothing else|nothing more|i'm done|all done)\b")


def is_goodbye(text):
    lowered = str(text).lower().replace("\u2019", "'")
    return len(re.findall(r"[a-z']+", lowered)) <= SHORT_MESSAGE_WORDS - 2 and bool(re.search(GOODBYE_PATTERN, lowered))


NAME_OPTIONS = ("type a patient name or ID, say 'new patient' to register one, "
                "or say 'cancel' to start over")
WAITING_FOR_NAME = "I'm waiting for the patient's full name or patient ID. You can " + NAME_OPTIONS + "."

# New-patient registration: the details we ask for, in this order.
REGISTRATION_QUESTIONS = {
    "name": "What is the new patient's full name?",
    "age": "How old is the patient (in years)?",
    "gender": "What is the patient's gender?",
    "contact_person": "Who is the contact person?",
    "contact_phone": "What is the contact phone number? (digits only, 7 to 15 digits, "
                     "optionally starting with + and the country code)",
}
GENDER_WORDS = {"male": "Male", "man": "Male", "m": "Male", "boy": "Male",
                "female": "Female", "woman": "Female", "f": "Female", "girl": "Female",
                "other": "Other"}
GENDER_CHOICES = ["Male", "Female", "Other"]                          # the gender dropdown
CONTACT_CHOICES = ["self", "son", "daughter", "wife", "husband", "other"]   # the contact dropdown


def clean_name_reply(text):
    """Turn "His name is Ramesh Kumar." into "Ramesh Kumar" before searching."""
    text = re.sub(r"^\s*(?:(?:his|her|their|the|patient'?s?)\s+)?(?:full\s+)?"
                  r"(?:name|id|patient id)\s*(?:is|:)?\s*", "", str(text), flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:it'?s|it is|he is|she is|this is)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" .!?\"'")


def looks_like_name_or_id(text):
    """True for replies like "Ramesh Kumar", "His name is Anita Sharma" or "P001".

    A name here is 1 to 4 words made of letters (plus . ' -), with none of the
    REQUEST_WORDS. So "create a new log" or a whole request sentence is NOT a name.
    """
    cleaned = clean_name_reply(text)
    if re.fullmatch(r"[Pp]\s*\d+", cleaned):
        return True
    words = cleaned.split()
    if not 1 <= len(words) <= 4:
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", w) for w in words):
        return False
    return not ({w.lower() for w in words} & REQUEST_WORDS)


def is_small_talk(text):
    """True for greetings and "what can you do?" style messages: every word is
    a small-talk word. "Hi there!" -> True. "Hi, book a doctor" -> False."""
    words = re.findall(r"[a-z']+", str(text).lower())
    return bool(words) and all(w in SMALL_TALK_WORDS for w in words)


def is_death_mention(text):
    words = re.findall(r"[a-z']+", str(text).lower())
    return len(words) <= SHORT_MESSAGE_WORDS and bool(re.search(DEATH_PATTERN, str(text).lower()))


def is_cancel_request(text, waiting=False, confirming_booking=False):
    """True if the user wants to drop the current request ("cancel", "never mind",
    "too late", "the patient is dead").

    waiting:            the agent is waiting for an answer, so a plain "cancel" or
                        "stop" is meant for that question.
    confirming_booking: a plain "cancel"/"stop" there still means "no" to the
                        booking (handled by the normal no-answer); only the
                        phrases and a death mention drop everything.
    """
    lowered = str(text).lower().replace("\u2019", "'")
    words = re.findall(r"[a-z']+", lowered)
    if is_death_mention(text):
        return True
    if len(words) <= SHORT_MESSAGE_WORDS and any(phrase in lowered for phrase in CANCEL_PHRASES):
        return True
    if confirming_booking:
        return False
    stop = bool(set(words) & STOP_WORDS)
    if waiting:
        return stop
    # No open question: only a short, plain "cancel" counts ("cancel my appointment" is a real request).
    return stop and len(words) <= 3 and not ({"appointment", "booking", "slot"} & set(words))


def asks_personal_advice(text):
    lowered = str(text).lower().replace("\u2019", "'")
    return any(re.search(pattern, lowered) for pattern in ADVICE_PATTERNS)


def looks_like_new_request(text):
    """True for replies that read like a new request: a question, 3+ words,
    or a request word such as "book" or "show"."""
    words = re.findall(r"[a-z']+", str(text).lower())
    return "?" in str(text) or len(words) >= 3 or bool(set(words) & REQUEST_WORDS)


def wants_new_patient(text):
    lowered = str(text).lower()
    return "new patient" in lowered or "register" in lowered


def parse_registration_value(field, text):
    """Check one registration answer. Returns (value, None) or (None, problem)."""
    text = str(text).strip()
    if field == "name":
        name = " ".join(clean_name_reply(text).split())
        if looks_like_name_or_id(name) and not re.fullmatch(r"[Pp]\s*\d+", name):
            return name, None
        return None, "That doesn't look like a name (letters only, up to 4 words)."
    if field == "age":
        match = re.search(r"\b(\d{1,3})\b", text)
        if match and 0 <= int(match.group(1)) <= 120:
            return int(match.group(1)), None
        return None, "Please give the age as a number between 0 and 120."
    if field == "gender":
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in GENDER_WORDS:
                return GENDER_WORDS[word], None
        return None, "Please reply male, female or other."
    if field == "contact_person":
        contact = " ".join(clean_name_reply(text).lower().split())
        if contact and len(contact) <= 30 and re.fullmatch(r"[a-z][a-z '-]*", contact):
            return contact, None
        return None, "Please give the contact person in a few words, e.g. self, son or daughter."
    if field == "contact_phone":
        number, problem = clean_phone(text)
        if number:
            return number, None
        return None, problem or "Please give a phone number."
    return None, "Unknown detail."


def nice_date(date_text):
    """'2026-10-26' -> 'Mon 26 Oct' (easier to read than the raw date)."""
    return datetime.strptime(date_text, "%Y-%m-%d").strftime("%a %d %b")


def slot_label(slot):
    """A readable slot for the dropdown: 'Mon 26 Oct, 09:30 – Dr. Sarah Thompson (General Medicine)'."""
    return f"{nice_date(slot['date'])}, {slot['time']} – {slot['doctor_name']} ({slot['specialty']})"


def clean_booking_sentence(text):
    """Check the LLM's one-sentence booking question. Returns None (use the
    template instead) if it is empty, too long, or contains numbers - the
    exact date/time/slot are shown separately, so the LLM must not restate
    (and possibly change) them. A leading 'Hi!'/'Hello,' is removed."""
    sentence = re.sub(r"^\s*(?:hi|hello|hey)\b[\s!,.]*", "", str(text or ""), flags=re.IGNORECASE).strip()
    if not sentence or len(sentence) > 200 or re.search(r"\d", sentence):
        return None
    return sentence[0].upper() + sentence[1:]


def specialties():
    """All specialties in doctors.csv, sorted."""
    return sorted(read_csv(config.DOCTORS_CSV)["specialty"].unique())


def match_specialty(text):
    """'nephrologist', 'Nephrology' or 'my nephrology' -> 'Nephrology' (None if no match)."""
    lowered = str(text).lower()
    for word, specialty in SPECIALIST_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return specialty
    return next((s for s in specialties() if s.lower() in lowered), None)


def interpret_reply(text):
    """Read a yes/no answer. Any "no"-type word wins, so we never write by mistake.

    Returns "yes", "no" or None (unclear).
    """
    words = set(re.findall(r"[a-z']+", str(text).lower()))
    if words & NO_WORDS:
        return "no"
    if words & YES_WORDS or "go ahead" in str(text).lower():
        return "yes"
    return None


# ---------------------------------------------------------------------------
# Agent steps (each one is logged as ONE row)
# ---------------------------------------------------------------------------

@log_tool_call
def retrieve_patient_history(patient_id, llm=None):
    """Get the records, then the stored summary (memory lookup). If the stored
    summary is missing or out of date, write a fresh one with the LLM."""
    history = get_patient_history(patient_id)
    if history["status"] in ("not_found", "error"):
        return history

    stored = get_patient_summary(patient_id)
    up_to_date = (stored["status"] == "success"
                  and stored["summary"]["record_count"] == len(history["records"])
                  # An EDITED record keeps the same count, so also compare the content
                  # fingerprint. (Summaries saved before this field existed: count only.)
                  and stored["summary"].get("records_hash", records_fingerprint(history["records"]))
                  == records_fingerprint(history["records"]))
    if up_to_date:
        summary, how = stored["summary"]["summary"], "stored summary"
    else:
        fresh = (refresh_patient_summary(patient_id, llm=llm) if stored["status"] != "no_index"
                 else summarize_patient(patient_id, llm=llm))
        if fresh["status"] != "success":
            return {"status": "error", "message": fresh["message"]}
        summary, how = fresh["summary"]["summary"], "new summary"

    p = history["patient"]
    lines = [f"Here's a summary of {p['name']}'s medical history:", summary]
    if history["records"]:
        latest = history["records"][-1]
        lines.append(f"Alerts on file: {latest['alerts']}. (Based on {len(history['records'])} "
                     f"record(s), most recent visit {latest['visit_date']}.)")
    return {"status": "success", "message": f"History retrieved for {p['patient_id']} ({how}).",
            "text": "\n".join(lines), "summary": summary}


@log_tool_call
def prepare_booking(patient, specialty, reason, slot_id=None, llm=None):
    """Find free slots for the specialty and build a booking preview (writes nothing).

    slot_id: a specific slot the user chose; otherwise the earliest one.
    """
    slots = list_available_slots(specialty=specialty)
    if slots["status"] != "success":
        return {"status": slots["status"], "message": slots["message"]}

    offered = slots["slots"]
    chosen = next((s for s in offered if s["slot_id"] == str(slot_id or "").upper()), None)
    if slot_id and chosen is None:
        # Ask the booking tool WHY (preview only, writes nothing): in the past,
        # already booked, or no such slot. Otherwise it belongs to another specialty.
        check = book_appointment(patient["patient_id"], slot_id, reason)
        why = (check["message"] if check["status"] != "needs_confirmation"
               else f"Slot {str(slot_id).upper()} is not a {specialty} slot.")
        return {"status": "invalid", "message": f"I can't book that: {why}"}
    chosen = chosen or offered[0]

    # Another doctor of this specialty has no openings: say so, then offer the others first.
    found = find_doctors(specialty)
    with_slots = {s["doctor_id"] for s in offered}
    empty = [d for d in found.get("doctors", []) if d["doctor_id"] not in with_slots]
    note = ""
    if empty:
        import pandas as pd
        note = no_openings_message(pd.DataFrame(empty)) + " Here is the earliest opening with another doctor. "

    preview = book_appointment(patient["patient_id"], chosen["slot_id"], reason)  # confirm=False
    if preview["status"] != "needs_confirmation":
        return {"status": preview["status"], "message": preview["message"]}

    # The LLM writes ONE short question; the exact details come from code below.
    question = None
    try:
        if llm is None:
            from agent.llm import get_llm
            llm = get_llm()
        booking_text = (f"Patient: {patient['name']}\nDoctor: {chosen['doctor_name']} "
                        f"({chosen['specialty']})\nReason: {reason}")
        reply = (BOOKING_CONFIRMATION_PROMPT | llm).invoke({"booking": booking_text})
        question = clean_booking_sentence(reply.text)
    except Exception:
        question = None
    if question is None:
        question = f"Shall I book this appointment for {patient['name']}?"

    # The exact details, shown ONCE, straight from the booking tool's preview.
    alternatives = [s for s in offered if s["slot_id"] != chosen["slot_id"]]
    question += (f"\n\nDoctor: {chosen['doctor_name']} ({chosen['specialty']}, {chosen['clinic']})"
                 f"\nWhen: {nice_date(chosen['date'])}, {chosen['time']} (slot {chosen['slot_id']})"
                 f"\nReason: {reason}"
                 "\n\nNothing is saved until you say yes.")   # other times: the dropdown (and "Show details")
    if note:
        question = note.strip() + "\n\n" + question
    return {"status": "needs_confirmation", "message": preview["message"],
            "slot": chosen, "offered": offered, "question": question,
            "will_write": preview["will_write"]}


@log_tool_call
def decline_booking(patient, slot):
    """The user said "no" to a booking. Writes nothing; exists so the decision
    is logged (status "declined") and can be counted as a correct refusal."""
    return {"status": "declined",
            "message": f"User declined booking {slot['slot_id']} for {patient['patient_id']}. Nothing was saved."}


@log_tool_call
def confirm_booking(patient, slot, reason):
    """Actually book (confirm=True), then save a long-term memory note."""
    result = book_appointment(patient["patient_id"], slot["slot_id"], reason, confirm=True)
    if result["status"] == "success":
        a = result["appointment"]
        save_patient_memory(patient["patient_id"],
                            f"Booked {a['appointment_id']} with {slot['doctor_name']} "
                            f"({slot['specialty']}) on {slot['date']} at {slot['time']}. Reason: {reason}.",
                            note_type="booking")
    return result


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def menu_question():
    """The four "what would you like to do" bubbles (also shown on the empty welcome screen)."""
    return {"type": "menu", "number": None, "select": None,
            "buttons": [{"label": label, "reply": reply, "display": label} for label, reply in MENU]}


class HealthcareAgent:
    """Holds one conversation: short-term memory, the current plan, and any
    question we are waiting for the user to answer."""

    def __init__(self, llm=None):
        self.llm = llm                    # None = real Claude (created when first needed)
        self.memory = ShortTermMemory()
        self.plan = None                  # the current Plan
        self.facts = {}                   # age / gender from the request
        self.request = ""                 # the request being worked on
        self.step_index = 0               # which plan step runs next
        self.pending = None               # what we asked the user, if anything
        self.trace = []                   # what happened at each step (for the dashboard)
        self.turn_parts = []              # text for the user produced during this turn
        self.showed_medical_info = False  # add the disclaimer to this turn's reply?
        self.last_disease_info = None     # latest MedlinePlus summary (for the dashboard)
        self.show_menu = False            # show the option bubbles under this reply?
        self.show_closing = False         # show "Anything else?" / "No, that's all"?
        self.show_farewell = False        # show "Start a new conversation"?
        self.just_ended = False           # the chat ended this turn (short-term memory was cleared)
        self.last_reply = None            # the previous reply (never send the same one twice)
        self.last_full_reply = None       # the last reply that was not a short 'pick one' line

    # ---- public ----------------------------------------------------------

    def handle(self, user_text):
        """Process one user message and return the reply plus plan and trace."""
        self.memory.add_user_message(user_text)
        self.turn_parts = []
        self.showed_medical_info = False
        self.show_menu = self.show_closing = self.show_farewell = self.just_ended = False
        waiting = bool(self.pending)
        confirming = waiting and self.pending["type"] == "confirm_booking"
        if is_goodbye(user_text):
            reply = self._end_conversation()
        elif is_cancel_request(user_text, waiting=waiting, confirming_booking=confirming):
            reply = self._cancel(user_text)
        elif self.pending:
            reply = self._resume(user_text)
        else:
            reply = self._start(user_text)
        reply = self._avoid_repeat(reply)
        if not self.just_ended:               # ending the chat clears short-term memory
            self.memory.add_ai_message(reply)
        return {"reply": reply,
                "waiting_for": self.pending["type"] if self.pending else None,
                "question": self.pending_question(),
                "plan": [s.model_dump() for s in self.plan.steps] if self.plan else [],
                "trace": list(self.trace),
                "patient_id": self.memory.get_context("patient_id")}

    # ---- planning --------------------------------------------------------

    def _start(self, user_text):
        self.trace = []
        self.request = user_text
        if is_small_talk(user_text):
            # A greeting: fixed reply plus the option bubbles. No plan, no LLM call.
            return self._out_of_scope("greeting", GREETING_REPLY, "Greeting or 'what can you do?'")
        if user_text.strip().lower() == NEW_CONVERSATION[1]:
            return self._out_of_scope("greeting", GREETING_REPLY, "Started a new conversation")
        if re.sub(r"[^a-z ]", "", user_text.lower()).strip() == CLOSING_OPTIONS[0][1]:
            return self._out_of_scope("anything_else", ANYTHING_ELSE_REPLY, "User has another request")
        if is_cancel_request(user_text):
            return self._cancel(user_text)
        if asks_personal_advice(user_text):
            return self._out_of_scope("medical_advice", ADVICE_REPLY, "Asked for personal medical advice")
        if user_text.strip() == ASK_DISEASE:          # option bubble: ask which disease
            self.plan = None
            self.pending = {"type": "ask_disease"}
            return "Which disease or condition would you like to learn about?"
        if len(user_text.split()) <= 5 and wants_new_patient(user_text) and not self.pending:
            self.plan = None
            return self._start_registration()
        patient_id = self.memory.get_context("patient_id")
        current = self.memory.get_context("patient_label", "none")
        planned = make_plan(user_text, conversation=self.memory.as_text(), current_patient=current,
                            patient_id=patient_id, llm=self.llm)
        self._add_trace("plan", "Break the request into sub-goals", planned)
        if planned["status"] == "out_of_scope":
            # The planner classified the message: not a request we can run.
            return self._out_of_scope_by_category(planned["category"], user_text)
        if planned["status"] == "no_steps":
            return self._out_of_scope("unclear", UNCLEAR_REPLY, "Nothing the assistant can do")
        if planned["status"] != "success":
            self.plan = None
            return "\n\n".join(self.turn_parts + [planned["message"]])

        # The plan is NOT put in the reply: it is returned separately
        # (handle() -> "plan") and shown under "Show details" in the dashboard.
        self.plan = planned["plan"]
        self.facts = extract_patient_facts(user_text)
        self.step_index = 0
        return self._run_steps()

    # ---- scope guard -------------------------------------------------------

    def _out_of_scope(self, category, template, goal):
        """Answer with a FIXED template and the option bubbles. Nothing else runs."""
        self.plan = None
        self.show_menu = True
        self._add_trace("scope_guard", goal, {"status": category, "message": "Replied with a fixed message."})
        return "\n\n".join(self.turn_parts + [template])

    def _out_of_scope_by_category(self, category, user_text):
        if category == "cancel":
            return self._cancel(user_text)
        templates = {"greeting": GREETING_REPLY, "off_topic": OFF_TOPIC_REPLY,
                     "medical_advice": ADVICE_REPLY, "unclear": UNCLEAR_REPLY}
        return self._out_of_scope(category, templates.get(category, UNCLEAR_REPLY),
                                  f"Planner classified the message as {category}")

    def _cancel(self, user_text):
        """End the current flow. Nothing is saved; a death mention gets a kind word."""
        if self.pending and self.pending["type"] == "confirm_booking":
            result = decline_booking(self.memory.get_context("patient"), self.pending["slot"])
            self._add_trace("decline_booking", "User dropped the booking", result)
        self.pending = None
        self.plan = None
        self.show_menu = True
        self._add_trace("cancel", "User dropped the request",
                        {"status": "cancelled", "message": "Ended the current request. Nothing was changed."})
        reply = CANCELLED_REPLY + (" " + SORRY_REPLY if is_death_mention(user_text) else "")
        return "\n\n".join(self.turn_parts + [reply])

    def _end_conversation(self):
        """"bye" / "that's all": close any open question, clear SHORT-TERM memory (this chat),
        keep LONG-TERM patient notes (they live in memory_store/)."""
        if self.pending and self.pending["type"] == "confirm_booking":
            result = decline_booking(self.memory.get_context("patient"), self.pending["slot"])
            self._add_trace("decline_booking", "Chat ended before the booking was confirmed", result)
        self._add_trace("end_conversation", "User ended the chat",
                        {"status": "ended", "message": "Cleared short-term memory; long-term notes are kept."})
        self.pending = None
        self.plan = None
        self.facts = {}
        self.request = ""
        self.memory.clear()
        self.show_farewell = True
        self.just_ended = True
        return FAREWELL_REPLY

    def _avoid_repeat(self, reply):
        """Never send the same bot message twice in a row. A repeat becomes a
        short line (the bubbles are still shown)."""
        if reply in (self.last_reply, self.last_full_reply):
            has_widgets = bool(self.pending_question())
            short = PICK_ONE_REPLY if has_widgets else STILL_WAITING
            if short == self.last_reply:
                short = PICK_ONE_AGAIN if has_widgets else "Please answer the question above."
            self.last_reply = short
            return short
        self.last_reply = self.last_full_reply = reply
        return reply

    # ---- running steps ---------------------------------------------------

    def _run_steps(self):
        """Run plan steps in order until done or until we need the user's answer."""
        while self.step_index < len(self.plan.steps):
            step = self.plan.steps[self.step_index]
            question = self._run_step(step)
            if question:                         # paused: ask the user
                return "\n\n".join(self.turn_parts + [question])
            self.step_index += 1
        self.pending = None
        self.show_closing = True                 # task finished: offer "Anything else?" / "No, that's all"
        if not self.turn_parts:                  # e.g. the plan only identified the patient
            self.turn_parts.append("Got it. Is there anything else I can help with?")
        if self.showed_medical_info:
            self.turn_parts.append(config.DISCLAIMER)
        return "\n\n".join(self.turn_parts)

    def _run_step(self, step):
        """Run one step. Returns a question for the user if we must pause, else None."""
        patient = self.memory.get_context("patient")

        if step.tool == "identify_patient":
            if patient and not step.input:
                self._add_trace(step.tool, step.goal, {"status": "already_known",
                                "message": f"Using {patient['name']} ({patient['patient_id']}) from this conversation."})
                return None
            name = step.input or (self.plan.patient_name or "")
            return self._identify(name)

        if step.tool in ("get_patient_history", "book_appointment") and not patient:
            self._add_trace(step.tool, step.goal, {"status": "skipped", "message": "No confirmed patient."})
            self.turn_parts.append("I can't do that yet because I don't know which patient this is.")
            return None

        if step.tool == "get_patient_history":
            result = retrieve_patient_history(patient["patient_id"], llm=self.llm)
            self._add_trace(step.tool, step.goal, result)
            if result["status"] == "success":
                self.showed_medical_info = True
                self.turn_parts.append(result["text"])
            else:
                self.turn_parts.append(f"Sorry, I couldn't get the medical history: {result['message']}")
            return None

        if step.tool == "book_appointment":
            if not self._specialty_was_given(step):
                self.pending = {"type": "choose_specialty"}
                return "Which kind of doctor would you like to book?"
            return self._prepare_booking(step)

        if step.tool == "search_disease_info":
            topic = step.input or self.plan.condition or self.request
            result = summarize_disease_info(topic, question=step.goal, llm=self.llm,
                                            patient_id=patient["patient_id"] if patient else None)
            self._add_trace(step.tool, step.goal, result)
            if result["status"] == "success":
                self.last_disease_info = result
                self.showed_medical_info = True
                self.turn_parts.append(f"Here's what MedlinePlus says about {topic}:\n{result['answer']}")
            else:
                self.turn_parts.append(result["message"])
            return None

        self._add_trace(step.tool, step.goal, {"status": "skipped", "message": "Unknown step."})
        return None

    def _identify(self, name):
        result = identify_patient(self.request, name=name, facts=self.facts)
        self._add_trace("identify_patient", "Identify the patient", result)
        if result["status"] == "needs_confirmation":
            self.pending = {"type": "confirm_patient", "patient": result["patient"]}
            return result["question"]
        if result["status"] == "needs_choice":
            self.pending = {"type": "choose_patient", "candidates": result["candidates"]}
            return result["question"]
        if result["status"] == "needs_name":
            if name and result.get("candidate_count") == 0:
                # A typed name that matches nobody: offer to register it.
                self.pending = {"type": "patient_name", "unknown_name": name}
                return (f"I could not find a patient matching '{name}'. You can type a patient name "
                        f"or ID, say 'new patient' to register '{name}' as a new patient, "
                        f"or say 'cancel' to start over.")
            self.pending = {"type": "patient_name"}
            return result["question"]
        self.pending = None
        self.step_index = len(self.plan.steps)   # cannot continue without a patient
        return f"{result['message']} I cannot continue without identifying the patient."

    def _after_booking_step(self, text=None, show_menu=False):
        """The booking step is over without a booking (waiting list, back to start):
        run any steps that are left (e.g. the disease information), then reply."""
        self.pending = None
        if text:
            self.turn_parts.append(text)
        self.step_index += 1
        reply = self._run_steps()
        if show_menu:
            self.show_menu = True
        return reply

    def _resume_no_slots(self, user_text):
        choice, days = match_no_slots_choice(user_text)
        patient = self.memory.get_context("patient")

        if choice == "waitlist":
            preview = add_to_waitlist(patient["patient_id"], self.pending["specialty"],
                                      reason=self.pending["reason"], recheck_days=days)   # writes nothing
            self._add_trace("add_to_waitlist", "Preview the waiting-list entry", preview)
            if preview["status"] == "needs_confirmation":
                w = preview["will_write"]
                self.pending = {"type": "confirm_waitlist", "specialty": w["specialty"],
                                "reason": w["reason"], "days": days}
                return (f"Add {patient['name']} to the waiting list for {w['specialty']}, with a reminder "
                        f"on {nice_date(w['recheck_on'])}? Nothing is saved until you say yes.")
            return self._after_booking_step(preview["message"], show_menu=True)   # e.g. already waiting

        if choice == "specialty":
            self.pending = {"type": "choose_specialty"}
            return "Which kind of doctor would you like to book?"

        if choice == "start":
            if self.step_index + 1 < len(self.plan.steps):     # still something else to answer
                return self._after_booking_step("Okay, I won't book anything.", show_menu=True)
            self.pending, self.plan = None, None
            self.show_menu = True
            return BACK_TO_START_REPLY

        if looks_like_new_request(user_text) and not looks_like_name_or_id(user_text):
            return self._switch_to_new_request(user_text)
        return PICK_ONE_REPLY

    def _resume_confirm_waitlist(self, answer):
        patient = self.memory.get_context("patient")
        if answer == "yes":
            result = add_to_waitlist(patient["patient_id"], self.pending["specialty"],
                                     reason=self.pending["reason"], recheck_days=self.pending["days"],
                                     confirm=True)
            self._add_trace("add_to_waitlist", "Save the waiting-list entry", result)
            text = (f"Done! {result['message']}" if result["status"] == "success"
                    else f"Sorry, I couldn't add them to the waiting list. {result['message']}")
            return self._after_booking_step(text)
        if answer == "no":
            self.pending = {"type": "no_slots", "specialty": self.pending["specialty"],
                            "reason": self.pending["reason"]}
            return "Okay, I haven't added anyone to the waiting list. " + NO_SLOTS_QUESTION
        return "Please reply 'yes' to join the waiting list, or 'no'."

    def _identify_by_number(self, number):
        """Digits only ("007" or "7") while waiting for a patient: try ID P007.
        One match -> ask "Did you mean ...?". Other patients are never listed."""
        patient_id = f"P{number:03d}"
        result = identify_patient(self.request, name=patient_id, facts=self.facts)
        self._add_trace("identify_patient", f"Look up patient ID {patient_id}", result)
        patient = result.get("patient")
        if result["status"] == "needs_confirmation" and patient["patient_id"] == patient_id:
            self.pending = {"type": "confirm_patient", "patient": patient}
            return f"Did you mean {patient_id}, {patient['name']}, age {patient['age']}?"
        return f"I couldn't find a patient with ID {patient_id}. " + ASK_NAME_QUESTION

    def _prepare_booking(self, step, slot_id=None):
        patient = self.memory.get_context("patient")
        reason = self.plan.condition or f"{step.input} appointment"
        result = prepare_booking(patient, step.input, reason, slot_id=slot_id, llm=self.llm)
        self._add_trace("book_appointment", step.goal, result)
        if result["status"] == "no_slots":
            # Nothing open: say so and offer the four bubbles (waiting list, check again, ...).
            self.pending = {"type": "no_slots", "specialty": step.input, "reason": reason}
            return result["message"] + "\n\n" + NO_SLOTS_QUESTION
        if result["status"] == "needs_confirmation":
            self.pending = {"type": "confirm_booking", "slot": result["slot"], "reason": reason,
                            "offered": result["offered"]}
            return result["question"]
        if result["status"] == "invalid" and slot_id:
            return result["message"] + " Reply 'yes' for the first offer, pick another time, or say 'no'."
        self.turn_parts.append(f"I couldn't book that appointment. {result['message']}")
        return None

    # ---- answering our questions ------------------------------------------

    def _resume(self, user_text):
        kind = self.pending["type"]
        answer = interpret_reply(user_text)

        if kind == "confirm_patient":
            if answer == "yes":
                return self._use_patient(self.pending["patient"])
            if answer == "no":
                self.pending = {"type": "patient_name"}
                return "Sorry about that. What is the patient's full name or patient ID?"
            return "Please reply 'yes' if this is the right patient, or 'no'."

        if kind == "choose_patient":
            # The user picks one of the listed patients by ID (or exact name).
            # Picking a listed patient counts as confirming them.
            candidates = self.pending["candidates"]
            typed_ids = {f"P{int(n):03d}" for n in re.findall(r"\bp\s*0*(\d+)\b", user_text.lower())}
            typed_name = clean_name_reply(user_text).lower()
            picked = [p for p in candidates
                      if p["patient_id"] in typed_ids or p["name"].lower() == typed_name]
            if len(picked) == 1:
                return self._use_patient(picked[0])
            if answer == "no" or {"neither", "none"} & set(re.findall(r"[a-z]+", user_text.lower())):
                self.pending = {"type": "patient_name"}
                return "Okay. What is the patient's full name or patient ID?"
            if not looks_like_name_or_id(user_text) and looks_like_new_request(user_text):
                return self._switch_to_new_request(user_text)
            return "Please pick one from the list, or choose 'None of these'."

        if kind == "patient_name":
            # Sort the reply before searching: a name/ID, "new patient",
            # a brand-new request, or something unclear.
            if wants_new_patient(user_text):
                return self._start_registration(self.pending.get("unknown_name"))
            if re.fullmatch(r"\d{1,4}", user_text.strip()):    # "007" or "7" means P007
                return self._identify_by_number(int(user_text.strip()))
            if is_small_talk(user_text):          # "Hi" is not a patient name
                return WAITING_FOR_NAME
            if looks_like_name_or_id(user_text):
                question = self._identify(clean_name_reply(user_text))
                return "\n\n".join(self.turn_parts + [question])
            if looks_like_new_request(user_text):
                return self._switch_to_new_request(user_text)
            return WAITING_FOR_NAME

        if kind == "no_slots":
            return self._resume_no_slots(user_text)

        if kind == "confirm_waitlist":
            return self._resume_confirm_waitlist(answer)

        if kind == "ask_disease":
            # The reply names a disease ("asthma"); a full question is used as it is.
            if not looks_like_new_request(user_text):
                user_text = f"What is {user_text.strip()} and what are its treatments?"
            return self._switch_to_new_request(user_text)

        if kind == "register_patient":
            return self._continue_registration(user_text, answer)

        if kind == "choose_specialty":
            specialty = match_specialty(user_text)
            if specialty:
                step = self.plan.steps[self.step_index]
                step.input = specialty
                self.pending = None
                question = self._prepare_booking(step)
                if question:
                    return "\n\n".join(self.turn_parts + [question])
                self.step_index += 1
                return self._run_steps()
            if looks_like_new_request(user_text) and not looks_like_name_or_id(user_text):
                return self._switch_to_new_request(user_text)
            return "Please choose a specialty from the list."

        if kind == "confirm_booking":
            step = self.plan.steps[self.step_index]
            chosen_slot = re.search(r"\bS\d{3}\b", user_text.upper())
            if chosen_slot and answer != "no":
                question = self._prepare_booking(step, slot_id=chosen_slot.group())
                if question:
                    return "\n\n".join(self.turn_parts + [question])
                self.step_index += 1
                return self._run_steps()
            if answer == "yes":
                patient, slot = self.memory.get_context("patient"), self.pending["slot"]
                result = confirm_booking(patient, slot, self.pending["reason"])
                self._add_trace("confirm_booking", "Book the confirmed slot", result)
                if result["status"] == "success":
                    self.turn_parts.append(
                        f"Done! I've booked {patient['name']} with {slot['doctor_name']} "
                        f"({slot['specialty']}, {slot['clinic']}) on {slot['date']} at {slot['time']}. "
                        f"The appointment ID is {result['appointment']['appointment_id']}.")
                else:
                    self.turn_parts.append(f"Sorry, I couldn't complete the booking. {result['message']}")
            elif answer == "no":
                result = decline_booking(self.memory.get_context("patient"), self.pending["slot"])
                self._add_trace("decline_booking", "User declined the booking", result)
                self.turn_parts.append("Okay, I won't book anything.")
            else:
                return "Please reply 'yes' to book, a slot ID (e.g. S006) to choose another slot, or 'no'."
            self.pending = None
            self.step_index += 1
            return self._run_steps()

        self.pending = None
        return "Sorry, I lost track of the question. Please repeat your request."

    # ---- registering a new patient ----------------------------------------
    #
    # pending = {"type": "register_patient", "fields": {...}, "asking": ...}
    # "asking" is the detail we asked for ("name", "age", ...), "confirm"
    # (preview shown, waiting for yes/no) or "edit" (which detail to change).

    def _start_registration(self, typed_name=None):
        """Begin registration. Details we already know are filled in:
        the typed name, and age/gender from the original request."""
        name = parse_registration_value("name", typed_name)[0] if typed_name else None
        fields = {"name": name,
                  "age": self.facts.get("age"),
                  "gender": GENDER_WORDS.get(self.facts.get("gender") or ""),
                  "contact_person": None,
                  "contact_phone": None}
        self.pending = {"type": "register_patient", "fields": fields, "asking": None}
        return "Let's register a new patient. " + self._next_registration_question()

    def _next_registration_question(self):
        """Ask for the next missing detail, or show the preview when all are known."""
        fields = self.pending["fields"]
        for field, question in REGISTRATION_QUESTIONS.items():
            if fields[field] is None:
                self.pending["asking"] = field
                return question

        preview = add_patient(fields["name"], fields["age"], fields["gender"],
                              fields["contact_person"], fields["contact_phone"])   # confirm=False: writes nothing
        self._add_trace("add_patient", "Preview the new patient", preview)
        if preview["status"] != "needs_confirmation":
            # e.g. "duplicate": same name and age already exists
            self.pending = {"type": "patient_name"}
            return f"{preview['message']} Nothing was saved. You can {NAME_OPTIONS}."
        self.pending["asking"] = "confirm"
        w = preview["will_write"]
        similar = re.search(r"Note: similar existing patient\(s\): .*?\.(?= Please)", preview["message"])
        return ("Here are the new patient's details:\n"
                f"Name: {w['name']}\nAge: {w['age']}\nGender: {w['gender']}\n"
                f"Contact person: {w['contact_person']}\nContact phone: {w['contact_phone']}\nPatient ID: {w['patient_id']}"
                + (f"\n\n{similar.group()}" if similar else "")
                + "\n\nNothing is saved until you say yes. Reply 'yes' to save, 'no' to change "
                  "a detail, or 'cancel' to stop.")

    def _continue_registration(self, user_text, answer):
        asking, fields = self.pending["asking"], self.pending["fields"]

        if asking == "confirm":
            if answer == "yes":
                result = add_patient(fields["name"], fields["age"], fields["gender"],
                                     fields["contact_person"], fields["contact_phone"], confirm=True)
                self._add_trace("add_patient", "Save the new patient", result)
                if result["status"] != "success":
                    self.pending = {"type": "patient_name"}
                    return f"{result['message']} Nothing was saved. You can {NAME_OPTIONS}."
                new = result["patient"]
                self.turn_parts.append(f"I've registered {new['name']} as patient {new['patient_id']}.")
                return self._use_patient(new)   # continue the original request
            if answer == "no":
                self.pending["asking"] = "edit"
                return ("Which detail should change? Reply with the detail and the new value, "
                        "for example 'age 71', 'contact person daughter' or 'phone +15550100123'.")
            return "Please reply 'yes' to save, 'no' to change a detail, or 'cancel' to stop."

        if asking == "edit":
            match = re.match(r"\s*(name|age|gender|sex|contact\s+phone|phone(?:\s+number)?|contact(?:\s+person)?)"
                             r"\s*[:=]?\s*(.+)",
                             user_text, flags=re.IGNORECASE)
            if not match:
                return "Please reply like 'age 71', 'gender female', 'name Grace Hopper', 'contact person son' or 'phone 5550100123'."
            field = {"sex": "gender"}.get(match.group(1).lower(), match.group(1).lower())
            if "phone" in field:
                field = "contact_phone"
            elif field.startswith("contact"):
                field = "contact_person"
            value, problem = parse_registration_value(field, match.group(2))
            if problem:
                return problem
            fields[field] = value
            return self._next_registration_question()

        value, problem = parse_registration_value(asking, user_text)   # one detail
        if problem:
            # Name what was typed, so two different wrong answers never get an identical reply.
            return f"I couldn't use '{user_text.strip()[:40]}'. {problem} {REGISTRATION_QUESTIONS[asking]}"
        fields[asking] = value
        return self._next_registration_question()

    # ---- the open question, as structured options (for clickable widgets) --

    def pending_question(self):
        """Describe what we are waiting for as structured options, so a UI can
        show buttons or a dropdown instead of parsing the reply text.

        Returns None (no question) or a dict:
            {"type": ..., "buttons": [option, ...],
             "select": None or {"label": ..., "options": [option, ...], "submit": ...},
             "number": None or {"label": ..., "min": ..., "max": ..., "submit": ...,
                                "display": "Age: {value}"}}
        Each option is {"label": shown on the widget,
                        "reply": the text sent to the agent (what it already understands),
                        "display": the user's chat message for this choice}.
        A "number" widget sends the number as text (e.g. "41"); its "display"
        template gives the user's chat message ("Age: 41").
        """
        def option(label, reply, display=None):
            return {"label": label, "reply": reply, "display": display or label}

        if not self.pending:
            if self.show_menu:     # after a fixed reply: the "what would you like to do" bubbles
                return menu_question()
            if self.show_closing:  # after a finished task
                return {"type": "closing", "number": None, "select": None,
                        "buttons": [option(label, reply) for label, reply in CLOSING_OPTIONS]}
            if self.show_farewell:
                return {"type": "ended", "number": None, "select": None,
                        "buttons": [option(*NEW_CONVERSATION)]}
            return None

        kind = self.pending["type"]
        question = {"type": kind, "buttons": [], "select": None, "number": None}

        if kind == "confirm_patient":
            question["buttons"] = [option("Yes", "yes"), option("No", "no")]

        elif kind == "choose_patient":
            # Exact matches first, then "Similar name"; "None of these" is always there.
            people = [option(f"{p['name']}, age {p['age']}, {p['gender']} ({p['patient_id']})"
                             + (f" - {p['match_label']}" if p.get("match_label") else "")
                             + (f" - {p['note']}" if p.get("note") else ""),
                             p["patient_id"], f"Patient: {p['name']} ({p['patient_id']})")
                      for p in self.pending["candidates"]]
            question["select"] = {"label": "Which patient do you mean?",
                                  "options": people + [option("None of these", "neither")],
                                  "submit": "Confirm"}

        elif kind == "confirm_booking":
            question["buttons"] = [option("Yes, book it", "yes"), option("No", "no")]
            others = [s for s in self.pending.get("offered", [])
                      if s["slot_id"] != self.pending["slot"]["slot_id"]]
            if others:
                question["select"] = {
                    "label": "Or choose another time",
                    "options": [option(slot_label(s), s["slot_id"],
                                       f"Slot: {nice_date(s['date'])}, {s['time']}, {s['doctor_name']}")
                                for s in others],
                    "submit": "Confirm"}

        elif kind == "choose_specialty":
            question["select"] = {"label": "Which kind of doctor?",
                                  "options": [option(s, s, f"Specialty: {s}") for s in specialties()],
                                  "submit": "Confirm"}

        elif kind == "no_slots":
            question["buttons"] = [option(label, reply) for label, reply in NO_SLOTS_OPTIONS]
            question["select"] = {"label": "Remind me when to check again",
                                  "options": [option(f"Remind me in {d} days", f"remind me in {d} days")
                                              for d in REMIND_DAYS],
                                  "submit": "Confirm"}

        elif kind == "confirm_waitlist":
            question["buttons"] = [option("Yes, add to the list", "yes"), option("No", "no")]

        elif kind == "ask_disease":    # the disease is typed in the text box
            question["buttons"] = [option("Cancel", "cancel")]

        elif kind == "patient_name":   # the name itself is typed in the text box
            question["buttons"] = [option("Register a new patient", "new patient"), option("Cancel", "cancel")]

        elif kind == "register_patient":
            asking = self.pending["asking"]
            if asking == "age":
                # A number input: the page sends the number as text, e.g. "41".
                question["number"] = {"label": "Age (years)", "min": 0, "max": 120,
                                      "submit": "Confirm", "display": "Age: {value}"}
            elif asking == "gender":
                question["select"] = {"label": "Gender",
                                      "options": [option(g, g.lower(), f"Gender: {g}") for g in GENDER_CHOICES],
                                      "submit": "Confirm"}
            elif asking == "contact_person":
                question["select"] = {"label": "Contact person",
                                      "options": [option(c.capitalize(), c, f"Contact person: {c}") for c in CONTACT_CHOICES],
                                      "submit": "Confirm"}
            elif asking == "confirm":
                question["buttons"] = [option("Yes, save patient", "yes"), option("No, change a detail", "no")]
            else:
                return None   # the name is typed in the text box

        if question["buttons"] or question["select"] or question["number"]:
            return question
        return None

    # ---- helpers -----------------------------------------------------------

    def _specialty_was_given(self, step):
        """False if the booking step has no specialty, or if the planner chose
        General Medicine although the user never asked for it. Then we ask."""
        specialty = (step.input or "").strip()
        if not specialty:
            return False
        if specialty.lower() == "general medicine":
            return bool(re.search(r"general|\bgp\b|physician|family doctor|primary care",
                                  self.request.lower()))
        return True

    def _switch_to_new_request(self, user_text):
        """The user moved on: drop the open question and handle the new request."""
        self.pending = None
        earlier_steps = len(self.trace)
        reply = self._start(user_text)     # _start begins a fresh trace
        self.trace.insert(0, {"step": 0, "tool": "new_request", "goal": "Handle the reply as a new request",
                              "status": "switched",
                              "message": f"Dropped the open question ({earlier_steps} earlier step(s))."})
        return reply

    def _use_patient(self, patient):
        """The user confirmed this patient: remember them and continue the plan."""
        self.memory.set_context("patient", patient)
        self.memory.set_context("patient_id", patient["patient_id"])
        self.memory.set_context("patient_label",
                                f"{patient['name']} ({patient['patient_id']}, age {patient['age']})")
        self.pending = None
        self._add_trace("confirm_patient", "User confirmed the patient",
                        {"status": "confirmed", "message": f"{patient['name']} ({patient['patient_id']})"})
        if self.plan is None:                    # registered from the menu: nothing else was planned
            self.show_closing = True
            return "\n\n".join(self.turn_parts)
        self.step_index += 1
        return self._run_steps()

    def _add_trace(self, tool, goal, result):
        self.trace.append({"step": len(self.trace) + 1, "tool": tool, "goal": goal,
                           "status": result.get("status"), "message": result.get("message", "")})


# ---------------------------------------------------------------------------
# Chat with the agent in the terminal:  python -m agent.agent
# (works on the REAL data/ files; bookings are only made after you type yes)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = HealthcareAgent()
    print(f"Healthcare assistant. {config.DISCLAIMER}\nType 'quit' to exit.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if text:
            result = agent.handle(text)
            print("\nAssistant:", result["reply"], "\n")
            q = result["question"]       # the options a web page would show as buttons / a dropdown
            if q:
                labels = [b["label"] for b in q["buttons"]] + [o["label"] for o in (q["select"] or {"options": []})["options"]]
                print("Options:", " | ".join(labels), "\n")
