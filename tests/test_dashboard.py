"""
test_dashboard.py - tests for the Streamlit dashboard (Phase 9).

Run from the project root:
    python -m tests.test_dashboard

Uses Streamlit's AppTest, which runs the real app script without a browser
and lets us click buttons, pick options and type in the chat. Offline and
free: fake LLM, saved MedlinePlus reply, test copy of the data.
"""

import json
import os
from pathlib import Path

import requests
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402
from app import dashboard_data as data  # noqa: E402
from memory.vector_store import build_patient_index  # noqa: E402
from tests.fake_data import FAKE_SUMMARIES  # noqa: E402
from tools import disease_search_tool  # noqa: E402
from tools.csv_helpers import read_csv  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")
CKD_XML = (Path(__file__).parent / "fixtures" / "medlineplus_ckd_sample.xml").read_text(encoding="utf-8")
SCENARIO = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist "
            "for him. Also, can you summarize latest treatment methods?")
PLAN = json.dumps({"patient_name": None, "condition": "chronic kidney disease", "steps": [
    {"goal": "Identify the patient", "tool": "identify_patient", "input": ""},
    {"goal": "Retrieve history", "tool": "get_patient_history", "input": ""},
    {"goal": "Book a nephrologist", "tool": "book_appointment", "input": "Nephrology"},
    {"goal": "Summarize latest treatment methods", "tool": "search_disease_info",
     "input": "chronic kidney disease"}]})
BOOKING = "Shall I book Ramesh Kumar with Dr. Meera Shah on 2026-10-21 at 14:00?"
ANSWER = ("- Treatments cannot cure kidney disease, but they may slow it [1].\n"
          "- If the kidneys fail, dialysis or a kidney transplant is needed [1].")


class FakeReply:
    status_code = 200
    text = CKD_XML
    def raise_for_status(self): pass


def all_text(at):
    """Every piece of text on the page, joined (for simple 'is it shown?' checks)."""
    parts = []
    for kind in ("text", "markdown", "info", "warning", "success", "error", "caption", "header", "subheader"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [str(e.value) for e in at.sidebar.warning]
    return "\n".join(parts)


def go(at, page):
    at.radio(key="page").set_value(page).run()
    return at


def new_app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[PLAN, BOOKING, ANSWER])
    at.run()
    return at


# ---------------------------------------------------------------------------

def test_every_page_loads():
    print("every page loads, with the disclaimer")
    at = new_app()
    for page in ["Health Assistant Helexa", "Patients", "Records", "Doctors", "Appointments", "Medical info",
                 "Evaluation", "Memory & logs"]:
        go(at, page)
        check(f"'{page}' renders without errors", not at.exception, at.exception)
        check(f"  disclaimer shown on '{page}'", "Demo only, not medical advice" in all_text(at))
    sidebar = " ".join(str(c.value) for c in at.sidebar.caption)
    check("start-up slot creation is skipped under the frozen test clock (shown in the sidebar)",
          "Future slots: The clock is frozen (APP_NOW)" in sidebar, sidebar)
    check("app uses the test copy, not data/ (no 'ORIGINAL' warning)",
          "ORIGINAL" not in all_text(at) and config.DATA_DIR != ORIGINAL_DATA_DIR)


def test_patient_and_doctor_views():
    print("patient and doctor views")
    at = go(new_app(), "Patients")
    at.selectbox[0].set_value("Daniel Brooks (P012)").run()
    check("Daniel Brooks -> 'has no medical records on file'",
          "Daniel Brooks has no medical records on file" in all_text(at), all_text(at)[:500])
    at.selectbox[0].set_value("Ramesh Kumar (P001)").run()
    check("Ramesh Kumar -> stored summary and 4 records",
          "chronic kidney disease" in all_text(at).lower() and len(at.dataframe[0].value) == 4)

    at = go(at, "Doctors")
    at.selectbox[0].set_value("Dr. Kavita Rao - Psychiatry").run()
    check("psychiatrist -> warning: no open upcoming slots",
          "no open upcoming slots" in all_text(at), all_text(at)[:400])
    at.selectbox[0].set_value("Dr. Meera Shah - Nephrology").run()
    states = list(at.dataframe[0].value["state"])
    check("nephrologist calendar shows available, booked and past slots",
          {"available", "booked", "past"} <= set(states), states)


def test_assistant_scenario_and_tracking():
    print("assistant chat -> booking appears in tracking, memory and medical info")
    real_get = disease_search_tool.requests.get
    disease_search_tool.requests.get = lambda url, params=None, timeout=None: FakeReply()
    at = new_app()
    at.chat_input[0].set_value(SCENARIO).run()
    check("turn 1: asks to confirm Ramesh Kumar", "Name: Ramesh Kumar" in all_text(at))
    check("  plan (4 steps) is inside 'Show details', not in the message",
          len(at.chat_message[-1].expander[0].dataframe[0].value) == 4)
    at.chat_input[0].set_value("yes").run()
    check("turn 2: booking question with S004", "(slot S004)" in all_text(at))
    at.chat_input[0].set_value("yes").run()
    check("turn 3: booked + sources shown",
          "I've booked" in all_text(at) and "medlineplus.gov" in all_text(at))
    check("no errors during the chat", not at.exception, at.exception)

    # The chat should feel like a normal chat.
    assistant = [m for m in at.chat_message if m.proto.name == "assistant"]
    replies = [m.markdown[0].value for m in assistant]
    check("each assistant message is only the answer/question (no plan, no step narration)",
          all(not any(x in reply for x in ("My plan", "[identify_patient]", "Booking:", "Continuing with"))
              for reply in replies), replies)
    check("every assistant reply has a 'Show details' expander, collapsed by default",
          all(m.expander[0].label == "Show details" and m.expander[0].proto.expanded is False
              for m in assistant))
    check("details include the plan, the steps and the memory",
          all(x in [md.value for md in assistant[-1].expander[0].markdown]
              for x in ("**Plan (sub-goals in order)**", "**Steps run so far**", "**Memory**")))
    check("no always-visible side panel on the chat page",
          "What the agent is doing" not in [h.value for h in at.subheader])

    go(at, "Appointments")
    table = at.dataframe[0].value
    check("Appointments page (Upcoming) lists the new booking for Ramesh Kumar on S004",
          ((table["slot_id"] == "S004") & (table["patient_name"] == "Ramesh Kumar")).any(), table)

    go(at, "Medical info")
    check("Medical info shows the latest summary with sources",
          "dialysis" in all_text(at) and "Sources" in all_text(at))

    go(at, "Memory & logs")
    text = all_text(at)
    check("short-term memory shows the conversation", "User: My 70-year-old father" in text, text[:300])
    check("full planning breakdown still on Memory & logs (plan + steps tables)",
          "**Plan (sub-goals in order)**" in text and "**Steps run so far**" in text
          and any(list(df.value.columns)[:2] == ["goal", "tool"] for df in at.dataframe))
    log_steps = set(at.dataframe[-1].value["tool_name"])
    check("tool log lists the agent steps",
          {"make_plan", "identify_patient", "confirm_booking"} <= log_steps, log_steps)
    check("long-term notes for P001 shown (booking + search)",
          any(len(df.value) == 2 for df in at.dataframe))
    disease_search_tool.requests.get = real_get


def test_chat_new_request_while_waiting_for_name():
    print("chat page: sample scenario sent while the agent waits for a name")
    ambiguous = json.dumps({"patient_name": None, "condition": None, "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": ""},
        {"goal": "History", "tool": "get_patient_history", "input": ""}]})
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[ambiguous, PLAN])
    at.run()
    at.chat_input[0].set_value("Can you show my father's medical history?").run()
    check("agent asks for the name", "What is the patient's full name or patient ID?" in all_text(at))
    at.selectbox[0].set_value(SCENARIO).run()
    [b for b in at.button if b.label == "Send scenario"][0].click().run()
    text = all_text(at)
    check("'Send scenario' -> treated as a new request, not as a name",
          "Name: Ramesh Kumar" in text
          and "could not find a patient matching 'My 70" not in text, text[-600:])
    steps = at.chat_message[-1].expander[0].dataframe[1].value
    check("  'Show details' records the switch to a new request", "new_request" in list(steps["tool"]))
    check("no errors", not at.exception, at.exception)


def buttons(at, label):
    return [b for b in at.button if b.label == label]


def user_messages(at):
    return [m.markdown[0].value for m in at.chat_message if m.proto.name == "user"]


def test_clickable_answers():
    print("chat page: closed questions are answered with buttons / dropdowns")
    data.reset_demo_data()   # fresh copy of the test data, so S004 is the first offer again
    real_get = disease_search_tool.requests.get
    disease_search_tool.requests.get = lambda url, params=None, timeout=None: FakeReply()
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[
        PLAN, "Shall I book this nephrology visit for Ramesh Kumar?",
        "Shall I book this time for Ramesh Kumar instead?", ANSWER])
    at.run()
    at.chat_input[0].set_value(SCENARIO).run()
    check("patient question -> a confirmation card with Yes / No buttons",
          buttons(at, "Yes") and buttons(at, "No") and "Patient ID: P001" in all_text(at)
          and "Is this the patient you're looking for?" in all_text(at))

    buttons(at, "Yes")[0].click().run()
    check("clicking shows only their choice as one line", user_messages(at)[-1] == "Yes", user_messages(at))
    check("old patient buttons are gone once answered", not buttons(at, "Yes"))
    check("booking question -> 'Yes, book it' / 'No' + a slot dropdown",
          buttons(at, "Yes, book it") and buttons(at, "No")
          and [s for s in at.selectbox if s.label == "Or choose another time"])
    reply = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("booking message has no 'Hi!' and shows the doctor only once",
          "Hi" not in reply and reply.count("Dr. Meera Shah") == 1, reply)

    dropdown = [s for s in at.selectbox if s.label == "Or choose another time"][0]
    check("slot options are readable, e.g. 'Thu 22 Oct, 10:30 – Dr. Meera Shah (Nephrology)'",
          "Thu 22 Oct, 10:30 – Dr. Meera Shah (Nephrology)" in dropdown.options, dropdown.options)
    dropdown.set_value("Thu 22 Oct, 10:30 – Dr. Meera Shah (Nephrology)").run()
    buttons(at, "Confirm")[0].click().run()
    check("choosing a slot -> one line: 'Slot: Thu 22 Oct, 10:30, Dr. Meera Shah'",
          user_messages(at)[-1] == "Slot: Thu 22 Oct, 10:30, Dr. Meera Shah", user_messages(at))
    check("  the chat does not list the other options (no slot dropdown text left in any message)",
          "Dr. Sarah Thompson" not in "\n".join(m.markdown[0].value for m in at.chat_message
                                               if m.proto.name == "assistant" and m.markdown), all_text(at)[:300])
    check("new preview for S005; only ONE set of booking buttons on the page",
          "(slot S005)" in all_text(at) and len(buttons(at, "Yes, book it")) == 1)

    buttons(at, "Yes, book it")[0].click().run()
    booked = read_csv(config.APPOINTMENTS_CSV)
    check("'Yes, book it' -> S005 booked for Ramesh Kumar",
          ((booked["slot_id"] == "S005") & (booked["patient_id"] == "P001")
           & (booked["status"] == "confirmed")).any())
    check("no question left -> no answer buttons", not buttons(at, "Yes, book it") and not at.exception,
          at.exception)
    disease_search_tool.requests.get = real_get

    # Specialty dropdown when the user did not say which kind of doctor.
    no_specialty = json.dumps({"patient_name": "Anita Sharma", "condition": None, "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Anita Sharma"},
        {"goal": "Book a doctor", "tool": "book_appointment", "input": ""}]})
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[no_specialty, "Shall I book this for Anita Sharma?"])
    at.run()
    at.chat_input[0].set_value("Book a doctor for Anita Sharma").run()
    buttons(at, "Yes")[0].click().run()
    dropdown = [s for s in at.selectbox if s.label == "Which kind of doctor?"]
    check("no specialty given -> specialty dropdown (10 specialties from doctors.csv)",
          dropdown and len(dropdown[0].options) == 10, dropdown and dropdown[0].options)
    dropdown[0].set_value("Cardiology").run()
    buttons(at, "Confirm")[0].click().run()
    check("choosing 'Cardiology' -> cardiology booking offered",
          user_messages(at)[-1] == "Specialty: Cardiology" and "Dr. Arjun Mehta" in all_text(at), user_messages(at))
    at.chat_input[0].set_value("no").run()
    check("typing 'no' still works instead of clicking",
          "won't book" in all_text(at) and not buttons(at, "Yes, book it"))


def test_greeting_in_chat():
    print("chat page: a greeting gets a friendly reply (no LLM call, no plan)")
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[])   # any LLM call would fail
    at.run()
    at.chat_input[0].set_value("Hi").run()
    reply = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("'Hi' -> short friendly reply with the option bubbles (no bullet list)",
          reply.startswith("Hello!") and "I can help with" not in reply
          and buttons(at, "Book an appointment") and buttons(at, "Register a new patient")
          and buttons(at, "Learn about a disease") and buttons(at, "Show a patient's history"), reply)
    check("  no error, no question waiting", not at.exception and not buttons(at, "Yes, book it"))
    details = [md.value for md in at.chat_message[-1].expander[0].markdown]
    check("  'Show details' has no plan (nothing was planned)", "**Plan (sub-goals in order)**" not in details)
    at.chat_input[0].set_value("Hi").run()
    reply2 = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("  saying 'Hi' again does not repeat the message",
          reply2 == "Pick one to continue" and buttons(at, "Book an appointment"), reply2)
    buttons(at, "Register a new patient")[0].click().run()
    reply3 = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("  clicking a bubble starts that task (registration asks for the name)",
          "full name" in reply3 and "Register a new patient" in user_messages(at) and not at.exception, reply3)


def test_registration_widgets():
    print("chat page: registering a patient with a number input and dropdowns")
    data.reset_demo_data()
    plan = json.dumps({"patient_name": "Grace Hopper", "condition": None, "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Grace Hopper"},
        {"goal": "History", "tool": "get_patient_history", "input": ""}]})
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[plan])
    at.run()
    at.chat_input[0].set_value("Show the medical history of Grace Hopper").run()
    check("not found -> 'Register a new patient' button", buttons(at, "Register a new patient"))
    buttons(at, "Register a new patient")[0].click().run()

    age = at.number_input[0]
    check("age is a NUMBER input (0 to 120), empty at first",
          age.label == "Age (years)" and age.proto.min == 0 and age.proto.max == 120 and age.value is None,
          (age.label, age.value))
    check("  the Confirm button is disabled until a number is typed", buttons(at, "Confirm")[0].proto.disabled)
    age.set_value(41).run()
    buttons(at, "Confirm")[0].click().run()
    check("age sent as the user's message 'Age: 41'", user_messages(at)[-1] == "Age: 41", user_messages(at))

    gender = [s for s in at.selectbox if s.label == "Gender"][0]
    check("gender is a dropdown: Male / Female / Other", list(gender.options) == ["Male", "Female", "Other"],
          gender.options)
    gender.set_value("Female").run()
    buttons(at, "Confirm")[0].click().run()

    contact = [s for s in at.selectbox if s.label == "Contact person"][0]
    check("contact person is a dropdown: self, son, daughter, wife, husband, other",
          list(contact.options) == ["Self", "Son", "Daughter", "Wife", "Husband", "Other"], contact.options)
    contact.set_value("Daughter").run()
    buttons(at, "Confirm")[0].click().run()
    at.chat_input[0].set_value("12345").run()
    check("a short phone number is rejected with a clear message",
          "doesn't look like a phone number" in all_text(at) and not at.exception)
    at.chat_input[0].set_value("+1 555 010 0188").run()
    check("preview shows all details; nothing saved yet",
          "Contact phone: +15550100188" in all_text(at) and "Name: Grace Hopper" in all_text(at) and "Age: 41" in all_text(at)
          and "Gender: Female" in all_text(at) and "Contact person: daughter" in all_text(at)
          and not (read_csv(config.PATIENTS_CSV)["name"] == "Grace Hopper").any())

    buttons(at, "Yes, save patient")[0].click().run()
    saved = read_csv(config.PATIENTS_CSV)
    saved = saved[saved["name"] == "Grace Hopper"].iloc[0]
    check("'Yes, save patient' -> saved: Grace Hopper, 41, Female, daughter",
          (saved["age"], saved["gender"], saved["contact_person"], saved["contact_phone"])
          == ("41", "Female", "daughter", "+15550100188"),
          saved.to_dict())
    check("original request continues: her history is shown",
          "medical history" in all_text(at) and not at.exception, at.exception)


def records_count():
    return len(read_csv(config.MEDICAL_RECORDS_CSV))


def test_records_page():
    print("records page: attendant form (preview, then confirm)")
    data.reset_demo_data()
    at = go(new_app(), "Records")
    before = records_count()
    expected_id = f"R{before + 1:03d}"

    at.selectbox(key="add_patient").set_value("Daniel Brooks (P012)")
    at.text_input(key="add_medications").set_value("Cetirizine 10mg")
    at.run()
    at.button(key="add_preview_button").click().run()
    check("empty diagnosis -> error from the tool, nothing to confirm",
          "Diagnosis cannot be empty" in all_text(at)
          and not [b for b in at.button if b.key == "add_confirm_button"])
    check("  nothing written", records_count() == before)

    at.text_input(key="add_diagnosis").set_value("Seasonal allergic rhinitis").run()
    at.button(key="add_preview_button").click().run()
    shown = [df.value for df in at.dataframe if "record_id" in df.value.columns]
    check(f"preview shows exactly what will be saved ({expected_id}, P012, ...)",
          shown and shown[0].iloc[0]["record_id"] == expected_id and shown[0].iloc[0]["patient_id"] == "P012"
          and shown[0].iloc[0]["diagnosis"] == "Seasonal allergic rhinitis", shown)
    check("  nothing written yet", records_count() == before)

    at.text_input(key="add_diagnosis").set_value("Something else").run()
    check("form changed after the preview -> warning, and NO confirm button",
          "The form changed after the preview" in all_text(at)
          and not [b for b in at.button if b.key == "add_confirm_button"])
    at.text_input(key="add_diagnosis").set_value("Seasonal allergic rhinitis").run()
    at.button(key="add_confirm_button").click().run()
    saved = read_csv(config.MEDICAL_RECORDS_CSV)
    check("confirm -> saved exactly the previewed record",
          records_count() == before + 1
          and saved.iloc[-1].to_dict() == {
              "record_id": expected_id, "patient_id": "P012", "visit_date": str(config.get_now().date()),
              "diagnosis": "Seasonal allergic rhinitis", "medications": "Cetirizine 10mg",
              "lab_results": "", "alerts": "", "doctor_notes": ""}, saved.iloc[-1].to_dict())
    check("  success message mentions the summary refresh", "stored summary refreshes" in all_text(at))

    # ---- update ----
    at.selectbox(key="update_patient").set_value("Ramesh Kumar (P001)").run()
    at.selectbox(key="update_record_P001").set_value(
        "R004 · 2026-04-20 · Chronic kidney disease stage 3b").run()
    check("record ID and patient ID are shown but disabled (cannot be edited)",
          at.text_input(key="update_rid_R004").proto.disabled and at.text_input(key="update_rid_R004").value == "R004"
          and at.text_input(key="update_pid_R004").proto.disabled
          and at.text_input(key="update_pid_R004").value == "P001")
    check("form is pre-filled with the saved values",
          at.text_input(key="update_medications_R004").value == "Losartan 100mg; sodium bicarbonate")

    at.button(key="update_preview_button_R004").click().run()
    check("no edits -> 'Nothing was changed'", "Nothing was changed" in all_text(at))

    original = read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").loc["R004"].to_dict()
    at.text_input(key="update_medications_R004").set_value("Losartan 100mg; sodium bicarbonate; furosemide 20mg")
    at.run()
    at.button(key="update_preview_button_R004").click().run()
    changes = [df.value for df in at.dataframe if "old value" in df.value.columns]
    check("preview shows old -> new for the ONE changed field",
          changes and list(changes[0]["field"]) == ["medications"]
          and changes[0].iloc[0]["old value"] == "Losartan 100mg; sodium bicarbonate"
          and "furosemide" in changes[0].iloc[0]["new value"], changes)
    check("  nothing written yet",
          read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").loc["R004"].to_dict() == original)
    at.button(key="update_confirm_button_R004").click().run()
    updated = read_csv(config.MEDICAL_RECORDS_CSV).set_index("record_id").loc["R004"].to_dict()
    check("confirm -> only that field changed; patient_id and record_id untouched",
          updated["medications"].endswith("furosemide 20mg") and updated["patient_id"] == "P001"
          and {k: v for k, v in updated.items() if k != "medications"}
          == {k: v for k, v in original.items() if k != "medications"})

    at.selectbox(key="update_record_P001").set_value("R003 · 2025-03-12 · Chronic kidney disease stage 3").run()
    at.text_input(key="update_diagnosis_R003").set_value("").run()
    at.button(key="update_preview_button_R003").click().run()
    check("clearing the diagnosis -> error, nothing to confirm",
          "Diagnosis cannot be empty" in all_text(at)
          and not [b for b in at.button if b.key == "update_confirm_button_R003"])
    check("no errors on the page", not at.exception, at.exception)


def test_waiting_list_on_appointments_page():
    print("waiting list is shown next to the appointment tracker")
    from tools.waitlist_tool import add_to_waitlist
    at = go(new_app(), "Appointments")
    check("empty list -> friendly caption, metric shows 0",
          "Nobody is on the waiting list." in all_text(at)
          and any(m.label == "On the waiting list" and m.value == "0" for m in at.metric))
    add_to_waitlist("P002", "Psychiatry", "anxiety", confirm=True)
    at = go(new_app(), "Appointments")
    tables = [df.value for df in at.dataframe]
    waiting = [t for t in tables if "waitlist_id" in t.columns]
    check("after an entry is added: a Waiting list table with the patient's name and specialty",
          len(waiting) == 1 and waiting[0].iloc[0]["patient_name"] == "Anita Sharma"
          and waiting[0].iloc[0]["specialty"] == "Psychiatry" and waiting[0].iloc[0]["doctor_name"] == "any doctor",
          tables)
    check("  the appointment tracker is still there, and the metric says 1",
          any("appointment_id" in t.columns for t in tables)
          and any(m.label == "On the waiting list" and m.value == "1" for m in at.metric))
    check("  no error", not at.exception)


def test_recheck_banner_phone_and_upgrade():
    print("recheck banner, phone on the patient view, older files get the new columns")
    from tools.csv_helpers import write_csv
    from tools.waitlist_tool import add_to_waitlist
    from tools.logger import read_logs

    waitlist_file = config.WAITLIST_CSV.read_text()
    write_csv(read_csv(config.WAITLIST_CSV).iloc[0:0], config.WAITLIST_CSV)       # start from an empty list
    add_to_waitlist("P002", "Psychiatry", "anxiety", doctor_id="D10", recheck_days=3, confirm=True)
    at = go(new_app(), "Health Assistant Helexa")
    check("recheck date still in the future -> no banner", "may have openings now" not in all_text(at))
    at = go(at, "Appointments")
    waiting = [df.value for df in at.dataframe if "waitlist_id" in df.value.columns][0]
    check("the dashboard waiting list shows recheck_on", waiting.iloc[0]["recheck_on"] == "2026-10-24"
          and "recheck_on" in waiting.columns, waiting)

    table = read_csv(config.WAITLIST_CSV)
    table.loc[0, "recheck_on"] = "2026-10-21"                       # today (APP_NOW) -> due
    write_csv(table, config.WAITLIST_CSV)
    at = go(new_app(), "Health Assistant Helexa")
    warnings = [w.value for w in at.warning]
    check("recheck_on today -> banner '<Doctor> may have openings now. Check availability?'",
          "Dr. Kavita Rao may have openings now. Check availability?" in warnings, warnings)
    check("  the banner is on the Appointments page too", "Dr. Kavita Rao may have openings now" in
          "\n".join(w.value for w in go(at, "Appointments").warning))
    log_start = len(read_logs())
    at.button(key="recheck_D10_check").click().run()
    check("'Check availability' runs the slot search (logged) and says there are no openings",
          "has no openings in the next 2 weeks" in all_text(at)
          and "list_available_slots" in [r["tool_name"] for r in read_logs()[log_start:]])
    slots = read_csv(config.SLOTS_CSV)
    slots.loc[len(slots)] = {"slot_id": "S902", "doctor_id": "D10", "date": "2026-10-23", "time": "10:00",
                             "status": "available"}
    write_csv(slots, config.SLOTS_CSV)
    at.button(key="recheck_D10_check").click().run()
    check("once a slot exists the banner shows it", any("S902" in df.value["slot_id"].tolist()
          for df in at.dataframe if "slot_id" in df.value.columns), all_text(at))
    write_csv(slots[slots["slot_id"] != "S902"], config.SLOTS_CSV)
    at.button(key="recheck_D10_dismiss").click().run()
    check("'Dismiss' hides the banner", "may have openings now" not in "\n".join(w.value for w in at.warning))

    write_csv(table.assign(doctor_id="", recheck_on="2026-10-01"), config.WAITLIST_CSV)
    at = go(new_app(), "Health Assistant Helexa")
    check("entry without a doctor -> '<Specialty> doctors may have openings now.'",
          "Psychiatry doctors may have openings now. Check availability?" in [w.value for w in at.warning])
    table.loc[0, "status"] = "booked"
    write_csv(table.assign(recheck_on="2026-10-01"), config.WAITLIST_CSV)
    check("entries that are no longer waiting give no banner",
          "may have openings now" not in "\n".join(w.value for w in go(new_app(), "Health Assistant Helexa").warning))
    config.WAITLIST_CSV.write_text(waitlist_file)

    # --- phone on the patient view ---
    at = go(new_app(), "Patients")
    check("Patients view: phone shows 'not on file' for an existing patient",
          [m.value for m in at.metric if m.label == "Contact phone"] == ["not on file"])
    patients_file = config.PATIENTS_CSV.read_text()
    people = read_csv(config.PATIENTS_CSV)
    people.loc[people["patient_id"] == "P001", "contact_phone"] = "+15550100777"      # a FAKE number
    write_csv(people, config.PATIENTS_CSV)
    at = go(new_app(), "Patients")
    check("  and the number when it is on file",
          [m.value for m in at.metric if m.label == "Contact phone"] == ["+15550100777"])
    config.PATIENTS_CSV.write_text(patients_file)

    # --- an older demo copy (no new columns) is upgraded, values untouched ---
    old_patients, old_waitlist = config.PATIENTS_CSV.read_text(), config.WAITLIST_CSV.read_text()
    write_csv(read_csv(config.PATIENTS_CSV).drop(columns=["contact_phone"]), config.PATIENTS_CSV)
    write_csv(read_csv(config.WAITLIST_CSV).drop(columns=["recheck_on"]), config.WAITLIST_CSV)
    data.upgrade_demo_files()
    upgraded = read_csv(config.PATIENTS_CSV)
    check("upgrade adds contact_phone after contact_person, blank, other columns unchanged",
          list(upgraded.columns) == list(read_csv(ORIGINAL_DATA_DIR / "patients.csv").columns)
          and (upgraded["contact_phone"] == "").all()
          and upgraded["name"].tolist() == read_csv(ORIGINAL_DATA_DIR / "patients.csv")["name"].tolist())
    check("  and recheck_on at the end of waitlist.csv", list(read_csv(config.WAITLIST_CSV).columns)[-1] == "recheck_on")
    data.upgrade_demo_files()
    check("  running it again changes nothing", read_csv(config.PATIENTS_CSV).equals(upgraded))
    config.PATIENTS_CSV.write_text(old_patients)
    config.WAITLIST_CSV.write_text(old_waitlist)


def test_closing_in_chat():
    print("chat: closing bubbles and the goodbye")
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[
        json.dumps({"steps": [{"goal": "Identify", "tool": "identify_patient", "input": "Anita Sharma"},
                              {"goal": "History", "tool": "get_patient_history", "input": ""}]})])
    at.run()
    at.chat_input[0].set_value("Show the medical history of Anita Sharma").run()
    buttons(at, "Yes")[0].click().run()
    check("after the finished task: 'Anything else?' and 'No, that's all' bubbles",
          buttons(at, "Anything else?") and buttons(at, "No, that's all"))
    buttons(at, "No, that's all")[0].click().run()
    reply = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("'No, that's all' -> 'Thank you! Visit again.' with a 'Start a new conversation' bubble",
          reply == "Thank you! Visit again." and buttons(at, "Start a new conversation") and not at.exception, reply)
    buttons(at, "Start a new conversation")[0].click().run()
    reply = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("'Start a new conversation' -> greeting and the menu bubbles",
          reply.startswith("Hello!") and buttons(at, "Book an appointment") and not at.exception, reply)


def test_choice_shown_as_one_line():
    print("chat: after answering a list question only the choice stays (options only in 'Show details')")
    from tools.csv_helpers import write_csv
    saved_file = config.PATIENTS_CSV.read_text()
    people = read_csv(config.PATIENTS_CSV)
    for pid, age in [("P013", "31"), ("P014", "52")]:       # two FAKE patients with the same name, test copy only
        people.loc[len(people)] = {"patient_id": pid, "name": "Sanskriti Prasad", "age": age, "gender": "Female",
                                   "contact_person": "self", "contact_phone": "", "created_date": "2026-01-01"}
    people.loc[len(people)] = {"patient_id": "P015", "name": "Ankit Prasad", "age": "28", "gender": "Male",
                               "contact_person": "self", "contact_phone": "", "created_date": "2026-01-01"}
    write_csv(people, config.PATIENTS_CSV)

    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[json.dumps({"steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Sanskriti Prasad"}]})])
    at.run()
    at.chat_input[0].set_value("Who is Sanskriti Prasad?").run()
    dropdown = [d for d in at.selectbox if d.label == "Which patient do you mean?"][0]
    check("two exact matches in the dropdown, 'None of these' last, no Ankit Prasad",
          list(dropdown.options) == ["Sanskriti Prasad, age 31, Female (P013)",
                                     "Sanskriti Prasad, age 52, Female (P014)", "None of these"], dropdown.options)
    asked = [m for m in at.chat_message if m.proto.name == "assistant"][-1]
    check("the question message does not list the options", "P013" not in asked.markdown[0].value
          and "age 31" not in asked.markdown[0].value, asked.markdown[0].value)
    dropdown.set_value("Sanskriti Prasad, age 52, Female (P014)").run()
    buttons(at, "Confirm")[0].click().run()
    check("answering leaves one line: 'Patient: Sanskriti Prasad (P014)'",
          user_messages(at)[-1] == "Patient: Sanskriti Prasad (P014)", user_messages(at))
    check("  the dropdown is gone from the chat", not [d for d in at.selectbox if d.label == "Which patient do you mean?"]
          and not at.exception)
    assistant = [m for m in at.chat_message if m.proto.name == "assistant"]
    details = [df.value for df in assistant[0].expander[0].dataframe]
    offered = [t for t in details if "option" in t.columns]
    check("'Show details' of the question keeps the full option list",
          len(offered) == 1 and len(offered[0]) == 3 and offered[0]["option"].tolist()[-1].endswith("None of these"),
          details)
    config.PATIENTS_CSV.write_text(saved_file)


def test_welcome_screen():
    print("chat: the empty welcome screen shows the four bubbles, all in the same style")
    labels = ["Show a patient's history", "Book an appointment", "Register a new patient", "Learn about a disease"]
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[])
    at.run()
    shown = [b for b in at.button if b.label in labels]
    check("before anything is typed: the four bubbles are shown", [b.label for b in shown] == labels, [b.label for b in at.button])
    check("  welcome text uses the name Helexa", "Hello! I'm Helexa. What would you like to do?" in all_text(at))
    check("  all four have the same style (none is highlighted)",
          len({b.proto.type for b in shown}) == 1 and shown[0].proto.type != "primary", [b.proto.type for b in shown])
    shown[2].click().run()
    check("clicking one starts that task (registration asks for the name)",
          user_messages(at)[-1] == "Register a new patient" and "full name" in all_text(at) and not at.exception)
    check("  the welcome bubbles are gone once the chat has started",
          not [b for b in at.button if b.label == "Learn about a disease"], [b.label for b in at.button])

    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["llm"] = FakeListChatModel(responses=[])
    at.run()
    at.chat_input[0].set_value("Hi").run()
    greeting = [m.markdown[0].value for m in at.chat_message if m.proto.name == "assistant"][-1]
    bubbles = [b for b in at.button if b.label in labels]
    check("a typed 'Hi' -> exactly 'Hello! I'm Helexa. What would you like to do?'", greeting
          == "Hello! I'm Helexa. What would you like to do?", greeting)
    check("  with one set of four bubbles, all the same style",
          [b.label for b in bubbles] == labels and len({b.proto.type for b in bubbles}) == 1
          and bubbles[0].proto.type != "primary", [(b.label, b.proto.type) for b in bubbles])


def test_cancel_and_reset():
    print("cancel with confirmation, then reset demo data")
    at = go(new_app(), "Appointments")
    at.selectbox[0].set_value(at.selectbox[0].options[0]).run()   # first upcoming appointment
    chosen = at.selectbox[0].value.split(":")[0]
    before = read_csv(config.APPOINTMENTS_CSV).set_index("appointment_id").at[chosen, "status"]
    [b for b in at.button if b.label == "Preview cancellation"][0].click().run()
    check(f"preview shown for {chosen}, nothing changed yet",
          "About to cancel" in all_text(at)
          and read_csv(config.APPOINTMENTS_CSV).set_index("appointment_id").at[chosen, "status"] == before)
    [b for b in at.button if b.label == "Confirm cancellation"][0].click().run()
    check(f"after confirming, {chosen} is cancelled",
          read_csv(config.APPOINTMENTS_CSV).set_index("appointment_id").at[chosen, "status"] == "cancelled")

    sidebar_button = lambda label: [b for b in at.sidebar.button if b.label == label]  # noqa: E731
    sidebar_button("Reset demo data")[0].click().run()
    check("'Reset demo data' button -> asks 'Yes, reset' / 'Cancel' first",
          sidebar_button("Yes, reset") and sidebar_button("Cancel"))
    sidebar_button("Cancel")[0].click().run()
    check("'Cancel' -> nothing reset, question gone",
          not sidebar_button("Yes, reset")
          and read_csv(config.APPOINTMENTS_CSV).set_index("appointment_id").at[chosen, "status"] == "cancelled")
    sidebar_button("Reset demo data")[0].click().run()
    sidebar_button("Yes, reset")[0].click().run()
    check("reset demo data -> CSVs equal the originals again",
          fingerprint(config.DATA_DIR) == fingerprint(ORIGINAL_DATA_DIR))


def test_evaluation_page():
    print("evaluation page")
    at = go(new_app(), "Evaluation")
    summary, results = data.latest_evaluation()
    if summary is None:
        check("no results yet -> friendly message", "No evaluation results yet" in all_text(at))
        return
    labels = [m.label for m in at.metric]
    check("headline tiles: precision, booking success, refusals, booking checks",
          {"Answer precision", "Booking success rate", "Correct refusals", "Booking checks"} <= set(labels), labels)
    check("per-question table has one row per question",
          len(at.dataframe[0].value) == summary["questions"])
    check("judge reasoning available per question", len(at.expander) == summary["questions"])


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    build_patient_index(llm=FakeListChatModel(responses=FAKE_SUMMARIES))

    test_every_page_loads()
    test_patient_and_doctor_views()
    test_assistant_scenario_and_tracking()
    test_chat_new_request_while_waiting_for_name()
    test_clickable_answers()
    test_greeting_in_chat()
    test_registration_widgets()
    test_records_page()
    test_waiting_list_on_appointments_page()
    test_recheck_banner_phone_and_upgrade()
    test_closing_in_chat()
    test_welcome_screen()
    test_choice_shown_as_one_line()
    test_cancel_and_reset()
    test_evaluation_page()

    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed (fake LLM, no internet).")
