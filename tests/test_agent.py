"""
test_agent.py - tests for the planner, patient identification, RAG and the agent.

Run from the project root:
    python -m tests.test_agent

Offline and free:
  - the LLM is FakeListChatModel, which returns our scripted replies in order
  - MedlinePlus is replaced by the saved real reply in tests/fixtures/
  - embeddings are the real local model
"now" is frozen at 2026-10-21 12:00, so the earliest nephrology slot is S004.
"""

import json
import os
import re
from pathlib import Path

import requests
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from agent.agent import (HealthcareAgent, interpret_reply, clean_name_reply,  # noqa: E402
                         looks_like_name_or_id, looks_like_new_request, parse_registration_value,
                         WAITING_FOR_NAME, nice_date)
from agent.patient_identifier import identify_patient  # noqa: E402
from agent.planner import Plan, PlanStep, extract_patient_facts, validate_plan, make_plan  # noqa: E402
from agent.disease_rag import summarize_disease_info  # noqa: E402
from memory.conversation_memory import recall_patient_memory  # noqa: E402
from memory.vector_store import build_patient_index  # noqa: E402
from tests.fake_data import FAKE_SUMMARIES  # noqa: E402
from tools import disease_search_tool  # noqa: E402
from tools.csv_helpers import read_csv, write_csv  # noqa: E402
from tools.logger import read_logs  # noqa: E402

CKD_XML = (Path(__file__).parent / "fixtures" / "medlineplus_ckd_sample.xml").read_text(encoding="utf-8")
SCENARIO = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist "
            "for him. Also, can you summarize latest treatment methods?")


def plan_json(steps, name=None, condition="chronic kidney disease"):
    return json.dumps({"patient_name": name, "condition": condition,
                       "steps": [{"goal": g, "tool": t, "input": i} for g, t, i in steps]})


SCENARIO_PLAN = plan_json([
    ("Identify the patient", "identify_patient", ""),
    ("Retrieve the patient's medical history", "get_patient_history", ""),
    ("Book a nephrologist appointment", "book_appointment", "Nephrology"),
    ("Summarize the latest treatment methods for chronic kidney disease", "search_disease_info",
     "chronic kidney disease"),
])
BOOKING_MESSAGE = ("I can book Ramesh Kumar with Dr. Meera Shah (Nephrology, City Care Hospital) on "
                   "2026-10-21 at 14:00. Other options are S005 and S006. Reply 'yes', a slot ID, or 'no'.")
DISEASE_ANSWER = ("- Treatments cannot cure kidney disease, but they may slow it [1].\n"
                  "- They include medicines to lower blood pressure, control blood sugar, and lower cholesterol [1].\n"
                  "- If the kidneys fail, dialysis or a kidney transplant is needed [1].\n"
                  "- The sources do not cover the newest research.")


def fake_llm(*responses):
    return FakeListChatModel(responses=list(responses))


def medlineplus_returns(text):
    class Reply:
        status_code = 200
        def __init__(self): self.text = text
        def raise_for_status(self): pass
    disease_search_tool.requests.get = lambda url, params=None, timeout=None: Reply()


def medlineplus_offline():
    def fail(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("no internet")
    disease_search_tool.requests.get = fail


def mentions_any_patient(text):
    """True if the text contains any patient's name or a patient ID like P003."""
    names = read_csv(config.PATIENTS_CSV)["name"]
    return any(n.lower() in text.lower() for n in names) or bool(re.search(r"\bP\d{3}\b", text))


def appointment(slot_id):
    a = read_csv(config.APPOINTMENTS_CSV)
    rows = a[(a["slot_id"] == slot_id) & (a["status"] == "confirmed")]
    return rows.iloc[0].to_dict() if len(rows) else None


# ---------------------------------------------------------------------------

def test_extract_facts():
    print("extract_patient_facts (rules)")
    cases = [
        (SCENARIO, {"age": 70, "gender": "male"}),
        ("My mother, aged 62, has diabetes", {"age": 62, "gender": "female"}),
        ("my 45 year old husband can't sleep", {"age": 45, "gender": "male"}),
        ("She is 34 years old", {"age": 34, "gender": "female"}),
        ("I am 40 years old and my 70-year-old father is ill", {"age": None, "gender": "male"}),
        ("I am 40 and my 70-year-old father is ill", {"age": 70, "gender": "male"}),
        ("My father and mother both have asthma", {"age": None, "gender": None}),
        ("Please book a cardiologist for him", {"age": None, "gender": "male"}),
        ("Book a dermatologist", {"age": None, "gender": None}),
    ]
    for text, expected in cases:
        check(f"{text[:45]!r} -> {expected}", extract_patient_facts(text) == expected,
              extract_patient_facts(text))


def test_validate_plan():
    print("validate_plan (order rules)")
    plan = Plan(steps=[PlanStep(goal="Book", tool="book_appointment", input="Nephrology"),
                       PlanStep(goal="Search", tool="search_disease_info", input="ckd")])
    tools = [s.tool for s in validate_plan(plan).steps]
    check("booking without identify/history -> both added in front",
          tools == ["identify_patient", "get_patient_history", "book_appointment", "search_disease_info"], tools)

    plan = Plan(steps=[PlanStep(goal="Id", tool="identify_patient"),
                       PlanStep(goal="Search", tool="search_disease_info", input="ckd")])
    tools = [s.tool for s in validate_plan(plan, patient_known=True).steps]
    check("patient already known + no patient step -> identify removed", tools == ["search_disease_info"], tools)

    plan = Plan(steps=[PlanStep(goal="a", tool="search_disease_info", input="CKD"),
                       PlanStep(goal="b", tool="search_disease_info", input="ckd ")])
    check("duplicate steps removed", len(validate_plan(plan).steps) == 1)


def test_reply_parsing():
    print("yes/no and name replies (rules)")
    for text, expected in [("yes", "yes"), ("Yes please", "yes"), ("go ahead", "yes"), ("OK", "yes"),
                           ("no", "no"), ("yes but not now", "no"), ("Don't book", "no"),
                           ("maybe later", None), ("S006", None)]:
        check(f"{text!r} -> {expected}", interpret_reply(text) == expected, interpret_reply(text))
    for text, expected in [("His name is Ramesh Kumar.", "Ramesh Kumar"), ("Ramesh Kumar", "Ramesh Kumar"),
                           ("It's Rakesh Kumar", "Rakesh Kumar"), ("patient id: P001", "P001")]:
        check(f"name reply {text!r} -> {expected!r}", clean_name_reply(text) == expected, clean_name_reply(text))


def test_identify_patient():
    print("identify_patient (3 steps)")
    r = identify_patient(SCENARIO, facts=extract_patient_facts(SCENARIO))
    check("scenario -> exactly one: Ramesh Kumar (P001) -> needs_confirmation",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P001", r["message"])
    check("  a confirmation card: name, age, gender, ID, contact person, then the question",
          all(x in r["question"] for x in ("Name: Ramesh Kumar", "Age: 70", "Gender: Male", "Patient ID: P001",
                                           "Contact person: son"))
          and r["question"].endswith("Is this the patient you're looking for?"), r["question"])
    check("  the card has no phone number and no medical information",
          "phone" not in r["question"].lower() and "kidney" not in r["question"].lower()
          and "Losartan" not in r["question"], r["question"])

    text = "My father has kidney disease"
    r = identify_patient(text, facts=extract_patient_facts(text))
    check("no age given -> several men match -> needs_name (no guessing)",
          r["status"] == "needs_name" and r["candidate_count"] > 1, r["message"])
    check("  PRIVACY: asks only the fixed question, no candidate list",
          r["question"] == "What is the patient's full name or patient ID?" and "candidates" not in r, r)
    check("  PRIVACY: no patient names or IDs in the question or the logged message",
          not mentions_any_patient(r["question"] + r["message"]), r)

    r = identify_patient("x", name="Kumar", facts={"age": 48, "gender": "male"})
    check("name 'Kumar' + age 48 -> Rakesh Kumar (P011)",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P011", r["message"])
    r = identify_patient("x", name="Kumar", facts={})
    check("name 'Kumar', no facts -> both Kumars -> needs_choice (pick by ID)",
          r["status"] == "needs_choice" and len(r["candidates"]) == 2, r["message"])
    r = identify_patient("x", name="Zzz Nobody", facts={})
    check("typed name matches nobody -> says so, asks the fixed question, lists nobody",
          r["status"] == "needs_name" and "'Zzz Nobody'" in r["question"]
          and r["question"].endswith("What is the patient's full name or patient ID?")
          and not mentions_any_patient(r["question"]), r["question"])
    r = identify_patient("x", name="Ramesh Kumar", facts={})
    check("full name 'Ramesh Kumar', no facts -> lists Ramesh and Rakesh, asks for ID (no loop)",
          r["status"] == "needs_choice" and [p["patient_id"] for p in r["candidates"]] == ["P001", "P011"]
          and "Which one do you mean?" in r["question"]
          and [p.get("match_label", "") for p in r["candidates"]] == ["", "Similar name"], r["question"])
    r = identify_patient("My 99-year-old father has kidney disease", facts={"age": 99, "gender": "male"})
    check("age 99 -> nobody -> needs_name, same fixed question",
          r["status"] == "needs_name" and r["candidate_count"] == 0
          and r["question"] == "What is the patient's full name or patient ID?", r)
    # Typed name EXACTLY matches a patient, but the age does not fit.
    r = identify_patient("x", name="Rakesh Kumar", facts={"age": 70, "gender": "male"})
    check("exact name 'Rakesh Kumar' + age 70 -> needs_choice",
          r["status"] == "needs_choice", r["message"])
    check("  shows both: the name match (P011) first, then the age match (P001)",
          [p["patient_id"] for p in r["candidates"]] == ["P011", "P001"], r["candidates"])
    check("  the exact-name patient carries a note, the look-alike (who fits) does not",
          r["candidates"][0]["note"].startswith("does not match age 70")
          and r["candidates"][1]["note"] == "" and r["candidates"][1]["match_label"] == "Similar name",
          [(p["patient_id"], p["note"]) for p in r["candidates"]])
    check("  the question never lists the options (they are in the dropdown)",
          "P011" not in r["question"] and "P001" not in r["question"], r["question"])
    r = identify_patient("x", name="P011", facts={"age": 70})
    check("exact ID 'P011' + age 70 -> one patient: card with a note (never dropped silently)",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P011"
          and "does not match age 70" in r["question"], r["question"])
    r = identify_patient("x", name="Daniel Brooks", facts={"age": 70})
    check("exact name, nobody fits the age -> still asked, with the note",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P012"
          and "does not match age 70" in r["question"], r["message"])

    # Typed name matches NOBODY exactly (a typo): keep the earlier behaviour.
    r = identify_patient("x", name="Ramesh Kumaar", facts={"age": 70, "gender": "male"})
    check("typo 'Ramesh Kumaar' + age 70 -> offers Ramesh with 'spelled differently' note",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P001"
          and "spelled differently" in r["question"], r["question"])
    r = identify_patient("x", name="Rakesh Kumar", facts={"age": 48})
    check("exact name and the age fits -> normal confirmation (no choice)",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P011", r["message"])


def test_clean_search_topic():
    print("search term cleaning (MedlinePlus wants the disease name only)")
    from agent.disease_rag import clean_search_topic
    for topic, expected in [("chronic kidney disease latest treatments", "chronic kidney disease"),
                            ("asthma treatment options", "asthma"), ("insomnia treatment", "insomnia"),
                            ("What are the treatment options for zorblaxitis?", "zorblaxitis"),
                            ("chronic kidney disease", "chronic kidney disease"),
                            ("latest treatments", "latest treatments")]:
        check(f"{topic!r} -> {expected!r}", clean_search_topic(topic) == expected, clean_search_topic(topic))
    captured = {}
    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        raise requests.exceptions.ConnectionError("offline")
    disease_search_tool.requests.get = fake_get
    summarize_disease_info("chronic kidney disease latest treatments", "latest treatment methods",
                           llm=fake_llm("unused [1]"))
    check("MedlinePlus is searched with the cleaned term",
          captured.get("term") == "chronic kidney disease", captured)


def test_disease_rag():
    print("summarize_disease_info (RAG)")
    medlineplus_returns(CKD_XML)
    r = summarize_disease_info("chronic kidney disease", "latest treatment methods",
                               llm=fake_llm(DISEASE_ANSWER))
    check("success, LLM summary kept (valid citations)",
          r["status"] == "success" and r["summary_type"] == "llm_summary", r["message"])
    check("  source link added by code",
          "https://medlineplus.gov/chronickidneydisease.html" in r["answer"] and "Sources" in r["answer"], r["answer"])
    check("  passages came from the MedlinePlus text",
          all(p["text"] in CKD_XML or p["link"].startswith("https://medlineplus.gov/") for p in r["passages"]))

    r = summarize_disease_info("chronic kidney disease", "treatments",
                               llm=fake_llm("CKD is treated with a new wonder drug."))
    check("answer without citations -> replaced by quoted MedlinePlus passages",
          r["summary_type"] == "quoted" and "wonder drug" not in r["answer"], r["answer"][:120])
    r = summarize_disease_info("chronic kidney disease", "treatments",
                               llm=fake_llm("Something [7]."))
    check("citation to a passage that does not exist -> quoted instead", r["summary_type"] == "quoted")

    medlineplus_offline()
    r = summarize_disease_info("asthma", "treatments", llm=fake_llm("should not be used [1]"))
    check("MedlinePlus unreachable -> no_connection, no summary invented",
          r["status"] == "no_connection" and r["answer"] == "" and "will not summarize" in r["message"], r)


def test_full_scenario():
    print("full sample scenario (3 turns)")
    medlineplus_returns(CKD_XML)
    agent = HealthcareAgent(llm=fake_llm(SCENARIO_PLAN, BOOKING_MESSAGE, DISEASE_ANSWER))
    log_start = len(read_logs())

    r1 = agent.handle(SCENARIO)
    check("turn 1: plan has the 4 sub-goals in order",
          [s["tool"] for s in r1["plan"]] == ["identify_patient", "get_patient_history",
                                              "book_appointment", "search_disease_info"], r1["plan"])
    check("turn 1: asks to confirm Ramesh Kumar",
          r1["waiting_for"] == "confirm_patient" and "Patient ID: P001" in r1["reply"] and "Name: Ramesh Kumar" in r1["reply"], r1["reply"])
    data_before = fingerprint(TEST_DATA_DIR)

    r2 = agent.handle("Yes, that's him")
    check("turn 2: patient remembered in short-term memory", r2["patient_id"] == "P001")
    check("turn 2: history summary shown with alerts",
          "summary of Ramesh Kumar's medical history" in r2["reply"] and "Penicillin allergy" in r2["reply"], r2["reply"])
    check("turn 2: waits for booking confirmation of S004 with Dr. Meera Shah",
          r2["waiting_for"] == "confirm_booking" and "(slot S004)" in r2["reply"]
          and "Dr. Meera Shah" in r2["reply"], r2["reply"])
    check("turn 2: nothing written before the user says yes", fingerprint(TEST_DATA_DIR) == data_before)

    r3 = agent.handle("yes")
    tools_logged = [row["tool_name"] for row in read_logs()[log_start:]]
    check("turn 3: booked", "I've booked" in r3["reply"] and r3["waiting_for"] is None, r3["reply"])
    booked = appointment("S004")
    check("  appointment.csv has P001 + S004 confirmed, slots.csv says booked",
          booked and booked["patient_id"] == "P001"
          and read_csv(config.SLOTS_CSV).set_index("slot_id").at["S004", "status"] == "booked", booked)
    check("turn 3: treatment summary with source link",
          "dialysis" in r3["reply"] and "https://medlineplus.gov/chronickidneydisease.html" in r3["reply"], r3["reply"])
    check("turn 3: disclaimer shown", config.DISCLAIMER in r3["reply"])

    notes = recall_patient_memory("P001", limit=None)["notes"]
    check("long-term memory: booking and search notes saved",
          [n["type"] for n in notes] == ["booking", "search"], notes)
    check("trace has every step with a status",
          [t["tool"] for t in r3["trace"]] == ["plan", "identify_patient", "confirm_patient", "get_patient_history",
                                               "book_appointment", "confirm_booking", "search_disease_info"]
          and all(t["status"] for t in r3["trace"]), r3["trace"])
    check("log: exactly one row per agent step",
          tools_logged == ["make_plan", "identify_patient", "retrieve_patient_history", "prepare_booking",
                           "confirm_booking", "summarize_disease_info"], tools_logged)

    # Follow-up in the same conversation: patient already known, no re-identification.
    agent.llm = fake_llm(plan_json([("Book a cardiologist", "book_appointment", "Cardiology")], condition=None),
                         "Shall I book Ramesh Kumar with Dr. Arjun Mehta on 2026-10-21 at 11:00?")
    r4 = agent.handle("Please also book a cardiologist for him")
    check("follow-up: no new identification, history then cardiology booking offered",
          "identify_patient" not in [s["tool"] for s in r4["plan"]]
          and r4["waiting_for"] == "confirm_booking" and "Dr. Arjun Mehta" in r4["reply"], r4["reply"])
    before = fingerprint(TEST_DATA_DIR)
    r5 = agent.handle("no")
    check("follow-up: 'no' -> nothing booked", "won't book" in r5["reply"]
          and fingerprint(TEST_DATA_DIR) == before, r5["reply"])


def test_wrong_patient_then_name():
    print("user says it's the wrong patient, then gives a name")
    agent = HealthcareAgent(llm=fake_llm(plan_json([("Identify", "identify_patient", ""),
                                                    ("History", "get_patient_history", "")])))
    agent.handle("My 70-year-old father has kidney disease, show me his history")
    r = agent.handle("no")
    check("'no' -> asks for the name", r["waiting_for"] == "patient_name" and "name" in r["reply"], r["reply"])
    r = agent.handle("His name is Ramesh Kumar")
    check("name given -> asks to confirm Ramesh Kumar (Rakesh filtered out by age)",
          r["waiting_for"] == "confirm_patient" and "P001" in r["reply"] and "P011" not in r["reply"], r["reply"])
    r = agent.handle("yes")
    check("confirmed -> history shown, plan finished",
          "summary of Ramesh Kumar's medical history" in r["reply"] and r["waiting_for"] is None, r["reply"])


def test_name_age_mismatch_choice():
    print("typed name matches one patient, age matches another -> user chooses")
    history_plan = plan_json([("Identify", "identify_patient", "Rakesh Kumar"),
                              ("History", "get_patient_history", "")], name="Rakesh Kumar")
    request = "Show the history of my 70-year-old father Rakesh Kumar"

    agent = HealthcareAgent(llm=fake_llm(history_plan))
    r = agent.handle(request)
    check("agent lists both and waits for a choice",
          r["waiting_for"] == "choose_patient" and "P011" not in r["reply"] and "P001" not in r["reply"]
          and [o["reply"] for o in r["question"]["select"]["options"]] == ["P011", "P001", "neither"], r["question"])
    r = agent.handle("maybe")
    check("unclear reply -> asks again (pick from the list)",
          r["waiting_for"] == "choose_patient" and "pick one from the list" in r["reply"], r["reply"])
    r = agent.handle("P011")
    check("reply 'P011' -> continues with Rakesh Kumar (the exact name match)",
          r["patient_id"] == "P011" and "summary of Rakesh Kumar's medical history" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(history_plan))
    agent.handle(request)
    r = agent.handle("I mean p1")
    check("reply 'p1' -> continues with Ramesh Kumar (the age match)",
          r["patient_id"] == "P001" and "summary of Ramesh Kumar's medical history" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(history_plan))
    agent.handle(request)
    r = agent.handle("P005")
    check("an ID that was not offered -> not accepted, asks again",
          r["waiting_for"] == "choose_patient" and r["patient_id"] is None, r["reply"])
    r = agent.handle("neither")
    check("'neither' -> asks for the name", r["waiting_for"] == "patient_name", r["reply"])


def test_ambiguous_asks_for_name():
    print("ambiguous request -> asks for the name")
    agent = HealthcareAgent(llm=fake_llm(plan_json([("Identify", "identify_patient", ""),
                                                    ("History", "get_patient_history", "")], condition=None)))
    r = agent.handle("Can you show my father's medical history?")
    check("no age -> several candidates -> asks for name (no guess)",
          r["waiting_for"] == "patient_name" and "full name" in r["reply"], r["reply"])
    check("  PRIVACY: the reply shows no other patients' names or IDs",
          not mentions_any_patient(r["reply"])
          and r["reply"].endswith("What is the patient's full name or patient ID?"), r["reply"])
    check("  PRIVACY: the step trace (shown on the dashboard) has no names or IDs either",
          not any(mentions_any_patient(t["message"]) for t in r["trace"]), r["trace"])
    r = agent.handle("cancel")
    check("'cancel' stops the request", r["waiting_for"] is None and "cancelled" in r["reply"], r["reply"])


def test_choose_other_slot():
    print("user picks another slot")
    medlineplus_returns(CKD_XML)
    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", ""), ("History", "get_patient_history", ""),
                   ("Book", "book_appointment", "Nephrology")]),
        "Shall I book Dr. Meera Shah on 2026-10-22 at 10:30?",
        "Shall I book Dr. Meera Shah on 2026-10-23 at 15:30?"))
    agent.handle(SCENARIO)
    r = agent.handle("yes")
    check("first offer is the earliest free slot S005 (S004 now taken)", "(slot S005)" in r["reply"], r["reply"])
    r = agent.handle("S003")
    check("past slot S003 -> refused with the reason, still waiting",
          "in the past" in r["reply"] and r["waiting_for"] == "confirm_booking", r["reply"])
    r = agent.handle("S013")
    check("cardiology slot S013 -> refused: not a Nephrology slot",
          "not a Nephrology slot" in r["reply"] and r["waiting_for"] == "confirm_booking", r["reply"])
    r = agent.handle("S006 please")
    check("'S006' -> new preview for S006", r["waiting_for"] == "confirm_booking" and "(slot S006)" in r["reply"], r["reply"])
    r = agent.handle("yes")
    check("booked S006, not S005", appointment("S006") is not None and appointment("S005") is None, r["reply"])


def test_fully_booked_and_offline():
    print("psychiatrist fully booked + MedlinePlus offline")
    medlineplus_offline()
    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Anita Sharma"), ("History", "get_patient_history", ""),
                   ("Book psychiatrist", "book_appointment", "Psychiatry"),
                   ("Info on anxiety", "search_disease_info", "generalized anxiety disorder")],
                  name="Anita Sharma", condition="anxiety")))
    agent.handle("Book a psychiatrist for Anita Sharma and tell me about anxiety treatment")
    r = agent.handle("yes")
    check("no psychiatry slots -> told clearly, with the four option bubbles",
          "Dr. Kavita Rao (Psychiatry) has no openings in the next 2 weeks." in r["reply"]
          and "booked" not in r["reply"] and r["waiting_for"] == "no_slots"
          and [b["label"] for b in r["question"]["buttons"]]
          == ["Join the waiting list", "Choose a different specialty", "Back to start"], r["reply"])
    r = agent.handle("Back to start")
    check("'Back to start' -> the rest of the plan still runs, then the menu shows",
          r["waiting_for"] is None and r["question"]["type"] == "menu", r["reply"])
    check("MedlinePlus offline -> says so, no invented treatment info",
          "Could not reach MedlinePlus" in r["reply"] and "will not summarize" in r["reply"], r["reply"])


AMBIGUOUS_PLAN = plan_json([("Identify", "identify_patient", ""),
                            ("History", "get_patient_history", "")], condition=None)


def patient_count():
    return len(read_csv(config.PATIENTS_CSV))


def test_reply_classification():
    print("is the reply a name, a new request, or neither? (rules)")
    for text, expected in [("Ramesh Kumar", True), ("His name is Anita Sharma", True), ("P001", True),
                           ("p 1", True), ("create a new log", False), (SCENARIO, False),
                           ("123", False), ("book a doctor", False), ("show me the logs", False)]:
        check(f"looks_like_name_or_id({text[:30]!r}) -> {expected}", looks_like_name_or_id(text) == expected)
    for text, expected in [("create a new log", True), (SCENARIO, True), ("what now?", True),
                           ("123", False), ("hmm", False)]:
        check(f"looks_like_new_request({text[:30]!r}) -> {expected}", looks_like_new_request(text) == expected)
    for field, text, expected in [("age", "41 years", 41), ("age", "seventy", None), ("age", "150", None),
                                  ("gender", "She is female", "Female"), ("gender", "m", "Male"),
                                  ("gender", "xyz", None), ("contact_person", "Daughter", "daughter"),
                                  ("name", "Grace Hopper", "Grace Hopper"), ("name", "P005", None),
                                  ("contact_phone", "+1 555-010-0123", "+15550100123"),
                                  ("contact_phone", "(555) 010 0123", "5550100123"),
                                  ("contact_phone", "12345", None), ("contact_phone", "abc", None),
                                  ("contact_phone", "5550100123456789", None), ("contact_phone", "", None)]:
        check(f"registration {field} {text!r} -> {expected!r}",
              parse_registration_value(field, text)[0] == expected, parse_registration_value(field, text))


def test_not_a_name_while_waiting():
    print("waiting for a name, but the reply is not a name")
    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN, SCENARIO_PLAN, BOOKING_MESSAGE))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle(SCENARIO)
    check("the sample scenario sentence -> handled as a NEW request (not searched as a name)",
          r["trace"][0]["tool"] == "new_request" and r["waiting_for"] == "confirm_patient"
          and "Name: Ramesh Kumar" in r["reply"]
          and "could not find a patient" not in r["reply"], r["reply"])
    check("  the new plan replaced the old one",
          [s["tool"] for s in r["plan"]][2:] == ["book_appointment", "search_disease_info"], r["plan"])

    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN, json.dumps({"steps": []})))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle("create a new log")
    check("'create a new log' -> new request; planner finds no task, nothing is waiting",
          r["trace"][0]["tool"] == "new_request" and "I didn't catch that" in r["reply"]
          and r["waiting_for"] is None, r["reply"])

    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle("12ab")
    check("unclear reply '12ab' -> says what it is waiting for and offers the options",
          r["reply"] == WAITING_FOR_NAME and "'new patient'" in r["reply"] and "'cancel'" in r["reply"]
          and r["waiting_for"] == "patient_name", r["reply"])
    r = agent.handle("cancel")
    check("'cancel' -> starts over", r["waiting_for"] is None and "cancelled" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Ramesh Kumar"), ("History", "get_patient_history", "")],
                  name="Ramesh Kumar"), json.dumps({"steps": []})))
    agent.handle("Show the medical history of Ramesh Kumar")
    r = agent.handle("please create a new log for me")
    check("also while choosing between similar names -> new request",
          r["trace"][0]["tool"] == "new_request" and r["waiting_for"] is None, r["reply"])


def test_register_new_patient():
    print("patient not found -> register a new patient, then continue")
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Grace Hopper"), ("History", "get_patient_history", "")],
        name="Grace Hopper", condition=None)))
    count = patient_count()
    r = agent.handle("Show the medical history of Grace Hopper")
    check("not found -> offers 'new patient' for 'Grace Hopper' and 'cancel'",
          "could not find a patient matching 'Grace Hopper'" in r["reply"]
          and "say 'new patient' to register 'Grace Hopper'" in r["reply"] and "'cancel'" in r["reply"], r["reply"])
    r = agent.handle("new patient")
    check("'new patient' -> name already known, asks the age",
          r["waiting_for"] == "register_patient" and "How old" in r["reply"], r["reply"])
    r = agent.handle("seventy")
    check("invalid age -> explains and asks again", "number between 0 and 120" in r["reply"]
          and "How old" in r["reply"], r["reply"])
    r = agent.handle("41")
    check("-> asks the gender", "gender" in r["reply"], r["reply"])
    r = agent.handle("xyz")
    check("invalid gender -> asks again", "male, female or other" in r["reply"], r["reply"])
    r = agent.handle("female")
    check("-> asks the contact person", "contact person" in r["reply"], r["reply"])
    r = agent.handle("Self")
    check("-> asks the contact phone number", "phone number" in r["reply"] and "7 to 15 digits" in r["reply"], r["reply"])
    for bad in ["call me", "12345", "+1 (555) 010-01x3"]:
        r = agent.handle(bad)
        check(f"invalid phone {bad!r} -> clear message, asks again", "doesn't look like a phone number" in r["reply"]
              and "digits only, 7 to 15 digits" in r["reply"] and "contact phone number" in r["reply"]
              and agent.pending["fields"]["contact_phone"] is None, r["reply"])
    r = agent.handle("+1 555 010 0123")
    check("all details -> preview of exactly what will be saved",
          "Nothing is saved until you say yes" in r["reply"] and "Patient ID: P013" in r["reply"]
          and "Name: Grace Hopper" in r["reply"] and "Age: 41" in r["reply"]
          and "Gender: Female" in r["reply"] and "Contact person: self" in r["reply"]
          and "Contact phone: +15550100123" in r["reply"], r["reply"])
    check("  nothing saved before 'yes'", patient_count() == count)
    r = agent.handle("no")
    check("'no' -> asks which detail to change", "Which detail should change" in r["reply"], r["reply"])
    r = agent.handle("phone 5550100999")
    check("'phone ...' changes the phone number in the preview",
          "Contact phone: 5550100999" in r["reply"] and agent.pending["asking"] == "confirm", r["reply"])
    r = agent.handle("no")
    r = agent.handle("age 42")
    check("'age 42' -> new preview with age 42", "Age: 42" in r["reply"]
          and agent.pending["asking"] == "confirm", r["reply"])
    check("  still nothing saved", patient_count() == count)
    r = agent.handle("yes")
    saved = read_csv(config.PATIENTS_CSV).set_index("patient_id").loc["P013"]
    check("'yes' -> saved with add_patient: P013 Grace Hopper, 42, Female, self, phone",
          (saved["name"], saved["age"], saved["gender"], saved["contact_person"], saved["contact_phone"])
          == ("Grace Hopper", "42", "Female", "self", "5550100999"), saved.to_dict())
    check("  the phone number is not repeated in the history reply", "5550100999" not in r["reply"], r["reply"])
    check("  original request continues: history step for the new patient",
          r["patient_id"] == "P013" and "summary of Grace Hopper's medical history" in r["reply"]
          and "No medical records" in r["reply"] and r["waiting_for"] is None, r["reply"])


def test_register_with_known_details_then_book():
    print("registration reuses details from the request, then the booking continues")
    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Zara Khan"),
                   ("Book a nephrologist", "book_appointment", "Nephrology")], name="Zara Khan",
                  condition="kidney check"),
        "Shall I book Zara Khan with Dr. Meera Shah on 2026-10-22 at 10:30?"))
    agent.handle("Book a nephrologist for my 66-year-old mother Zara Khan")
    r = agent.handle("please register her as a new patient")
    check("age 66 and gender female taken from the request -> only the contact person is asked",
          "contact person" in r["reply"] and agent.pending["fields"]["age"] == 66
          and agent.pending["fields"]["gender"] == "Female", r["reply"])
    r = agent.handle("daughter")
    check("then the phone number is asked", "phone number" in r["reply"], r["reply"])
    r = agent.handle("5550100456")
    check("preview shows the details from the request", "Age: 66" in r["reply"]
          and "Gender: Female" in r["reply"] and "Contact person: daughter" in r["reply"], r["reply"])
    r = agent.handle("yes")
    check("saved as P014, then the booking is offered for her",
          r["patient_id"] == "P014" and r["waiting_for"] == "confirm_booking"
          and "Zara Khan (P014)" in r["reply"], r["reply"])
    agent.handle("yes")
    booked = read_csv(config.APPOINTMENTS_CSV)
    check("booking saved for the new patient",
          ((booked["patient_id"] == "P014") & (booked["status"] == "confirmed")).any())


def test_duplicate_registration_rejected():
    print("registering a duplicate is rejected")
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Bob Stone"), ("History", "get_patient_history", "")],
        name="Bob Stone", condition=None)))
    count = patient_count()
    agent.handle("Show the medical history of Bob Stone")
    agent.handle("new patient")
    for reply in ["70", "male", "son", "5550100789"]:
        agent.handle(reply)
    agent.handle("no")
    r = agent.handle("name Ramesh Kumar")
    check("same name + age as P001 -> duplicate message, nothing saved",
          "already exists as P001" in r["reply"] and "Nothing was saved" in r["reply"]
          and patient_count() == count and r["waiting_for"] == "patient_name", r["reply"])
    r = agent.handle("cancel")
    check("'cancel' -> starts over", r["waiting_for"] is None and "cancelled" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Bob Stone")], name="Bob Stone", condition=None)))
    agent.handle("Show the medical history of Bob Stone")
    agent.handle("new patient")
    r = agent.handle("cancel")
    check("'cancel' in the middle of registration -> stops, nothing saved",
          r["waiting_for"] is None and patient_count() == count, r["reply"])


INTERNAL_MARKERS = ["My plan", "[identify_patient]", "[get_patient_history]", "[book_appointment]",
                    "[search_disease_info]", "Booking:", "Continuing with", "Skipped", "needs_confirmation",
                    "make_plan", "treat that as a new request"]


def shows_internals(reply):
    return [m for m in INTERNAL_MARKERS if m in reply]


def test_natural_replies():
    print("replies are for the user: no plan, no tool names, no log lines")
    medlineplus_returns(CKD_XML)
    agent = HealthcareAgent(llm=fake_llm(SCENARIO_PLAN, BOOKING_MESSAGE, DISEASE_ANSWER))
    replies = [agent.handle(SCENARIO), agent.handle("yes"), agent.handle("S007"), agent.handle("yes")]
    for number, r in enumerate(replies, 1):
        check(f"turn {number}: no internals in the reply", not shows_internals(r["reply"]),
              shows_internals(r["reply"]))
    check("turn 1 is just the question", replies[0]["reply"].startswith("Name: Ramesh Kumar")
          and replies[0]["reply"].endswith("Is this the patient you're looking for?"), replies[0]["reply"])
    check("the plan is still returned separately (for 'Show details')",
          [s["tool"] for s in replies[0]["plan"]][0] == "identify_patient")
    check("the steps are still returned separately (trace)",
          [t["tool"] for t in replies[-1]["trace"]][:3] == ["plan", "identify_patient", "confirm_patient"],
          replies[-1]["trace"])
    check("booking reply is a plain sentence",
          "Done! I've booked Ramesh Kumar with Dr. Meera Shah (Nephrology, City Care Hospital) on "
          "2026-10-26 at 09:30" in replies[-1]["reply"], replies[-1]["reply"])
    check("disclaimer only at the end, after medical information",
          config.DISCLAIMER in replies[-1]["reply"] and config.DISCLAIMER not in replies[1]["reply"])

    agent = HealthcareAgent(llm=fake_llm(plan_json([("Identify", "identify_patient", "Anita Sharma")],
                                                   name="Anita Sharma", condition=None)))
    agent.handle("Is Anita Sharma one of your patients?")
    r = agent.handle("yes")
    check("nothing to report after the steps -> short friendly reply, no disclaimer",
          r["reply"] == "Got it. Is there anything else I can help with?", r["reply"])

    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Priya Patel"), ("Book", "book_appointment", "Dermatology")],
                  name="Priya Patel", condition="skin check"),
        "Shall I book Priya Patel with Dr. Priya Nair on 2026-10-21 at 16:00?"))
    agent.handle("Book a dermatologist for Priya Patel")
    agent.handle("yes")
    r = agent.handle("no")
    check("declining -> 'Okay, I won't book anything.' and nothing about logs",
          r["reply"].startswith("Okay, I won't book anything.") and not shows_internals(r["reply"]), r["reply"])


def slot(slot_id, date, time, doctor, specialty, clinic="City Care Hospital"):
    return {"slot_id": slot_id, "date": date, "time": time, "doctor_name": doctor,
            "specialty": specialty, "clinic": clinic}


def replies_of(options):
    return [o["reply"] for o in options]


def test_pending_question_mapping():
    print("pending question -> options (for clickable widgets)")
    agent = HealthcareAgent(llm=fake_llm("unused"))
    check("nothing pending -> no question", agent.pending_question() is None)

    agent.pending = {"type": "confirm_patient", "patient": {"patient_id": "P001"}}
    q = agent.pending_question()
    check("confirm_patient -> two buttons sending 'yes' / 'no'",
          replies_of(q["buttons"]) == ["yes", "no"] and q["select"] is None
          and [b["label"] for b in q["buttons"]] == ["Yes", "No"], q)

    agent.pending = {"type": "choose_patient", "candidates": [
        {"patient_id": "P001", "name": "Ramesh Kumar", "age": "70", "gender": "Male"},
        {"patient_id": "P011", "name": "Rakesh Kumar", "age": "48", "gender": "Male",
         "match_label": "Similar name", "note": "does not match age 70"}]}
    q = agent.pending_question()
    check("choose_patient -> dropdown with name, age, ID + 'None of these'",
          [o["label"] for o in q["select"]["options"]]
          == ["Ramesh Kumar, age 70, Male (P001)",
              "Rakesh Kumar, age 48, Male (P011) - Similar name - does not match age 70", "None of these"]
          and replies_of(q["select"]["options"]) == ["P001", "P011", "neither"]
          and q["select"]["submit"] == "Confirm", q)
    check("  chat message for a choice is ONE short line: 'Patient: name (ID)'",
          q["select"]["options"][0]["display"] == "Patient: Ramesh Kumar (P001)"
          and q["select"]["options"][2]["display"] == "None of these")

    agent.pending = {"type": "confirm_booking", "reason": "x",
                     "slot": slot("S004", "2026-10-21", "14:00", "Dr. Meera Shah", "Nephrology"),
                     "offered": [slot("S004", "2026-10-21", "14:00", "Dr. Meera Shah", "Nephrology"),
                                 slot("S005", "2026-10-22", "10:30", "Dr. Meera Shah", "Nephrology"),
                                 slot("S023", "2026-10-26", "09:30", "Dr. Sarah Thompson", "General Medicine")]}
    q = agent.pending_question()
    check("confirm_booking -> 'Yes, book it' / 'No' buttons",
          [(o["label"], o["reply"]) for o in q["buttons"]] == [("Yes, book it", "yes"), ("No", "no")], q)
    check("  + dropdown of the OTHER slots in readable form (current slot left out)",
          [o["label"] for o in q["select"]["options"]]
          == ["Thu 22 Oct, 10:30 – Dr. Meera Shah (Nephrology)",
              "Mon 26 Oct, 09:30 – Dr. Sarah Thompson (General Medicine)"], q["select"])
    check("  choosing a slot sends its ID and shows one line: 'Slot: Mon 26 Oct, 09:30, Dr. Sarah Thompson'",
          q["select"]["options"][1]["reply"] == "S023"
          and q["select"]["options"][1]["display"] == "Slot: Mon 26 Oct, 09:30, Dr. Sarah Thompson",
          q["select"]["options"][1])

    agent.pending = {"type": "choose_specialty"}
    q = agent.pending_question()
    expected = sorted(read_csv(config.DOCTORS_CSV)["specialty"].unique())
    check("choose_specialty -> dropdown of every specialty in doctors.csv",
          replies_of(q["select"]["options"]) == expected and len(expected) == 10, q)

    agent.pending = {"type": "patient_name"}
    q = agent.pending_question()
    check("patient_name -> name is typed; buttons for 'new patient' and 'cancel'",
          replies_of(q["buttons"]) == ["new patient", "cancel"] and q["select"] is None, q)

    agent.pending = {"type": "register_patient", "fields": {}, "asking": "age"}
    q = agent.pending_question()
    check("registration, age -> a NUMBER input (0 to 120), no buttons or dropdown",
          q["number"] == {"label": "Age (years)", "min": 0, "max": 120, "submit": "Confirm",
                          "display": "Age: {value}"} and not q["buttons"] and q["select"] is None, q)
    agent.pending["asking"] = "gender"
    q = agent.pending_question()
    check("registration, gender -> dropdown Male / Female / Other",
          [o["label"] for o in q["select"]["options"]] == ["Male", "Female", "Other"]
          and replies_of(q["select"]["options"]) == ["male", "female", "other"]
          and q["number"] is None and not q["buttons"], q)
    agent.pending["asking"] = "contact_person"
    q = agent.pending_question()
    check("registration, contact person -> dropdown self, son, daughter, wife, husband, other",
          replies_of(q["select"]["options"]) == ["self", "son", "daughter", "wife", "husband", "other"]
          and [o["label"] for o in q["select"]["options"]][:2] == ["Self", "Son"], q)
    agent.pending["asking"] = "confirm"
    check("registration, preview -> 'Yes, save patient' / 'No, change a detail'",
          [(o["label"], o["reply"]) for o in agent.pending_question()["buttons"]]
          == [("Yes, save patient", "yes"), ("No, change a detail", "no")])
    agent.pending["asking"] = "name"
    check("registration, name -> typed (no widget)", agent.pending_question() is None)
    agent.pending["asking"] = "contact_phone"
    check("registration, phone -> typed (no widget)", agent.pending_question() is None)
    agent.pending["asking"] = "edit"
    check("registration, 'which detail to change' -> typed (no widget)", agent.pending_question() is None)


def test_pending_question_round_trips():
    print("each option's reply is understood by the agent")
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Ramesh Kumar"), ("History", "get_patient_history", "")],
        name="Ramesh Kumar", condition=None)))
    r = agent.handle("Show the medical history of Ramesh Kumar")
    check("handle() returns the structured question", r["question"]["type"] == "choose_patient", r["question"])
    r = agent.handle(r["question"]["select"]["options"][0]["reply"])
    check("picking the first patient option -> Ramesh Kumar's history",
          r["patient_id"] == "P001" and r["question"]["type"] == "closing", r["reply"])

    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Sunita Verma"),
                   ("Book", "book_appointment", "Endocrinology")], name="Sunita Verma", condition="diabetes check"),
        "Hi! Shall I book this diabetes check for Sunita Verma?",
        "Shall I book this one for Sunita Verma instead?"))
    agent.handle("Book an endocrinologist for Sunita Verma")   # no other test books Endocrinology
    r = agent.handle("yes")
    first_slot = agent.pending["slot"]["slot_id"]
    check("booking reply: no 'Hi!', and the details appear only once",
          "\n\nShall I book this diabetes check for Sunita Verma?\n" in r["reply"] and "Hi" not in r["reply"]
          and r["reply"].count("Dr. Rohan Verma") == 1 and r["reply"].count(f"(slot {first_slot})") == 1,
          r["reply"])
    other = r["question"]["select"]["options"][0]
    r = agent.handle(other["reply"])
    check("choosing another slot from the dropdown -> new preview for that slot",
          r["waiting_for"] == "confirm_booking" and agent.pending["slot"]["slot_id"] == other["reply"], r["reply"])
    r = agent.handle(r["question"]["buttons"][0]["reply"])
    booked = read_csv(config.APPOINTMENTS_CSV)
    check("'Yes, book it' -> that slot is booked",
          ((booked["slot_id"] == other["reply"]) & (booked["patient_id"] == "P004")
           & (booked["status"] == "confirmed")).any(), r["reply"])


def test_specialty_question():
    print("no specialty given -> ask (never assume General Medicine)")
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Anita Sharma"), ("Book", "book_appointment", "")],
        name="Anita Sharma", condition=None), "Shall I book this for Anita Sharma?"))
    agent.handle("Book a doctor for Anita Sharma")
    r = agent.handle("yes")
    check("empty specialty -> asks which kind of doctor (dropdown)",
          r["waiting_for"] == "choose_specialty" and "Which kind of doctor" in r["reply"]
          and r["question"]["select"]["options"], r["reply"])
    r = agent.handle("Cardiology")
    check("'Cardiology' -> cardiology booking offered", r["waiting_for"] == "confirm_booking"
          and "Dr. Arjun Mehta" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Anita Sharma"), ("Book", "book_appointment", "General Medicine")],
        name="Anita Sharma", condition=None)))
    agent.handle("Book a doctor for Anita Sharma")
    r = agent.handle("yes")
    check("planner guessed General Medicine without being asked -> still asks",
          r["waiting_for"] == "choose_specialty", r["reply"])
    r = agent.handle("a nephrologist please")
    check("typed 'a nephrologist please' -> Nephrology", r["waiting_for"] == "confirm_booking"
          and "Dr. Meera Shah" in r["reply"], r["reply"])

    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Anita Sharma"), ("Book", "book_appointment", "General Medicine")],
        name="Anita Sharma", condition=None), "Shall I book this for Anita Sharma?"))
    agent.handle("Book a general physician for Anita Sharma")
    r = agent.handle("yes")
    check("user asked for a general physician -> General Medicine used, no question",
          r["waiting_for"] == "confirm_booking" and "Dr. Sarah Thompson" in r["reply"], r["reply"])


def test_booking_sentence_check():
    print("the LLM's booking sentence is checked")
    from agent.agent import clean_booking_sentence
    for text, expected in [("Hi! Shall I book this for Ramesh Kumar?", "Shall I book this for Ramesh Kumar?"),
                           ("Hello, shall I go ahead for Ramesh?", "Shall I go ahead for Ramesh?"),
                           ("Shall I book Ramesh on 2026-10-21 at 14:00?", None),
                           ("", None), ("x" * 250, None)]:
        check(f"{text[:40]!r} -> {expected!r}", clean_booking_sentence(text) == expected,
              clean_booking_sentence(text))


class NoLLM:
    """Stands in for the LLM and fails loudly if anything calls it."""
    def invoke(self, *args, **kwargs):
        raise AssertionError("the LLM must not be called for this message")

    def __ror__(self, other):          # lets "prompt | llm" build a chain
        raise AssertionError("the LLM must not be used for this message")


def psychiatry_agent(*extra):
    """An agent that has asked about a psychiatrist for Anita Sharma, confirmed her, and been told
    there are no openings (Dr. Kavita Rao has none)."""
    agent = HealthcareAgent(llm=fake_llm(
        plan_json([("Identify", "identify_patient", "Anita Sharma"), ("History", "get_patient_history", ""),
                   ("Book psychiatrist", "book_appointment", "Psychiatry")],
                  name="Anita Sharma", condition="anxiety"), *extra))
    agent.handle("Book a psychiatrist for Anita Sharma")
    return agent, agent.handle("yes")


def waitlist_rows():
    return read_csv(config.WAITLIST_CSV)


def test_no_slots_and_waiting_list():
    print("no open slots: message, bubbles, waiting list")
    anita = read_csv(config.PATIENTS_CSV).query("name == 'Anita Sharma'").iloc[0]["patient_id"]
    labels = ["Join the waiting list", "Choose a different specialty", "Back to start"]
    appointments_before = len(read_csv(config.APPOINTMENTS_CSV))
    agent, r = psychiatry_agent()
    check("message is the always-true sentence, then the question",
          "Dr. Kavita Rao (Psychiatry) has no openings in the next 2 weeks." in r["reply"]
          and "All upcoming slots are booked" not in r["reply"], r["reply"])
    check("  three bubbles + a 'Remind me' dropdown (3/7/14 days), no text menu and no 'Check again'", [b["label"] for b in r["question"]["buttons"]] == labels
          and [o["label"] for o in r["question"]["select"]["options"]]
          == ["Remind me in 3 days", "Remind me in 7 days", "Remind me in 14 days"]
          and "waiting list" not in r["reply"].lower() and "Check again" not in str(r["question"]), r["reply"])

    r = agent.handle("Join the waiting list")
    check("'Join the waiting list' -> preview + Yes/No, nothing saved yet",
          r["waiting_for"] == "confirm_waitlist" and "Anita Sharma" in r["reply"] and "Psychiatry" in r["reply"]
          and [b["reply"] for b in r["question"]["buttons"]] == ["yes", "no"] and len(waitlist_rows()) == 0, r["reply"])
    r = agent.handle("no")
    check("'no' -> nothing saved, the four bubbles come back",
          len(waitlist_rows()) == 0 and r["waiting_for"] == "no_slots"
          and [b["label"] for b in r["question"]["buttons"]] == labels, r["reply"])
    agent.handle("Join the waiting list")
    r = agent.handle("yes")
    saved = waitlist_rows()
    check("  'Join the waiting list' reminds in 7 days by default", saved.iloc[0]["recheck_on"] == "2026-10-28"
          and "Recheck on 2026-10-28" in r["reply"], saved.iloc[0].to_dict())
    check("'yes' -> saved to waitlist.csv", len(saved) == 1 and saved.iloc[0]["patient_id"] == anita
          and saved.iloc[0]["specialty"] == "Psychiatry" and saved.iloc[0]["status"] == "waiting"
          and "waiting list for Psychiatry" in r["reply"] and r["waiting_for"] is None, r["reply"])
    check("  no appointment was booked", len(read_csv(config.APPOINTMENTS_CSV)) == appointments_before)

    agent, r = psychiatry_agent()
    r = agent.handle("Join the waiting list")
    check("joining twice is rejected (no duplicate row)", len(waitlist_rows()) == 1
          and "already on the waiting list" in r["reply"], r["reply"])

    # --- the reminder dropdown ---
    for days, due in [(3, "2026-10-24"), (14, "2026-11-04")]:
        waitlist = waitlist_rows()
        write_csv(waitlist[waitlist["patient_id"] != anita], config.WAITLIST_CSV)
        agent, r = psychiatry_agent()
        reply = [o["reply"] for o in r["question"]["select"]["options"] if f"{days} days" in o["label"]][0]
        r = agent.handle(reply)
        check(f"dropdown 'Remind me in {days} days' -> preview mentions the reminder date, nothing saved yet",
              r["waiting_for"] == "confirm_waitlist" and len(waitlist_rows()) == 0
              and nice_date(due) in r["reply"], r["reply"])
        r = agent.handle("yes")
        check("  'yes' saves recheck_on = today + %d days" % days,
              waitlist_rows().iloc[-1]["recheck_on"] == due and f"Recheck on {due}" in r["reply"], r["reply"])
        check("  the task is finished: 'Anything else?' bubbles", r["question"]["type"] == "closing", r["question"])

    # --- the other bubbles ---
    agent, r = psychiatry_agent()
    r = agent.handle("Choose a different specialty")
    check("'Choose a different specialty' -> specialty dropdown", r["waiting_for"] == "choose_specialty"
          and r["question"]["select"], r["reply"])
    r = agent.handle("Nephrology")
    check("  picking one continues the booking", r["waiting_for"] == "confirm_booking", r["reply"])

    agent, r = psychiatry_agent()
    r = agent.handle("Back to start")
    check("'Back to start' -> menu bubbles, nothing saved",
          r["waiting_for"] is None and r["question"]["type"] == "menu" and len(waitlist_rows()) == 1)
    agent, r = psychiatry_agent()
    r = agent.handle("what is asthma?")
    check("a new request while choosing is handled as a new request", r["waiting_for"] != "no_slots", r["reply"])


def test_other_doctor_offered_first():
    print("another doctor of the same specialty has openings")
    doctors, slots = read_csv(config.DOCTORS_CSV), read_csv(config.SLOTS_CSV)
    doctors.loc[len(doctors)] = {"doctor_id": "D11", "name": "Dr. Sam Test", "specialty": "Psychiatry",
                                 "clinic": "Test Clinic", "experience_years": "5", "available_days": "Mon-Fri"}
    write_csv(doctors, config.DOCTORS_CSV)
    slots.loc[len(slots)] = {"slot_id": "S901", "doctor_id": "D11", "date": "2026-10-23", "time": "11:00",
                             "status": "available"}
    write_csv(slots, config.SLOTS_CSV)
    agent, r = psychiatry_agent(BOOKING_MESSAGE)
    check("the empty doctor is named, then the other doctor's slot is offered",
          "Dr. Kavita Rao (Psychiatry) has no openings in the next 2 weeks." in r["reply"]
          and "Dr. Sam Test" in r["reply"] and "S901" in r["reply"]
          and r["reply"].index("no openings") < r["reply"].index("S901")
          and r["waiting_for"] == "confirm_booking", r["reply"])
    write_csv(doctors[doctors["doctor_id"] != "D11"], config.DOCTORS_CSV)
    write_csv(slots[slots["slot_id"] != "S901"], config.SLOTS_CSV)


def test_small_talk():
    print("greetings get a short fixed reply plus the option bubbles")
    from agent.agent import is_small_talk, GREETING_REPLY, MENU
    for text, expected in [("Hi", True), ("hello!", True), ("Hey there", True), ("Good morning", True),
                           ("thanks", True), ("Thank you", True), ("What can you do?", True),
                           ("who are you", True), ("help", True), ("Hi, book a cardiologist", False),
                           ("What can you do for asthma?", False), ("", False), ("Ramesh Kumar", False)]:
        check(f"is_small_talk({text!r}) -> {expected}", is_small_talk(text) == expected)

    log_start = len(read_logs())
    agent = HealthcareAgent(llm=NoLLM())
    r = agent.handle("Hi")
    check("the greeting says who Helexa is", GREETING_REPLY == "Hello! I'm Helexa. What would you like to do?")
    check("'Hi' -> fixed greeting (no LLM call), no bullet list",
          r["reply"] == GREETING_REPLY and "\n" not in r["reply"] and "- " not in r["reply"], r["reply"])
    check("  the options are bubbles instead", [b["label"] for b in r["question"]["buttons"]]
          == [label for label, _ in MENU], r["question"])
    check("  nothing is waiting, no plan was made",
          r["waiting_for"] is None and r["plan"] == [])
    check("  the step is recorded as scope_guard, and no planner call was logged",
          [t["tool"] for t in r["trace"]] == ["scope_guard"]
          and "make_plan" not in [row["tool_name"] for row in read_logs()[log_start:]], r["trace"])

    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle("Hi")
    check("'Hi' while waiting for a name is NOT searched as a name",
          r["reply"] == WAITING_FOR_NAME and r["waiting_for"] == "patient_name"
          and "could not find" not in r["reply"], r["reply"])


def category_plan(category):
    """A planner reply that only classifies the message (no steps)."""
    return json.dumps({"category": category, "steps": []})


def test_scope_guard():
    print("scope guard: off-topic, personal advice, vague input, cancel words")
    from agent.agent import (OFF_TOPIC_REPLY, ADVICE_REPLY, UNCLEAR_REPLY, CANCELLED_REPLY, SORRY_REPLY,
                             is_cancel_request, asks_personal_advice, MENU)
    labels = [label for label, _ in MENU]

    def bubbles(r):
        return [b["label"] for b in (r["question"] or {"buttons": []})["buttons"]]

    # --- classified by the planner (LLM) -> fixed template, no steps ---
    for category, expected in [("off_topic", OFF_TOPIC_REPLY), ("medical_advice", ADVICE_REPLY),
                               ("unclear", UNCLEAR_REPLY)]:
        agent = HealthcareAgent(llm=fake_llm(category_plan(category)))
        r = agent.handle("some message")
        check(f"planner says {category} -> exactly the fixed template", r["reply"] == expected, r["reply"])
        check("  with the option bubbles, nothing waiting, no steps run",
              bubbles(r) == labels and r["waiting_for"] is None and r["plan"] == []
              and [t["tool"] for t in r["trace"]] == ["plan", "scope_guard"], r["trace"])
    check("  fixed wording: off-topic", OFF_TOPIC_REPLY == "Sorry, that's outside what I can help with. "
          "I can help with patient records, appointments and disease information.")
    check("  fixed wording: advice", ADVICE_REPLY == "I can't give personal medical advice. I can show "
          "general information from MedlinePlus, or help with an appointment.")
    check("  fixed wording: unclear", UNCLEAR_REPLY == "I didn't catch that. What would you like to do?")

    # steps the LLM adds to a non-in-scope message are ignored
    agent = HealthcareAgent(llm=fake_llm(json.dumps({"category": "off_topic", "steps": [
        {"goal": "g", "tool": "book_appointment", "input": "Cardiology"}]})))
    r = agent.handle("tell me a joke")
    check("an off-topic message with steps still runs nothing", r["reply"] == OFF_TOPIC_REPLY and r["plan"] == [])

    # a plan with no category (older style) is still in scope; no steps -> unclear
    agent = HealthcareAgent(llm=fake_llm(json.dumps({"steps": []})))
    r = agent.handle("asdf qwerty zxcv")
    check("in scope but nothing to do -> the 'didn't catch that' reply", r["reply"] == UNCLEAR_REPLY, r["reply"])

    # --- personal medical advice is caught by rules too (no LLM call) ---
    for text in ["How much ibuprofen should I take?", "What is the right dosage of metformin?",
                 "Do I have diabetes?", "Can you diagnose my rash?"]:
        agent = HealthcareAgent(llm=NoLLM())
        r = agent.handle(text)
        check(f"advice rule: {text!r}", r["reply"] == ADVICE_REPLY and bubbles(r) == labels, r["reply"])
    check("general disease questions are not advice", not asks_personal_advice(
        "What are the treatment options for asthma?"))

    # --- cancel words ---
    for text in ["cancel", "never mind", "Nevermind", "too late", "forget it"]:
        check(f"is_cancel_request({text!r})", is_cancel_request(text))
        agent = HealthcareAgent(llm=NoLLM())
        r = agent.handle(text)
        check(f"  {text!r} -> 'I've cancelled this request.' + bubbles",
              r["reply"] == CANCELLED_REPLY and bubbles(r) == labels, r["reply"])
    for text in ["the patient is dead", "he died yesterday", "She has passed away"]:
        agent = HealthcareAgent(llm=NoLLM())
        r = agent.handle(text)
        check(f"death mention {text!r} -> cancelled + sorry", r["reply"] == f"{CANCELLED_REPLY} {SORRY_REPLY}",
              r["reply"])
    check("a long request containing 'too late' is not cancelled", not is_cancel_request(
        "Book a cardiologist for Arvind Iyer, I think it is not too late for a checkup this week"))
    check("'cancel my appointment' (no open question) is a real request, not a drop",
          not is_cancel_request("cancel my appointment"))

    # cancel ends whatever flow is open
    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle("too late")
    check("cancel while waiting for a name -> flow ended, nothing waiting",
          r["reply"] == CANCELLED_REPLY and r["waiting_for"] is None and r["plan"] == [], r["reply"])
    before = patient_count()
    agent = HealthcareAgent(llm=fake_llm(plan_json([("Identify", "identify_patient", "")]),))
    agent.handle("Register a new patient")
    r = agent.handle("never mind")
    check("cancel during registration -> nothing saved",
          r["reply"] == CANCELLED_REPLY and r["waiting_for"] is None and patient_count() == before, r["reply"])
    r = agent.handle("the patient is dead")
    check("death mention with nothing open -> still cancelled + sorry", SORRY_REPLY in r["reply"])

    # cancel while confirming a booking: no booking, and the decline is logged
    appointments_before = len(read_csv(config.APPOINTMENTS_CSV))
    agent = HealthcareAgent(llm=fake_llm(SCENARIO_PLAN, BOOKING_MESSAGE))
    agent.handle(SCENARIO)
    agent.handle("yes")
    check("(setup) waiting to confirm a booking", agent.pending and agent.pending["type"] == "confirm_booking")
    r = agent.handle("never mind, too late")
    check("cancel while confirming a booking -> nothing booked, flow ended",
          r["reply"] == CANCELLED_REPLY and r["waiting_for"] is None
          and len(read_csv(config.APPOINTMENTS_CSV)) == appointments_before, r["reply"])

    # --- the menu bubbles lead somewhere ---
    agent = HealthcareAgent(llm=NoLLM())
    r = agent.handle("Register a new patient")
    check("bubble 'Register a new patient' starts registration without the LLM",
          r["waiting_for"] == "register_patient" and "full name" in r["reply"], r["reply"])
    agent = HealthcareAgent(llm=NoLLM())
    r = agent.handle("Learn about a disease")
    check("bubble 'Learn about a disease' asks which disease", r["waiting_for"] == "ask_disease"
          and "Which disease" in r["reply"], r["reply"])
    r = agent.handle("cancel")
    check("  and 'cancel' drops it", r["reply"] == CANCELLED_REPLY and r["waiting_for"] is None)


def test_digit_patient_id():
    print("waiting for a patient: '007' or '7' means P007")
    for typed in ["007", "7"]:
        agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
        agent.handle("Can you show my father's medical history?")
        r = agent.handle(typed)
        patient = read_csv(config.PATIENTS_CSV).set_index("patient_id").loc["P007"]
        check(f"{typed!r} -> 'Did you mean P007, <name>, age <age>?'",
              r["reply"] == f"Did you mean P007, {patient['name']}, age {patient['age']}?"
              and r["waiting_for"] == "confirm_patient", r["reply"])
        check("  Yes/No buttons, and no other patient is mentioned",
              [b["reply"] for b in r["question"]["buttons"]] == ["yes", "no"]
              and r["reply"].count("P0") == 1, r["reply"])
        r = agent.handle("yes")
        check("  'yes' confirms the patient", r["patient_id"] == "P007", r["reply"])
    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
    agent.handle("Can you show my father's medical history?")
    r = agent.handle("999")
    check("a number that matches nobody -> asks for the name again, lists nobody",
          "couldn't find a patient with ID P999" in r["reply"] and "full name or patient ID" in r["reply"]
          and r["waiting_for"] == "patient_name" and r["reply"].count("P0") == 0, r["reply"])


def test_no_repeated_replies():
    print("the same bot message is never sent twice in a row")
    from agent.agent import GREETING_REPLY, PICK_ONE_REPLY, PICK_ONE_AGAIN, OFF_TOPIC_REPLY
    agent = HealthcareAgent(llm=NoLLM())
    replies = [agent.handle("Hi")["reply"] for _ in range(4)]
    check("'Hi' x4: greeting first, then short lines",
          replies[0] == GREETING_REPLY and replies[1] == PICK_ONE_REPLY, replies)
    check("  no two neighbouring replies are equal", all(a != b for a, b in zip(replies, replies[1:])), replies)
    r = agent.handle("Hi")
    check("  the short line keeps the bubbles", r["question"] and r["question"]["type"] == "menu")

    agent = HealthcareAgent(llm=fake_llm(*[category_plan("off_topic")] * 3))
    replies = [agent.handle(t)["reply"] for t in ["tell me a joke", "who won the game", "weather today?"]]
    check("off-topic x3: template, then short lines, never the same twice in a row",
          replies[0] == OFF_TOPIC_REPLY and replies[1] == PICK_ONE_REPLY
          and all(a != b for a, b in zip(replies, replies[1:])), replies)

    agent = HealthcareAgent(llm=fake_llm(AMBIGUOUS_PLAN))
    agent.handle("Can you show my father's medical history?")
    replies = [agent.handle("Hi")["reply"] for _ in range(3)]
    check("waiting for a name, 'Hi' x3: the 'waiting' message is not repeated",
          all(a != b for a, b in zip(replies, replies[1:])) and replies[1] == PICK_ONE_REPLY, replies)
    check("  the chat memory holds the same replies", agent.memory.as_text().count(WAITING_FOR_NAME) == 1)


def history_agent():
    """An agent that has just finished showing Ramesh Kumar's history (a completed task)."""
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Ramesh Kumar"), ("History", "get_patient_history", "")],
        name="Ramesh Kumar", condition=None)))
    agent.handle("Show the medical history of Ramesh Kumar")
    agent.handle("P001") if agent.pending["type"] == "choose_patient" else agent.handle("yes")
    return agent


def test_closing_and_goodbye():
    print("closing: 'Anything else?' / 'No, that's all', goodbye ends the chat")
    from agent.agent import (is_goodbye, FAREWELL_REPLY, ANYTHING_ELSE_REPLY, CLOSING_OPTIONS, NEW_CONVERSATION,
                             GREETING_REPLY)
    from memory.conversation_memory import save_patient_memory

    agent = history_agent()
    state = agent.pending_question()
    check("after a completed task: bubbles 'Anything else?' and 'No, that's all'",
          state is not None and state["type"] == "closing"
          and [b["label"] for b in state["buttons"]] == ["Anything else?", "No, that's all"], state)
    for text, expected in [("bye", True), ("Bye!", True), ("goodbye", True), ("thanks, that's all", True),
                           ("no more", True), ("No, that's all", True), ("nothing else", True),
                           ("thanks", False), ("no", False), ("yes", False), ("hello", False),
                           ("Show the medical history of Ramesh Kumar and then book a nephrologist, no more waiting", False)]:
        check(f"is_goodbye({text[:40]!r}) -> {expected}", is_goodbye(text) == expected)

    agent.llm = NoLLM()
    r = agent.handle("Anything else?")
    check("'Anything else?' -> asks what next, with the menu bubbles (no LLM)",
          r["reply"] == ANYTHING_ELSE_REPLY and r["question"]["type"] == "menu", r)

    save_patient_memory("P001", "Test note kept after the chat ends.", note_type="test")
    for text in ["No, that's all", "bye", "thanks, that's all", "no more"]:
        agent = history_agent()
        agent.llm = NoLLM()
        check("  (setup) chat has messages", len(agent.memory.messages) > 0 and agent.memory.get_context("patient_id") == "P001")
        r = agent.handle(text)
        check(f"{text!r} -> exactly 'Thank you! Visit again.' + 'Start a new conversation' bubble",
              r["reply"] == FAREWELL_REPLY == "Thank you! Visit again." and r["waiting_for"] is None
              and [b["label"] for b in r["question"]["buttons"]] == ["Start a new conversation"], r)
        check("  short-term memory is cleared (no messages, no patient)",
              agent.memory.messages == [] and agent.memory.get_context("patient_id") is None
              and agent.plan is None and agent.pending is None)
        check("  long-term patient memory is kept", any("Test note kept" in n["text"]
              for n in recall_patient_memory("P001", limit=10)["notes"]))
    r = agent.handle(NEW_CONVERSATION[1])
    check("'Start a new conversation' -> fresh greeting with the menu",
          r["reply"] == GREETING_REPLY and r["question"]["type"] == "menu"
          and len(agent.memory.messages) == 2, r)
    r = agent.handle("bye")
    r2 = agent.handle("bye")
    check("saying bye twice never repeats the message", r["reply"] == FAREWELL_REPLY and r2["reply"] != r["reply"], r2["reply"])

    # a goodbye while a booking question is open books nothing
    before = len(read_csv(config.APPOINTMENTS_CSV))
    agent = HealthcareAgent(llm=fake_llm(SCENARIO_PLAN, BOOKING_MESSAGE))
    agent.handle(SCENARIO)
    agent.handle("yes")
    log_start = len(read_logs())
    r = agent.handle("ok bye")
    check("bye during a booking question -> chat ends, nothing booked, the decline is logged",
          r["reply"] == FAREWELL_REPLY and len(read_csv(config.APPOINTMENTS_CSV)) == before
          and "decline_booking" in [row["tool_name"] for row in read_logs()[log_start:]], r["reply"])

    # a registration that was started from the menu bubble can be completed (no plan behind it)
    agent = HealthcareAgent(llm=NoLLM())
    agent.handle("Register a new patient")
    for answer in ["Lena Fischer", "29", "female", "self", "5550100555"]:
        r = agent.handle(answer)
    r = agent.handle("yes")
    check("menu registration completes: saved, then 'Anything else?' bubbles",
          "registered Lena Fischer" in r["reply"] and r["question"]["type"] == "closing", r["reply"])


def test_phone_privacy():
    print("phone numbers are not shown in the chat (or the log) unless typed by the user")
    path = config.PATIENTS_CSV
    original = path.read_text()
    patients = read_csv(path)
    patients.loc[patients["patient_id"] == "P001", "contact_phone"] = "+15550100777"   # a FAKE number
    write_csv(patients, path)

    log_start = len(read_logs())
    agent = HealthcareAgent(llm=fake_llm(SCENARIO_PLAN, BOOKING_MESSAGE))
    replies = [agent.handle(SCENARIO)["reply"], agent.handle("yes")["reply"]]
    check("identify + history + booking replies never contain the stored phone number",
          not any("5550100777" in reply for reply in replies), replies)
    shown = str(agent.trace) + agent.memory.as_text()
    check("  nor the steps / memory shown under 'Show details'", "5550100777" not in shown)
    check("  nor the tool log", "5550100777" not in config.TOOL_LOG_CSV.read_text())
    from agent.patient_identifier import identify_patient
    r = identify_patient("Ramesh Kumar", name="Ramesh Kumar")
    check("  nor identify_patient's question / message", "5550100777" not in r["question"] + r["message"])
    path.write_text(original)


def test_registration_with_widget_replies():
    print("registration answered with the widgets' replies")
    agent = HealthcareAgent(llm=fake_llm(plan_json(
        [("Identify", "identify_patient", "Omar Haddad"), ("History", "get_patient_history", "")],
        name="Omar Haddad", condition=None)))
    r = agent.handle("Show the medical history of Omar Haddad")
    r = agent.handle(r["question"]["buttons"][0]["reply"])              # "Register a new patient"
    check("after 'Register a new patient' the age number input is asked",
          r["question"]["number"] is not None, r["question"])
    r = agent.handle("52")                                               # number input sends "52"
    check("-> gender dropdown", r["question"]["select"]["label"] == "Gender", r["question"])
    r = agent.handle(r["question"]["select"]["options"][0]["reply"])     # Male
    check("-> contact person dropdown", r["question"]["select"]["label"] == "Contact person", r["question"])
    other = [o for o in r["question"]["select"]["options"] if o["label"] == "Other"][0]
    r = agent.handle(other["reply"])
    check("then the phone number is asked in the text box (no widget)",
          "phone number" in r["reply"] and r["question"] is None, r["reply"])
    r = agent.handle("5550100321")
    check("'Other' contact person is accepted -> preview with 'Contact person: other'",
          "Contact person: other" in r["reply"] and "Contact phone: 5550100321" in r["reply"] and "Age: 52" in r["reply"] and "Gender: Male" in r["reply"]
          and r["waiting_for"] == "register_patient", r["reply"])
    before = patient_count()
    r = agent.handle(r["question"]["buttons"][0]["reply"])               # Yes, save patient
    saved = read_csv(config.PATIENTS_CSV)
    saved = saved[saved["name"] == "Omar Haddad"].iloc[0]
    check("'Yes, save patient' -> saved through add_patient",
          patient_count() == before + 1 and (saved["age"], saved["gender"], saved["contact_person"])
          == ("52", "Male", "other"), saved.to_dict())


def test_summary_freshness_after_edit():
    print("an EDITED record makes the stored summary stale (not just a new record)")
    from agent.agent import retrieve_patient_history
    from memory.vector_store import get_patient_summary, records_fingerprint
    from tools.medical_records_tool import update_record, get_patient_history

    r = retrieve_patient_history("P002", llm=NoLLM())
    check("unchanged records -> the stored summary is used (no LLM call)",
          r["status"] == "success" and "stored summary" in r["message"], r["message"])
    before = get_patient_summary("P002")["summary"]
    check("the stored summary carries a content fingerprint of the records",
          before["records_hash"] == records_fingerprint(get_patient_history("P002")["records"]), before)

    update_record("R006", {"medications": "Sertraline 100mg"}, confirm=True)   # same record count
    fresh = "Anita Sharma (P002) has generalized anxiety disorder and now takes Sertraline 100mg."
    r = retrieve_patient_history("P002", llm=fake_llm(fresh))
    check("after editing a record -> a NEW summary is written (the old one was stale)",
          "new summary" in r["message"] and "Sertraline 100mg" in r["text"], r)
    after = get_patient_summary("P002")["summary"]
    check("  the new summary and fingerprint are saved",
          after["summary"] == fresh and after["records_hash"] != before["records_hash"])
    r = retrieve_patient_history("P002", llm=NoLLM())
    check("  asking again -> stored summary, no LLM call", "stored summary" in r["message"], r["message"])

    # Summaries saved before the fingerprint existed are still judged by record count only.
    import json as _json
    path = config.VECTOR_STORE_DIR / "patient_summaries.json"
    saved = _json.loads(path.read_text())
    del saved["P003"]["records_hash"]
    path.write_text(_json.dumps(saved))
    r = retrieve_patient_history("P003", llm=NoLLM())
    check("an older summary without a fingerprint -> still accepted by record count",
          "stored summary" in r["message"], r["message"])


def test_planner_failure():
    print("planner returns nonsense")
    r = make_plan("Book a doctor", llm=fake_llm("not json", "still not json"))
    check("invalid JSON twice -> status error, no crash", r["status"] == "error", r)
    agent = HealthcareAgent(llm=fake_llm("not json", "still not json"))
    r = agent.handle("Book a doctor")
    check("agent replies with the problem and waits for nothing",
          "valid plan" in r["reply"] and r["waiting_for"] is None, r["reply"])


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    real_get = requests.get
    build_patient_index(llm=fake_llm(*FAKE_SUMMARIES))   # FAISS store in test_data/

    test_extract_facts()
    test_validate_plan()
    test_reply_parsing()
    test_identify_patient()
    test_clean_search_topic()
    test_disease_rag()
    test_full_scenario()
    test_wrong_patient_then_name()
    test_name_age_mismatch_choice()
    test_ambiguous_asks_for_name()
    test_choose_other_slot()
    test_fully_booked_and_offline()
    test_reply_classification()
    test_not_a_name_while_waiting()
    test_register_new_patient()
    test_register_with_known_details_then_book()
    test_duplicate_registration_rejected()
    test_natural_replies()
    test_pending_question_mapping()
    test_pending_question_round_trips()
    test_specialty_question()
    test_booking_sentence_check()
    test_no_slots_and_waiting_list()
    test_other_doctor_offered_first()
    test_small_talk()
    test_scope_guard()
    test_closing_and_goodbye()
    test_phone_privacy()
    test_digit_patient_id()
    test_no_repeated_replies()
    test_registration_with_widget_replies()
    test_summary_freshness_after_edit()
    test_planner_failure()

    disease_search_tool.requests.get = real_get
    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed (fake LLM, saved MedlinePlus reply).")
