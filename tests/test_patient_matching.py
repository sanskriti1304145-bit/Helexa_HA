"""
test_patient_matching.py - tests for finding the right patient by name

Run from the project root:
    python -m tests.test_patient_matching

Runs on a COPY of the data (test_data/). Some FAKE patients are added to that
copy only (duplicate names, look-alike names); the real data/ is never touched.

Rules being tested (see agent/patient_identifier.py):
  - exact full-name matches (ignoring case) first; there can be several
  - "Similar name" only if surname >= 0.85 AND first name >= 0.78 (config.py),
    compared separately
  - no exact match -> similar names as "Did you mean...?"; none -> not found
  - age/gender never add a patient
  - always "None of these" in a list; one match -> a confirmation card
"""

import json
import os

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR,
)

use_fresh_test_data()
os.environ["APP_NOW"] = "2026-10-21 12:00"

import config  # noqa: E402
from agent.agent import HealthcareAgent  # noqa: E402
from agent.patient_identifier import identify_patient  # noqa: E402
from tools.csv_helpers import read_csv, write_csv  # noqa: E402
from tools.medical_records_tool import find_patient, name_similarity  # noqa: E402

# FAKE patients, added to the test copy only.
FAKE = [
    ("P013", "Sanskriti Prasad", "31", "Female", "self"),
    ("P014", "Sanskriti Prasad", "52", "Female", "daughter"),   # same name, other person
    ("P015", "Ankit Prasad", "28", "Male", "self"),              # same surname, different first name
    ("P016", "Sangeeta Devi", "44", "Female", "self"),
    ("P017", "Ramya Kumar", "33", "Female", "self"),             # surname Kumar, first name too different
]
patients = read_csv(config.PATIENTS_CSV)
for pid, name, age, gender, contact in FAKE:
    patients.loc[len(patients)] = {"patient_id": pid, "name": name, "age": age, "gender": gender,
                                   "contact_person": contact, "contact_phone": "+15550100999",
                                   "created_date": "2026-01-01"}
write_csv(patients, config.PATIENTS_CSV)


def ids(result):
    return [p["patient_id"] for p in result.get("candidates", [])]


def labels(result):
    return [p.get("match_label", "") for p in result["candidates"]]


def test_similarity_rule():
    print("name_similarity: first name and surname compared separately")
    check("thresholds live in config.py", config.SURNAME_SIMILARITY_MIN == 0.85
          and config.FIRST_NAME_SIMILARITY_MIN == 0.78)
    for a, b, expected in [("Rakesh Kumar", "Ramesh Kumar", True), ("Ramesh Kumaar", "Ramesh Kumar", True),
                           ("Sangita Devi", "Sangeeta Devi", True),
                           ("Sanskriti Prasad", "Ankit Prasad", False),      # first name too different
                           ("Ramesh Kumar", "Ramya Kumar", False),            # first name below 0.78
                           ("Ramesh Kumer", "Ramesh Kumar", False),           # surname 0.80 < 0.85
                           ("Ramesh Kumar", "Ramesh Kumar", True), ("RAMESH  kumar", "Ramesh Kumar", True)]:
        check(f"{a!r} vs {b!r} -> {expected}", name_similarity(a, b)[0] == expected, name_similarity(a, b))

    saved = config.FIRST_NAME_SIMILARITY_MIN
    config.FIRST_NAME_SIMILARITY_MIN = 0.5
    check("lowering the first-name minimum in config makes 'Ramya Kumar' similar to 'Ramesh Kumar'",
          name_similarity("Ramesh Kumar", "Ramya Kumar")[0])
    config.FIRST_NAME_SIMILARITY_MIN = saved


def test_exact_duplicates():
    print("two patients with the exact same name")
    r = identify_patient("x", name="Sanskriti Prasad", facts={})
    check("both exact matches are shown (and ONLY them): P013, P014", r["status"] == "needs_choice"
          and ids(r) == ["P013", "P014"], r["message"])
    check("  'Ankit Prasad' (P015) is NOT shown", "P015" not in ids(r))
    check("  neither is labeled 'Similar name'", labels(r) == ["", ""], labels(r))
    r2 = identify_patient("x", name="sanskriti PRASAD", facts={})
    check("  case is ignored", ids(r2) == ["P013", "P014"], ids(r2))

    agent = HealthcareAgent(llm=FakeListChatModel(responses=[json.dumps({"steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Sanskriti Prasad"}]})]))
    reply = agent.handle("Who is Sanskriti Prasad?")
    options = [o["label"] for o in reply["question"]["select"]["options"]]
    check("dropdown: each with age, gender and ID, then 'None of these'",
          options == ["Sanskriti Prasad, age 31, Female (P013)", "Sanskriti Prasad, age 52, Female (P014)",
                      "None of these"], options)
    check("  the reply text does not list them again", "P013" not in reply["reply"]
          and "age 52" not in reply["reply"], reply["reply"])
    reply = agent.handle("P014")
    check("choosing P014 -> the card/plan continues with that patient", reply["patient_id"] == "P014"
          or "Patient ID: P014" in reply["reply"], reply["reply"])


def test_similar_names():
    print("similar names")
    r = identify_patient("x", name="Rakesh Kumar", facts={})
    check("'Rakesh Kumar' -> Rakesh (exact) then Ramesh (Similar name); not Ramya",
          r["status"] == "needs_choice" and ids(r) == ["P011", "P001"] and labels(r) == ["", "Similar name"], r["message"])

    r = identify_patient("x", name="Ramesh Kumaar", facts={})
    check("'Ramesh Kumaar' (no exact) -> 'Did you mean...?' with Ramesh Kumar first",
          r["status"] == "needs_choice" and ids(r)[0] == "P001" and "P017" not in ids(r)
          and set(labels(r)) == {"Similar name"} and "Did you mean" in r["question"], (ids(r), r["question"]))

    r = identify_patient("x", name="Sangita Devi", facts={})
    check("'Sangita Devi' -> Sangeeta Devi is suggested (one match -> card, 'Did you mean')",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P016"
          and "Did you mean" in r["question"] and "Patient ID: P016" in r["question"], r["question"])

    r = identify_patient("x", name="Ramesh Kumar", facts={})
    check("first name below the threshold ('Ramya Kumar') is NOT shown for 'Ramesh Kumar'",
          "P017" not in ids(r) and ids(r) == ["P001", "P011"], ids(r))
    r = identify_patient("x", name="Ramesh Kumer", facts={})
    check("surname below the threshold ('Kumer', 0.80) -> nobody is shown", r["status"] == "needs_name", r["message"])

    f = find_patient("Sanskriti Prasad")
    check("find_patient also keeps 'Ankit Prasad' out", "P015" not in [m["patient_id"] for m in f["matches"]])


def test_none_single_and_mismatch():
    print("not found, single match, age/gender mismatch")
    r = identify_patient("x", name="Zelda Fitzgerald", facts={})
    check("no match -> says the patient was not found, asks the fixed question, lists nobody",
          r["status"] == "needs_name" and "could not find a patient named 'Zelda Fitzgerald'" in r["question"]
          and r["question"].endswith("What is the patient's full name or patient ID?")
          and "candidates" not in r, r["question"])
    r = identify_patient("x", name="Unknown Person", facts={"age": 70, "gender": "male"})
    check("a fitting age/gender NEVER adds a patient (nobody is similar to 'Unknown Person')",
          r["status"] == "needs_name" and "patient" not in r, r["message"])

    r = identify_patient("x", name="Anita Sharma", facts={})
    check("single match -> confirmation card + the question",
          r["status"] == "needs_confirmation" and r["question"].endswith("Is this the patient you're looking for?")
          and all(x in r["question"] for x in ("Name: Anita Sharma", "Age: 34", "Gender: Female",
                                               "Patient ID: P002", "Contact person: self")), r["question"])
    check("  no phone number and no medical information before the user confirms",
          "5550100" not in r["question"] and "phone" not in r["question"].lower()
          and "anxiety" not in r["question"].lower() and "Sertraline" not in r["question"], r["question"])
    check("  the patient dict is only returned for the agent to use after 'yes'", r["patient"]["patient_id"] == "P002")

    r = identify_patient("x", name="Anita Sharma", facts={"age": 70})
    check("mismatch, one exact match -> still the card, with a note that it does not match age 70",
          r["status"] == "needs_confirmation" and "does not match age 70" in r["question"], r["question"])
    r = identify_patient("x", name="Rakesh Kumar", facts={"age": 70, "gender": "male"})
    check("mismatch with a look-alike that fits -> both shown (exact first), exact one has the note",
          ids(r) == ["P011", "P001"] and r["candidates"][0]["note"] and not r["candidates"][1]["note"], ids(r))
    r = identify_patient("x", name="Rakesh Kumar", facts={"age": 48})
    check("exact name and the age fits -> just one card (look-alikes dropped)",
          r["status"] == "needs_confirmation" and r["patient"]["patient_id"] == "P011", r["message"])


def test_agent_flow():
    print("the agent: list, 'None of these', card, Yes/No")
    plan = json.dumps({"steps": [{"goal": "Identify", "tool": "identify_patient", "input": "Rakesh Kumar"}]})
    agent = HealthcareAgent(llm=FakeListChatModel(responses=[plan]))
    r = agent.handle("Who is Rakesh Kumar?")
    options = [o["label"] for o in r["question"]["select"]["options"]]
    check("list: exact first, 'Similar name' label, 'None of these' last",
          options == ["Rakesh Kumar, age 48, Male (P011)", "Ramesh Kumar, age 70, Male (P001) - Similar name",
                      "None of these"], options)
    r = agent.handle("neither")
    check("'None of these' -> asks for the name", r["waiting_for"] == "patient_name", r["reply"])

    plan = json.dumps({"steps": [{"goal": "Identify", "tool": "identify_patient", "input": "Anita Sharma"}]})
    agent = HealthcareAgent(llm=FakeListChatModel(responses=[plan]))
    r = agent.handle("Who is Anita Sharma?")
    check("single match -> card with Yes / No buttons",
          r["waiting_for"] == "confirm_patient" and [b["label"] for b in r["question"]["buttons"]] == ["Yes", "No"]
          and "Patient ID: P002" in r["reply"], r["reply"])
    check("  nothing is remembered about the patient until Yes", agent.memory.get_context("patient_id") is None)
    r = agent.handle("yes")
    check("  Yes -> the patient is used", r["patient_id"] == "P002")

    agent = HealthcareAgent(llm=FakeListChatModel(responses=[plan.replace("Anita Sharma", "Zelda Fitzgerald")]))
    r = agent.handle("Who is Zelda Fitzgerald?")
    check("not found -> offers to register, shows no other patient",
          r["waiting_for"] == "patient_name" and "not find" in r["reply"] and "P0" not in r["reply"], r["reply"])


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    test_similarity_rule()
    test_exact_duplicates()
    test_similar_names()
    test_none_single_and_mismatch()
    test_agent_flow()
    print("safety")
    check("original data/ files are unchanged (the fake patients exist only in test_data/)",
          fingerprint(ORIGINAL_DATA_DIR) == originals_before
          and "Sanskriti" not in (ORIGINAL_DATA_DIR / "patients.csv").read_text())
    print(f"\nAll {passed_count['passed']} checks passed (fake LLM, test copy of the data).")
