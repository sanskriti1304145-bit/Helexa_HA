"""
test_evaluation.py - OFFLINE tests for the evaluation code (Phase 8).

Run from the project root:
    python -m tests.test_evaluation

Checks the questions file, the grade parser, the metrics, and a small
evaluation run with FAKE LLMs (agent and judge), so it costs nothing.
"""

import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()   # BEFORE importing evaluate, so it uses test_data/ (setdefault)

import config  # noqa: E402
from evaluation import evaluate  # noqa: E402
from evaluation.metrics import compute_metrics, read_log  # noqa: E402
from memory.vector_store import build_patient_index  # noqa: E402
from tests.fake_data import FAKE_SUMMARIES  # noqa: E402
from tools.csv_helpers import read_csv  # noqa: E402

REQUIRED_EDGE_CASES = {"similar_names", "name_age_mismatch", "no_records", "fully_booked",
                       "past_slot", "disease_not_found", "waiting_list", "off_topic",
                       "personal_medical_advice", "vague_input", "cancel_midway"}


def fake(*responses):
    return FakeListChatModel(responses=list(responses))


def test_questions_file():
    print("questions file")
    data = json.loads(evaluate.QUESTIONS_FILE.read_text())
    qs = data["questions"]
    check("10 to 25 questions (the original 16, plus out-of-scope and waiting-list cases)", 10 <= len(qs) <= 25, len(qs))
    check("question IDs are unique", len({q["id"] for q in qs}) == len(qs))
    check("every question has turns, a category and an expected answer",
          all(q["turns"] and q["category"] and len(q["expected_answer"]) > 20 for q in qs))
    edge = {q["edge_case"] for q in qs if q["edge_case"]}
    check("all required edge cases are covered", REQUIRED_EDGE_CASES <= edge, REQUIRED_EDGE_CASES - edge)
    patients = set(read_csv(config.PATIENTS_CSV)["patient_id"])
    slots = set(read_csv(config.SLOTS_CSV)["slot_id"])
    bookings = [q["expected_booking"] for q in qs if q["expected_booking"]]
    check("expected bookings use real patient and slot IDs",
          all(b["patient_id"] in patients and b["slot_id"] in slots for b in bookings), bookings)
    check("reference_now is set", data["reference_now"] == "2026-10-21 12:00")


def test_parse_grade():
    print("parse_grade")
    g, r = evaluate.parse_grade("EXPLANATION: Step 1 matches.\nStep 2 fine.\n\nGRADE: CORRECT")
    check("CORRECT + reasoning", g == "CORRECT" and r == "Step 1 matches.\nStep 2 fine.", (g, r))
    g, _ = evaluate.parse_grade("The doctor differs.\nGRADE: INCORRECT")
    check("INCORRECT is not mistaken for CORRECT", g == "INCORRECT", g)
    g, _ = evaluate.parse_grade("**GRADE:** incorrect")
    check("bold / lower-case grade still read", g == "INCORRECT", g)
    g, _ = evaluate.parse_grade("I think it is fine.")
    check("no grade -> None", g is None)


def test_metrics():
    print("metrics (from a hand-made log)")
    def row(tool, status, success="True", ms="10"):
        return {"tool_name": tool, "status": status, "success": success, "duration_ms": ms}
    rows = [row("confirm_booking", "success"), row("confirm_booking", "success"),
            row("confirm_booking", "slot_unavailable"),
            row("judge_answer", "correct"), row("judge_answer", "correct"),
            row("judge_answer", "incorrect"), row("judge_answer", "error", "False"),
            row("prepare_booking", "no_slots"), row("prepare_booking", "invalid"),
            row("prepare_booking", "needs_confirmation"), row("decline_booking", "declined")]
    m = compute_metrics(rows)
    check("correct refusals counted separately, by reason",
          m["booking_refusals"] == {"total": 3, "by_reason": {
              "doctor fully booked": 1, "requested slot refused (past, taken or wrong specialty)": 1,
              "user declined the offer": 1}}, m["booking_refusals"])
    check("booking success rate = 2 saved / 3 confirmed = 0.667",
          m["booking_success_rate"] == {"rate": 0.667, "saved": 2, "confirmed": 3}, m["booking_success_rate"])
    check("answer precision = 2 correct / 3 graded (error left out)",
          m["answer_precision"] == {"precision": 0.667, "correct": 2, "graded": 3, "judge_errors": 1},
          m["answer_precision"])
    check("'no_slots' is not counted as a booking failure",
          m["booking_success_rate"]["confirmed"] == 3)
    check("per-tool summary counts calls and statuses",
          m["per_tool"]["judge_answer"]["calls"] == 4
          and m["per_tool"]["judge_answer"]["statuses"]["correct"] == 2, m["per_tool"])
    empty = compute_metrics([])
    check("empty log -> rates are None (no division by zero)",
          empty["booking_success_rate"]["rate"] is None and empty["answer_precision"]["precision"] is None)


def test_small_run():
    print("a small evaluation run (fake agent LLM + fake judge)")
    results_dir = TEST_DATA_DIR / "eval_results"
    # Q05: plan -> identify Daniel Brooks -> "yes" -> history (no LLM needed).
    plan = json.dumps({"patient_name": "Daniel Brooks", "condition": None, "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Daniel Brooks"},
        {"goal": "History", "tool": "get_patient_history", "input": ""}]})
    # Q12: plan -> Priya Patel -> "yes" -> booking message -> "no".
    plan12 = json.dumps({"patient_name": "Priya Patel", "condition": "skin", "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Priya Patel"},
        {"goal": "Book", "tool": "book_appointment", "input": "Dermatology"}]})
    agent_llm = fake(plan, plan12, "Shall I book Dr. Priya Nair on 2026-10-21 at 16:00?")
    judge_llm = fake("EXPLANATION: Says no records, invents nothing.\nGRADE: CORRECT",
                     "EXPLANATION: Offered S043, declined, nothing booked.\nGRADE: CORRECT")
    data_before = fingerprint(ORIGINAL_DATA_DIR)

    rows, summary = evaluate.run_evaluation(ids=["Q05", "Q12"], llm=agent_llm, judge_llm=judge_llm,
                                            results_dir=results_dir)
    check("2 questions ran", [r["id"] for r in rows] == ["Q05", "Q12"], [r["id"] for r in rows])
    check("Q05 conversation says Daniel has no records",
          "No medical records" in rows[0]["agent_conversation"], rows[0]["agent_conversation"])
    check("Q12 offered S043 and the booking check passed (nothing booked)",
          "S043" in rows[1]["agent_conversation"] and rows[1]["booking_check"] == "pass"
          and rows[1]["new_bookings"] == "none", rows[1])
    # A question where the expected booking IS made. The agent's last reply carries answer
    # bubbles ("Anything else?"), which must not disturb the booking check.
    book = {"id": "QT", "turns": ["Book Priya Patel", "yes", "yes"],
            "expected_booking": {"patient_id": "P006", "slot_id": "S043"}}
    plan_book = json.dumps({"patient_name": "Priya Patel", "condition": "skin", "steps": [
        {"goal": "Identify", "tool": "identify_patient", "input": "Priya Patel"},
        {"goal": "Book", "tool": "book_appointment", "input": "Dermatology"}]})
    outcome = evaluate.run_question(book, llm=fake(plan_book, "Shall I book Dr. Priya Nair on 2026-10-21 at 16:00?"))
    check("a made booking matches expected_booking (pass), also when the last reply has answer bubbles",
          outcome["booking_check"] == "pass" and outcome["new_bookings"] == ["P006/S043"], outcome)
    check("the transcript lists the choices the user would see",
          "[Choices shown to the user: Yes | No" in outcome["transcript"], outcome["transcript"])
    book["expected_booking"] = {"patient_id": "P006", "slot_id": "S999"}
    check("a booking that is not the expected one fails the check",
          evaluate.run_question(book, llm=fake(plan_book, "Shall I book?"))["booking_check"] == "fail")
    evaluate.reset_workspace()

    check("judge grade and reasoning saved",
          rows[0]["judge_grade"] == "CORRECT" and "invents nothing" in rows[0]["judge_reasoning"], rows[0])

    files = sorted(p.name for p in results_dir.iterdir())
    check("results CSV, summary JSON, log copy and 'latest' files written",
          any(f.startswith("results_") for f in files) and any(f.startswith("summary_") for f in files)
          and any(f.startswith("tool_calls_") for f in files)
          and "latest_results.csv" in files and "latest_summary.json" in files, files)
    saved = list(read_csv(results_dir / "latest_results.csv").to_dict(orient="records"))
    check("latest_results.csv has one row per question with all columns",
          len(saved) == 2 and list(saved[0].keys()) == evaluate.RESULT_COLUMNS)

    m = summary["metrics_from_log"]
    check("answer precision from the log = 2/2",
          m["answer_precision"]["correct"] == 2 and m["answer_precision"]["graded"] == 2, m["answer_precision"])
    check("no booking confirmed -> booking success rate None (0 confirmed)",
          m["booking_success_rate"]["confirmed"] == 0, m["booking_success_rate"])
    check("Q12's declined booking counted as a correct refusal",
          m["booking_refusals"]["by_reason"] == {"user declined the offer": 1}, m["booking_refusals"])
    tools = set(m["per_tool"])
    check("log has only top-level step rows (no inner tool rows)",
          tools <= {"make_plan", "identify_patient", "retrieve_patient_history", "prepare_booking",
                    "confirm_booking", "decline_booking", "summarize_disease_info", "judge_answer"}, tools)
    check("original data/ untouched by the run", fingerprint(ORIGINAL_DATA_DIR) == data_before)


def test_safety_guard():
    print("safety guard")
    source = Path(evaluate.__file__).read_text()
    check("evaluate.py refuses to run on the original data/ folder",
          "Refusing to run" in source and config.DATA_DIR != ORIGINAL_DATA_DIR)


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    build_patient_index(llm=fake(*FAKE_SUMMARIES))

    test_questions_file()
    test_parse_grade()
    test_metrics()
    test_small_run()
    test_safety_guard()

    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed (no LLM calls, no internet).")
