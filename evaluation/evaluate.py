"""
evaluate.py - run the test questions, grade the answers, compute metrics.

Run from the project root (needs the API key and internet; about 40-60 LLM calls):
    python -m evaluation.evaluate                 # all questions
    python -m evaluation.evaluate --ids Q05,Q08   # only some
    python -m evaluation.evaluate --judge-model claude-opus-5-5   # another judge

HOW IT WORKS (in simple words)
  1. The questions live in evaluation/test_questions.json. Each one is a short
     scripted conversation ("turns": the question, then replies like "yes")
     plus the known correct answer and the booking that should (or should not)
     exist afterwards.
  2. For EACH question we copy the clean baseline (data_seed/) into eval_data/data/
     and start a brand-new agent, so questions cannot affect each other and
     the original files are never touched. "Now" is frozen at the time in the
     file (reference_now), so the expected slots are always the same.
  3. We play the turns to the agent and record the whole conversation.
  4. Two independent checks:
       a. JUDGE (LLM-as-judge, LangChain's CotQAEvalChain): a second LLM
          reads the question, the known correct answer and the agent's
          conversation, writes its reasoning step by step, and ends with
          GRADE: CORRECT or INCORRECT. Extra details are fine; anything that
          contradicts the correct answer makes it INCORRECT.
       b. BOOKING CHECK (plain code, no LLM): compare appointment.csv before
          and after. The expected booking must exist, and nothing else may
          have been booked.
  5. Metrics are calculated from the tool-call log (evaluation/metrics.py).
  6. Results are saved to evaluation/results/: one CSV row per question
     (with the judge's reasoning), a JSON summary, and a copy of the log.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_data"

# --- SAFETY: point every data folder at eval_data/ BEFORE importing project code.
# (config.py reads these when it is first imported.) setdefault keeps any value
# a test has already set (tests use test_data/).
os.environ.setdefault("DATA_DIR", str(EVAL_DIR / "data"))
os.environ.setdefault("TOOL_LOG_CSV", str(EVAL_DIR / "logs" / "tool_calls.csv"))
os.environ.setdefault("VECTOR_STORE_DIR", str(EVAL_DIR / "vector_store"))
os.environ.setdefault("MEMORY_DIR", str(EVAL_DIR / "memory_store"))
os.environ.setdefault("CACHE_DIR", str(EVAL_DIR / "cache"))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from langchain_classic.evaluation.qa import CotQAEvalChain  # noqa: E402

import config  # noqa: E402
from evaluation.metrics import compute_metrics, read_log  # noqa: E402
from tools.csv_helpers import read_csv  # noqa: E402
from tools.logger import log_tool_call  # noqa: E402

ORIGINAL_DATA_DIR = PROJECT_ROOT / "data"          # only used for the safety guard below
SEED_DATA_DIR = PROJECT_ROOT / "data_seed"         # every question starts from this clean baseline
QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

if config.DATA_DIR.resolve() == ORIGINAL_DATA_DIR.resolve():
    raise SystemExit("Refusing to run: the evaluation must not use the original data/ folder.")


# ---------------------------------------------------------------------------
# 1. Questions
# ---------------------------------------------------------------------------

def load_questions(path=QUESTIONS_FILE, ids=None):
    """Read the questions file. ids: optional list like ["Q01", "Q05"]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return data["reference_now"], questions


# ---------------------------------------------------------------------------
# 2. A clean workspace for every question
# ---------------------------------------------------------------------------

def prepare_vector_store():
    """The agent needs the patient summaries. Copy the real store if it exists,
    otherwise build it (11 LLM calls). Summaries match data/ exactly."""
    if (config.VECTOR_STORE_DIR / "faiss_index" / "index.faiss").exists():
        return
    real_store = PROJECT_ROOT / "vector_store"
    if real_store.exists():
        shutil.copytree(real_store, config.VECTOR_STORE_DIR, dirs_exist_ok=True)
    else:
        from memory.vector_store import build_patient_index
        print("Building the patient store (11 LLM calls)...")
        print(build_patient_index()["message"])


def reset_workspace():
    """Fresh copy of the baseline CSVs (data_seed/) and empty long-term memory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for csv_file in SEED_DATA_DIR.glob("*.csv"):
        shutil.copy(csv_file, config.DATA_DIR / csv_file.name)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)


def confirmed_bookings():
    """The set of (patient_id, slot_id) pairs with a confirmed appointment."""
    a = read_csv(config.APPOINTMENTS_CSV)
    return {(r["patient_id"], r["slot_id"]) for _, r in a[a["status"] == "confirmed"].iterrows()}


# ---------------------------------------------------------------------------
# 3. Run one question
# ---------------------------------------------------------------------------

def run_question(question, llm=None):
    """Play one scripted conversation to a new agent. Returns what happened."""
    from agent.agent import HealthcareAgent   # imported here, after the paths are set

    reset_workspace()
    before = confirmed_bookings()
    agent = HealthcareAgent(llm=llm)
    transcript, start = [], time.perf_counter()
    for turn in question["turns"]:
        try:
            result = agent.handle(turn)
            reply = result["reply"]
            # In the web page the choices are buttons / a dropdown, not text. The judge
            # must see what the user would see, so they are added to the transcript.
            widgets = result["question"]
            if widgets:
                shown = [b["label"] for b in widgets["buttons"]]
                shown += [o["label"] for o in (widgets["select"] or {"options": []})["options"]]
                reply += f"\n[Choices shown to the user: {' | '.join(shown)}]"
        except Exception as error:   # a crash is recorded, not hidden
            reply = f"[AGENT CRASHED: {type(error).__name__}: {error}]"
        transcript.append(f"User: {turn}\nAssistant: {reply}")

    new_bookings = confirmed_bookings() - before
    expected = question.get("expected_booking")
    expected_set = {(expected["patient_id"], expected["slot_id"])} if expected else set()
    return {
        "transcript": "\n\n".join(transcript),
        "seconds": round(time.perf_counter() - start, 1),
        "new_bookings": sorted(f"{p}/{s}" for p, s in new_bookings),
        "booking_check": "pass" if new_bookings == expected_set else "fail",
    }


# ---------------------------------------------------------------------------
# 4. The judge (LLM-as-judge with LangChain's CotQAEvalChain)
# ---------------------------------------------------------------------------

def parse_grade(judge_text):
    """Split the judge's reply into (grade, reasoning).

    The judge ends with "GRADE: CORRECT" or "GRADE: INCORRECT". We take the
    LAST grade in the text; INCORRECT is checked first because the word
    "CORRECT" is inside it. Returns grade None if no grade is found.
    """
    matches = re.findall(r"GRADE\W*\s*(INCORRECT|CORRECT)", judge_text, flags=re.IGNORECASE)
    grade = matches[-1].upper() if matches else None
    reasoning = re.split(r"GRADE\W*\s*(?:INCORRECT|CORRECT)", judge_text, flags=re.IGNORECASE)[0]
    reasoning = re.sub(r"^\s*EXPLANATION:\s*", "", reasoning).strip()
    return grade, reasoning


@log_tool_call
def judge_answer(question, transcript, judge_llm=None):
    """Grade one conversation. Logged, so answer precision comes from the log.

    CotQAEvalChain's prompt calls the known correct answer the "context":
    the judge checks the agent's answer against it, reasoning step by step.

    Status values: "correct", "incorrect", "error".
    """
    if judge_llm is None:
        from agent.llm import get_llm
        judge_llm = get_llm(model=config.JUDGE_MODEL)
    chain = CotQAEvalChain.from_llm(llm=judge_llm)
    query = (f"{question['turns'][0]}\n(The user's later replies in this conversation were: "
             f"{', '.join(question['turns'][1:]) or 'none'}.) Judge the whole conversation.")
    try:
        output = chain.evaluate([{"query": query, "context": question["expected_answer"]}],
                                [{"result": transcript}])[0]
        text = output.get("text") or output.get("results") or ""
    except Exception as error:
        return {"status": "error", "message": f"Judge failed: {error}", "grade": None, "reasoning": ""}
    grade, reasoning = parse_grade(text)
    if grade is None:
        return {"status": "error", "message": "Could not find GRADE in the judge's reply.",
                "grade": None, "reasoning": text}
    return {"status": grade.lower(), "message": f"Judge says {grade}.", "grade": grade,
            "reasoning": reasoning}


# ---------------------------------------------------------------------------
# 5. Run everything and save the results
# ---------------------------------------------------------------------------

RESULT_COLUMNS = ["id", "category", "edge_case", "question", "user_replies", "expected_answer",
                  "agent_conversation", "judge_grade", "judge_reasoning", "expected_booking",
                  "new_bookings", "booking_check", "seconds"]


def run_evaluation(ids=None, llm=None, judge_llm=None, results_dir=RESULTS_DIR,
                   questions_file=QUESTIONS_FILE):
    """Run the questions, save results + summary, return (rows, summary)."""
    reference_now, questions = load_questions(questions_file, ids)
    os.environ["APP_NOW"] = reference_now          # freeze "now" for predictable slots
    if config.TOOL_LOG_CSV.exists():
        config.TOOL_LOG_CSV.unlink()               # this run's log starts empty
    prepare_vector_store()

    rows = []
    for number, q in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {q['id']} ({q['edge_case'] or q['category']}) ...", flush=True)
        outcome = run_question(q, llm=llm)
        verdict = judge_answer(q, outcome["transcript"], judge_llm=judge_llm)
        expected = q.get("expected_booking")
        rows.append({
            "id": q["id"], "category": q["category"], "edge_case": q["edge_case"] or "",
            "question": q["turns"][0], "user_replies": " | ".join(q["turns"][1:]),
            "expected_answer": q["expected_answer"],
            "agent_conversation": outcome["transcript"],
            "judge_grade": verdict["grade"] or "ERROR",
            "judge_reasoning": verdict["reasoning"] or verdict["message"],
            "expected_booking": f"{expected['patient_id']}/{expected['slot_id']}" if expected else "none",
            "new_bookings": ", ".join(outcome["new_bookings"]) or "none",
            "booking_check": outcome["booking_check"],
            "seconds": outcome["seconds"],
        })
        print(f"      judge: {rows[-1]['judge_grade']:<9} booking check: {outcome['booking_check']}")

    # --- metrics from the log (top-level calls only) ---
    metrics = compute_metrics(read_log(config.TOOL_LOG_CSV))
    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reference_now": reference_now,
        "agent_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "questions": len(rows),
        "judge_correct": sum(r["judge_grade"] == "CORRECT" for r in rows),
        "booking_checks_passed": sum(r["booking_check"] == "pass" for r in rows),
        "metrics_from_log": metrics,
    }

    # --- save ---
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_dir / f"results_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.TOOL_LOG_CSV.exists():
        shutil.copy(config.TOOL_LOG_CSV, results_dir / f"tool_calls_{stamp}.csv")
    # "latest" copies, so the dashboard always knows which files to show
    shutil.copy(results_dir / f"results_{stamp}.csv", results_dir / "latest_results.csv")
    shutil.copy(results_dir / f"summary_{stamp}.json", results_dir / "latest_summary.json")
    summary["files"] = [f"results_{stamp}.csv", f"summary_{stamp}.json", f"tool_calls_{stamp}.csv"]
    return rows, summary


def print_report(rows, summary):
    m = summary["metrics_from_log"]
    print("\n" + "=" * 70)
    print(f"{'ID':<5}{'edge case / category':<24}{'judge':<11}{'booking':<9}seconds")
    for r in rows:
        print(f"{r['id']:<5}{(r['edge_case'] or r['category']):<24}{r['judge_grade']:<11}"
              f"{r['booking_check']:<9}{r['seconds']}")
    bsr, ap = m["booking_success_rate"], m["answer_precision"]
    print("=" * 70)
    print(f"Answer precision (judge, from log):   {ap['correct']}/{ap['graded']} = {ap['precision']}"
          + (f"   ({ap['judge_errors']} unreadable)" if ap["judge_errors"] else ""))
    print(f"Booking success rate (from log):      {bsr['saved']}/{bsr['confirmed']} = {bsr['rate']}")
    refusals = m["booking_refusals"]
    print(f"Correct refusals (from log):          {refusals['total']}   {refusals['by_reason']}")
    print(f"Booking checks passed (code):         {summary['booking_checks_passed']}/{summary['questions']}")
    print("Per step (top-level calls):")
    for tool, s in m["per_tool"].items():
        print(f"  {tool:<26} calls {s['calls']:>3}   no-error {s['no_error_rate']}   "
              f"avg {s['avg_ms']:>8} ms   {s['statuses']}")
    print(f"Saved: {', '.join(summary['files'])} in evaluation/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the healthcare agent.")
    parser.add_argument("--ids", help="comma-separated question IDs, e.g. Q01,Q05")
    parser.add_argument("--judge-model", help=f"model for the judge (default: {config.JUDGE_MODEL})")
    args = parser.parse_args()
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model
    chosen = [i.strip().upper() for i in args.ids.split(",")] if args.ids else None
    rows, summary = run_evaluation(ids=chosen)
    print_report(rows, summary)
