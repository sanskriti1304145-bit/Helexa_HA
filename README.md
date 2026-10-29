# Agentic Healthcare Assistant

Capstone project: a virtual medical assistant that books appointments, manages
patient records, summarizes medical history, and searches trusted sources for
disease information.

> **Demo only, not medical advice.** All patient data in `data/` is synthetic.

## Sample scenario

> "My 70-year-old father has chronic kidney disease. I want to book a
> nephrologist for him. Also, can you summarize latest treatment methods?"

The agent will:
1. Identify the patient
2. Retrieve his medical history
3. Check the doctor calendar and book an appointment
4. Search and summarize treatment options (RAG over MedlinePlus results)

## Tech stack

| Part | Choice |
|------|--------|
| LLM | Anthropic Claude (key loaded from `.env`) |
| Agent framework | LangChain |
| Vector store | FAISS + free local `sentence-transformers` embeddings |
| Disease search | MedlinePlus web service (no key needed) |
| Data | CSV files in `data/` |
| UI | Streamlit |

## Folder structure

```
config.py                  All settings and data file paths
data/                      Synthetic CSV data (the working copy, never changed by tests)
data_seed/                 The clean baseline of every CSV (master copy, new schema)
reset_data.py              Restores data/ from data_seed/ and prints a consistency report
tools/
  medical_records_tool.py  Find patients, read/add/update records   (Phase 2)
  appointment_tool.py      Doctors, slots, booking, cancelling      (Phase 3)
  logger.py                Logs every tool call to CSV              (Phase 4)
  disease_search_tool.py   MedlinePlus search                       (Phase 5)
  schedule_tool.py         Creates future slots from the template
memory/
  vector_store.py          Patient summaries in FAISS               (Phase 6)
  conversation_memory.py   Short-term and long-term memory          (Phase 6)
agent/
  prompts.py               One prompt per sub-task                  (Phase 7)
  planner.py               Breaks a request into sub-goals          (Phase 7)
  agent.py                 Runs the plan                            (Phase 7)
agent/
  patient_identifier.py    Find + check + confirm the patient       (Phase 7)
  disease_rag.py           MedlinePlus RAG with citations           (Phase 7)
evaluation/
  test_questions.json      16 test conversations + correct answers  (Phase 8)
  evaluate.py              Runs them, judge + booking check         (Phase 8)
  metrics.py               Metrics from the tool-call log           (Phase 8)
  reference/               Saved MedlinePlus pages (expected answers)
  results/                 Evaluation results
app/
  streamlit_app.py         Dashboard, one function per page         (Phase 9)
  dashboard_data.py        Read-only loaders for the dashboard      (Phase 9)
tests/                     One test script per tool
logs/                      Tool-call log CSV (created at runtime)
```

## Setup

```bash
source .venv/bin/activate          # activate the virtual environment
pip install -r requirements.txt    # install packages
cp .env.example .env               # then put your real key in .env
python -m tests.test_setup         # check that everything is ready
```

## How tests protect the data

Every test script first copies `data/` to `test_data/` and points the
`DATA_DIR` environment variable at the copy (see `tests/test_helpers.py`).
The original CSV files are never modified by tests.

## Resetting the data (data_seed/)

`data_seed/` holds the clean baseline: 12 patients (P001-P012, `contact_phone` blank), 23 records
(R001-R023), 10 doctors, 80 slots, appointments A001-A010, the weekly schedule template and an
empty `waitlist.csv`. It has no test rows. It is the **source** for:

- `python reset_data.py` - restores `data/` from `data_seed/` (asks for "yes"; `--yes` skips the
  question) and prints the row count of every file plus any consistency problems.
- the dashboard's **Reset demo data** button (restores `demo_data/`),
- the evaluation (every question starts from a fresh copy of `data_seed/`).

```
python reset_data.py              # restore data/ from data_seed/
python reset_data.py --check      # only report on data/ (changes nothing)
python reset_data.py --check-seed # only report on data_seed/
```

The consistency check reports: a booked slot without exactly one confirmed appointment, a
confirmed appointment whose slot is not booked, an appointment whose doctor differs from its
slot's doctor, unknown patient IDs (appointments, records, waiting list) and duplicate IDs.
`reset_data.py` does not touch `vector_store/`, `memory_store/` or `logs/`. Edit `data_seed/` only on
purpose (for example when the schema changes); keep a copy first. `data_before_reset/` (a copy of
`data/` taken before the baseline was created) is ignored by git. The tests still copy `data/`
(not `data_seed/`) into `test_data/`, so run `python reset_data.py` first if you edited `data/`.

## Running the tests

```bash
python -m tests.test_setup                  # Phase 1: packages, paths, API key
python -m tests.test_medical_records_tool   # Phase 2: medical records tool
python -m tests.test_appointment_tool       # Phase 3: appointment tool
python -m tests.test_logger                 # Phase 4: tool-call logger
python -m tests.test_disease_search_tool    # Phase 5: MedlinePlus search (offline, saved replies)

python -m tests.test_vector_store           # Phase 6: patient summaries + FAISS (fake LLM)
python -m tests.test_memory                 # Phase 6: short- and long-term memory
python -m tests.test_schedule_tool          # automatic future slots
python -m tests.test_agent                  # Phase 7: planner, identification, RAG, full scenario (fake LLM)
python -m tests.test_evaluation             # Phase 8: questions file, judge parsing, metrics (fake LLMs)
python -m tests.test_dashboard              # Phase 9: every dashboard page (Streamlit AppTest, fake LLM)

# Optional, needs internet: checks the real MedlinePlus service
python -m tests.live_check_medlineplus
python -m tests.live_check_medlineplus "asthma"
```

Disease information comes from the MedlinePlus web service
(<https://medlineplus.gov/about/developers/webservices/>), courtesy of the
U.S. National Library of Medicine. Limit: 85 requests per minute per IP.

## Design notes

- **Confirmation before writes.** `add_record`, `update_record` and `add_patient`
  write nothing unless called with `confirm=True`. Without it they return
  `status: "needs_confirmation"` and a `will_write` preview to show the user.
- **Finding a patient by name.** Exact full-name matches (ignoring case) are shown first;
  there can be several (two patients with the same name), each with age, gender and ID.
  **Similar names** come after them, labeled "Similar name", only if the surname is at least
  `SURNAME_SIMILARITY_MIN` (0.85) alike AND the first name at least `FIRST_NAME_SIMILARITY_MIN`
  (0.78) alike. The two parts are compared separately with `difflib.SequenceMatcher` on
  lower-case text (never the whole name together), so "Ankit Prasad" is never offered for
  "Sanskriti Prasad" while "Rakesh"/"Ramesh Kumar" and "Sangita"/"Sangeeta Devi" are. Both
  numbers are in `config.py` for tuning. With no exact match the similar names are offered as
  "Did you mean...?"; with none the agent says the patient was not found. Age or gender from the
  request only narrow the look-alikes (and add a note such as "does not match age 70"); they
  never add a patient. Lists always end with "None of these".
- **Two files, one change.** Booking and cancelling update both `slots.csv` and
  `appointment.csv`. All checks run before any write; if the second write
  fails, the first file is restored, so the two files never disagree.
- **Every tool call is logged.** The `@log_tool_call` decorator writes one row
  per call to `logs/tool_calls.csv` (tool name, status, success, time taken,
  error message) without changing what the tool returns. Only top-level calls
  are logged: a tool called from inside another tool writes no row, so one
  agent step = one row. Tests log to `test_data/logs/` instead.
- **Scope guard.** The assistant only handles patient records, appointments and disease
  information. The planner classifies every message as `in_scope`, `greeting`, `cancel`,
  `medical_advice`, `unclear` or `off_topic` (obvious greetings, cancel words and dosing /
  diagnosis questions are also caught by simple rules, with no LLM call). Everything except
  `in_scope` gets a **fixed reply template** (never LLM-written) plus clickable option
  bubbles: *Show a patient's history, Book an appointment, Register a new patient, Learn about
  a disease*. "cancel", "never mind", "too late" or a death mention ends the current flow
  (nothing is saved; a death mention adds "I'm sorry to hear that."). The same bot message is
  never sent twice in a row: a repeat becomes "Pick one to continue" with the bubbles. While
  waiting for a patient, typing only digits ("007" or "7") asks "Did you mean P007, <name>,
  age <age>?" and lists no other patient.
- **No open slots and the waiting list.** If a specialty has no open slot at all, the
  assistant says "<Doctor> (<Specialty>) has no openings in the next 2 weeks." (always true:
  it is only said when there is not one future open slot) and shows the bubbles *Join the
  waiting list, Choose a different specialty, Back to start* plus a dropdown **Remind me in 3
  days / 7 days / 14 days**. If only one of several doctors in a specialty is empty, that is
  mentioned and the other doctors' slots are offered first. Joining the list (or choosing a
  reminder; "Join" uses 7 days) shows a preview and needs a Yes before `data/waitlist.csv`
  (`waitlist_id, patient_id, specialty, doctor_id, reason, status, created_at, recheck_on`)
  is written; the same patient cannot wait twice for the same specialty. The tools are
  `add_to_waitlist` and `list_waitlist` in `tools/waitlist_tool.py` (logged like the others).
  **Reminders:** every page checks the list when the app runs; an entry that is still
  "waiting" and whose `recheck_on` is today or earlier gives a banner "<Doctor> may have
  openings now. Check availability?" (or "<Specialty> doctors ..." if any doctor will do).
  The button runs the normal slot search (`list_available_slots`, logged); "Dismiss" hides it
  for the session. The Appointments page shows the waiting list, with `recheck_on`, next to
  the appointment tracker. In the evaluation a fully booked doctor still counts as a correct
  refusal (Q06), and Q21 joins the waiting list.
- **Contact phone.** `patients.csv` has a `contact_phone` column (existing patients: blank).
  Registration asks for it after the contact person: digits only, 7 to 15 digits, optional
  `+` and country code (spaces, dashes and brackets are ignored); anything else gets a clear
  message and the question is asked again. `add_patient` validates it too and the preview
  shows it. **Privacy:** the chat never prints a stored phone number (only the number the
  user is typing in for a registration), phone numbers are not in the patient search index,
  the steps/memory under "Show details" or the tool log (the log has no inputs). The
  dashboard's Patients page shows it ("not on file" when blank). Older demo copies get the new
  columns added (blank) when the app starts. All numbers in tests and examples are fake
  (555-01xx). A backup of `data/` from before this schema change is in `data_backup_*/`
  (ignored by git).
- **Closing the chat.** After a finished task the assistant shows the bubbles *Anything else?*
  and *No, that's all*. "No, that's all", "bye", "thanks, that's all", "no more", "nothing
  else" (short messages only) end the chat with the fixed reply "Thank you! Visit again." and
  a *Start a new conversation* bubble. Ending clears short-term chat memory (messages and the
  current patient) and keeps long-term patient notes (`memory_store/`). An open booking
  question is declined (logged) and nothing is booked.
- **Records form and summaries.** The Records page (and the chat) change records only after
  a preview and a confirmation. Each stored patient summary keeps a fingerprint of the
  records it was written from, so after a record is added **or edited** the summary is
  rewritten the next time the history is requested (counting records alone would have
  missed an edit).
- **Future slots are created automatically.** `data/schedule_template.csv` holds each
  doctor's weekly pattern (`doctor_id, weekday, time`), derived from the times in
  `slots.csv`. Dr. Kavita Rao (psychiatry) has no entries, so she stays fully booked.
  `ensure_future_slots(days_ahead=14)` (`tools/schedule_tool.py`) adds any missing
  weekday slot from the template for the next 14 days, with the next free slot IDs
  (S081, S082, ...). It only appends: existing slots and appointments are never changed,
  and running it twice adds nothing. Weekends and times already past are skipped.
  It runs when the dashboard starts and after "Reset demo data", **not** inside the
  booking or search tools, and it does nothing while `APP_NOW` freezes the clock, so the
  tests and the Phase 8 evaluation stay deterministic.
- **"Now" is configurable.** `config.get_now()` returns the real time, or the
  value of the `APP_NOW` environment variable (`YYYY-MM-DD HH:MM`) in tests.

## Building the patient store (Phase 6)

```bash
python -m memory.vector_store    # Claude summarizes each patient, FAISS index saved in vector_store/
```

Uses about 11 LLM calls (patients without records get a fixed sentence, no LLM).
Re-run it, or call `refresh_patient_summary(patient_id)`, after records change.

Generated folders (all in `.gitignore`): `vector_store/` (summaries + FAISS),
`memory_store/` (long-term notes per patient), `cache/` (MedlinePlus answers, 24 h),
`logs/` (tool-call log).

## Running the agent (Phase 7)

```bash
python -m tests.live_scenario   # sample scenario, real Claude + MedlinePlus, on a COPY of the data
python -m agent.agent           # chat in the terminal (uses the REAL data/; writes only after "yes")
```

How the agent works:
1. **Plan** - the LLM breaks the request into ordered sub-goals, each with one tool
   (`identify_patient`, `get_patient_history`, `book_appointment`, `search_disease_info`).
   Code then checks the plan's order (identify first, history before booking).
2. **Identify** - age and gender are taken from the request by simple rules
   ("70-year-old father" -> 70, male). Candidates come from the FAISS search (or the
   name), are checked against `patients.csv` (see "Finding a patient by name"). Exactly one
   patient -> a **confirmation card** (name, age, gender, patient ID, contact person; never the
   phone number or anything medical) and "Is this the patient you're looking for?" with Yes / No
   buttons. Several (same name twice, or look-alikes) -> a dropdown list with "None of these".
   No candidate, or several from a description -> the agent asks only "What is the
   patient's full name or patient ID?" and, for privacy, shows no other patients.
   While it waits for a name, a reply that is clearly not a name (e.g. a new request)
   is handled as a new request; an unclear reply gets the options "type a patient name
   or ID, say 'new patient' to register one, or say 'cancel' to start over".
   If a typed name is not found, the agent offers to **register a new patient**: it asks
   for the missing details (name, age, gender, contact person; age/gender from the
   request are reused), shows a preview, saves with `add_patient` only after "yes"
   (duplicates are rejected), and then continues the original request.
   It never guesses.
3. **History** - the stored summary (memory), refreshed if the records changed.
4. **Book** - earliest free slot; the exact booking is shown and only saved after "yes".
5. **Disease info (RAG)** - MedlinePlus is searched with the disease name only (words like
   "latest treatments" are removed; they made MedlinePlus return unrelated topics),
   the text is split into passages, the most relevant
   are retrieved, the LLM summarizes only those with citations, and code adds the
   source links. Without a source, nothing is summarized.

Known limitations: MedlinePlus gives general patient information, not the newest
research. `langchain-community` (FAISS wrapper) shows a deprecation warning; it
still works.

## Running the dashboard (Phase 9)

```bash
streamlit run app/streamlit_app.py      # opens http://localhost:8501
python -m tests.test_dashboard          # offline test of every page (Streamlit AppTest, fake LLM)
```

The dashboard works on a **demo copy** of the data in `demo_data/` (created on first
start), so nothing you click changes `data/`. On start it creates the next 14 days of slots
(see "Future slots" above; the sidebar shows how many were added). The sidebar's
**Reset demo data** button (it asks "Yes, reset" / "Cancel" first) copies the clean baseline
(`data_seed/`) back, clears long-term memory and creates the future slots again. The disclaimer is shown on every page.

| Page | What it shows |
|------|---------------|
| Health Assistant Helexa | A normal chat (or pick a sample scenario). Replies contain only the answer or question; each has a collapsed "Show details" with the plan, the steps and the memory. Closed questions are answered by clicking: yes/no buttons (patient, booking, saving a new patient), a slot dropdown, a patient dropdown for similar names, a specialty dropdown. Registering a patient uses a number input for age and dropdowns for gender and contact person (self, son, daughter, wife, husband, other). Only the latest question has widgets; once answered, the widget is removed and the chat keeps just the choice as one line ("Slot: Wed 28 Oct, 13:30, Dr. Arjun Mehta", "Patient: Ramesh Kumar (P001)"); the full option list is only under "Show details". Typing still works. Greetings ("Hi") get a short reply listing what the assistant can do |
| Patients | Details, stored LLM summary, records, appointments, long-term notes |
| Records | Attendant form to add or update a medical record: preview first, saved only after "Confirm and save". Patient ID and record ID are shown but cannot be edited |
| Doctors | A doctor's calendar: available / booked / past slots; warns when fully booked |
| Appointments | Real-time tracking (refreshes every 5 s); cancel with preview + confirmation |
| Medical info | Latest MedlinePlus summary with its retrieved passages and links; recent searches |
| Evaluation | Answer precision, booking success rate, correct refusals, booking checks; per-question judge reasoning; time per step |
| Memory & logs | Try-a-scenario box, planning breakdown, short- and long-term memory, tool-call log |

How the clickable answers work: `HealthcareAgent.pending_question()` turns what the
agent is waiting for into structured options (buttons and/or a dropdown). Each option
carries the exact reply the agent already understands ("yes", "S023", "P001",
"Nephrology") plus a readable chat message ("Slot S023 (Mon 26 Oct, 09:30)"), so the
agent's logic is the same whether the user clicks or types. If the user didn't say
which kind of doctor (or the planner silently picked General Medicine), the agent
asks with a specialty dropdown instead of assuming.

Viewing pages reads the files directly (`app/dashboard_data.py`), so it adds no rows
to the tool log; only real agent steps, cancellations and the Records form are logged.

## Evaluation (Phase 8)

```bash
python -m tests.test_evaluation              # offline check of the evaluation code (fake LLMs)
python -m evaluation.evaluate                # real run: all 20 questions (~50-70 LLM calls; Q17-Q20 are out-of-scope cases)
python -m evaluation.evaluate --ids Q05,Q08  # only some questions
```

**The test set** is `evaluation/test_questions.json` (edit it freely). Each of the 16
questions is a short scripted conversation: the user's first message, their later
replies ("yes", a patient ID, a slot ID), the known correct answer, and the booking
that must exist afterwards (or none). It covers the edge cases: similar names,
name/age mismatch, a patient with no records, a fully booked psychiatrist, past-dated
slots, an unknown specialty, an ambiguous patient, and a disease MedlinePlus can't
find. Expected disease answers come from the MedlinePlus pages saved in
`evaluation/reference/`.

**How scoring works, in simple words**

1. *Fair, repeatable setup.* Before each question the original CSVs are copied
   into `eval_data/` and a brand-new agent is started, so questions can't affect
   each other and `data/` is never touched. The clock is frozen at
   2026-10-21 12:00, so the "earliest free slot" is always the same.
2. *The judge (LLM-as-judge).* A second LLM call, using LangChain's
   `CotQAEvalChain` (the step-by-step version of `QAEvalChain`), acts like a
   teacher marking a quiz. It is given the question, the known correct answer and
   the agent's whole conversation. It writes its reasoning step by step and ends
   with **CORRECT** or **INCORRECT**. Extra details are allowed; anything that
   contradicts the correct answer, or is missing from it, is marked wrong.
3. *The booking check (plain code, no LLM).* `appointment.csv` is compared before
   and after. The expected booking must exist and nothing else may be booked.
   This catches mistakes the judge might miss, because it reads the actual file.
4. *Metrics from the log.* Every agent step writes one log row (top-level calls
   only, so nothing is counted twice). The judge's grades are logged too.
   - **Answer precision** = answers graded CORRECT / answers graded.
   - **Booking success rate** = bookings saved / bookings the user said "yes" to.
     "No slots" or a declined booking never reaches that step, so it isn't counted
     as a failure: not booking was the right answer there.
   - **Correct refusals** (separate figure) = requests that rightly ended without a
     booking, counted by reason: doctor fully booked, specialty does not exist,
     requested slot refused (e.g. in the past), user declined.
   - Per step: number of calls, share without errors, and average time.

Results go to `evaluation/results/`: `results_<time>.csv` (one row per question
with the conversation, the judge's grade and reasoning, and the booking check),
`summary_<time>.json` (all metrics), a copy of that run's log, and
`latest_results.csv` / `latest_summary.json` for the dashboard.

**Results (2026-10-22, agent and judge: claude-sonnet-5-5)**

| Run | Answer precision (judge) | Booking success rate | Booking checks (code) |
|-----|--------------------------|----------------------|-----------------------|
| 1 (before fix) | 15/16 = 0.94 | 3/3 = 1.0 | 16/16 |
| 2 (after fix)  | 16/16 = 1.0  | 3/3 = 1.0 | 16/16 |
| 3 (privacy change, refusals added) | 16/16 = 1.0 | 3/3 = 1.0 | 16/16 |

Run 3 also reports 5 correct refusals: 1 doctor fully booked, 1 specialty does not
exist, 1 past slot refused, 2 user declined.

Run 1 found a real bug (Q15): asked for an oncologist, the planner silently dropped the
booking step, so the user was never told there is no oncologist. The plan prompt now
says to keep every requested step and let the booking tool report "not available".
Run 1 is kept in `evaluation/results/run1_before_fix/`.

**Limitations to mention:** the judge is an LLM too, so it can occasionally be
wrong, and here it is the same model family as the agent (it may be lenient with
its own style; `JUDGE_MODEL` in `config.py` can be set to a different model).
16 questions is a small sample, so one question changes the precision by about
6 percentage points. Disease questions call the live MedlinePlus service, so
its text can change over time.
