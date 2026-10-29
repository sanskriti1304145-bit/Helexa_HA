"""
streamlit_app.py - the dashboard for the Agentic Healthcare Assistant.

Run from the project root:
    streamlit run app/streamlit_app.py

Pages (chosen in the sidebar):
    Assistant      a normal chat; each reply has a collapsed "Show details"
                   with the plan, the steps and the memory
    Patients       one patient's details, summary, records, appointments, notes
    Records        attendant form: add or update a medical record (preview, then confirm)
    Doctors        one doctor's calendar: available / booked / past slots
    Appointments   real-time appointment tracking (refreshes every 5 s) + cancel
    Medical info   the latest MedlinePlus summary with sources, recent searches
    Evaluation     judge results, booking metrics, time per step
    Memory & logs  short- and long-term memory, plan, trace, tool log, try a scenario

SAFETY: the dashboard works on a DEMO COPY of the data (demo_data/), never on
data/ itself. "Reset demo data" in the sidebar copies the originals back.
"""

import os
import sys
from pathlib import Path

# --- Point the project at the demo copy BEFORE importing project code. ---
# (config.py reads these when first imported; setdefault respects any value
# already set, e.g. by the tests.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo_data"
os.environ.setdefault("DATA_DIR", str(DEMO_DIR / "data"))
os.environ.setdefault("TOOL_LOG_CSV", str(DEMO_DIR / "logs" / "tool_calls.csv"))
os.environ.setdefault("VECTOR_STORE_DIR", str(DEMO_DIR / "vector_store"))
os.environ.setdefault("MEMORY_DIR", str(DEMO_DIR / "memory_store"))
sys.path.insert(0, str(PROJECT_ROOT))   # so "import config" works under `streamlit run`

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from app import dashboard_data as data  # noqa: E402

BAR_COLOR = "#2a78d6"   # single-series charts: one hue, no legend needed
PAGES = ["Health Assistant Helexa", "Patients", "Records", "Doctors", "Appointments", "Medical info",
         "Evaluation", "Memory & logs"]

from tools.appointment_tool import list_available_slots  # noqa: E402
from tools.medical_records_tool import add_record, update_record  # noqa: E402
from tools.schedule_tool import ensure_future_slots  # noqa: E402

st.set_page_config(page_title="Health Assistant Helexa", page_icon="🩺", layout="wide")
data.setup_demo_workspace()


@st.cache_resource
def add_future_slots_on_start():
    """Create missing future slots ONCE when the app starts (cache_resource
    runs this a single time per server, not on every page refresh).
    It does nothing when the clock is frozen with APP_NOW (tests)."""
    return ensure_future_slots()


startup_slots = add_future_slots_on_start()


# ---------------------------------------------------------------------------
# Shared state: one agent per browser session
# ---------------------------------------------------------------------------

def get_agent():
    """The agent lives in st.session_state so it survives page reruns.
    (Tests put a fake LLM in st.session_state["llm"].)"""
    if "agent" not in st.session_state:
        from agent.agent import HealthcareAgent
        st.session_state.agent = HealthcareAgent(llm=st.session_state.get("llm"))
        st.session_state.chat = []      # list of (role, text) for display
        st.session_state.last = None    # last result from agent.handle()
    return st.session_state.agent


def send_to_agent(text, display=None):
    """Send one message to the agent and remember the conversation.

    text:    what the agent receives (e.g. "S023", "P001", "yes")
    display: what the chat shows as the user's message, if different
             (e.g. "Slot S023 (Mon 26 Oct, 09:30)" after picking from a dropdown)

    Each chat entry is a dict: {"role", "text", "details", "question"}.
    "details" keeps a snapshot of the plan, steps and memory at the time of
    that reply (collapsed "Show details"). "question" is the agent's
    structured pending question, drawn as buttons / a dropdown.
    """
    agent = get_agent()
    st.session_state.chat.append({"role": "user", "text": display or text, "details": None, "question": None})
    details, question = None, None
    with st.spinner("Working on it..."):
        try:
            result = agent.handle(text)
            reply = result["reply"]
            question = result["question"]
            st.session_state.last = result
            patient_id = result["patient_id"]
            details = {
                "plan": result["plan"],
                "trace": result["trace"],
                "waiting_for": result["waiting_for"],
                "context": {k: v for k, v in agent.memory.context.items() if k != "patient"},
                "messages_in_memory": len(agent.memory.get_messages()),
                "notes": data.patient_notes(patient_id)[-5:] if patient_id else [],
                "options": question_options(question),
            }
        except Exception as error:   # show problems instead of crashing the page
            reply = f"Sorry, something went wrong: {error}"
    st.session_state.chat.append({"role": "assistant", "text": reply, "details": details,
                                  "question": question})


def question_options(question):
    """The labels of every widget option of a question (kept only in 'Show details')."""
    if not question:
        return []
    labels = [b["label"] for b in question["buttons"]]
    if question["select"]:
        labels += [f"{question['select']['label']}: {o['label']}" for o in question["select"]["options"]]
    if question.get("number"):
        labels.append(question["number"]["label"])
    return labels


def show_question(question, key):
    """Clickable answers for a closed-choice question: buttons and/or a
    dropdown + Confirm. Each sends the option's "reply" to the agent and
    shows its "display" text as the user's chat message."""
    if question["buttons"]:
        with st.container(horizontal=True):
            for i, choice in enumerate(question["buttons"]):
                # Every bubble has the same style: nothing is highlighted until it is clicked.
                if st.button(choice["label"], key=f"{key}_button_{i}"):
                    send_to_agent(choice["reply"], display=choice["display"])
                    st.rerun()
    select = question["select"]
    if select:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            labels = [o["label"] for o in select["options"]]
            picked = st.selectbox(select["label"], labels, key=f"{key}_select")
            if st.button(select["submit"], key=f"{key}_submit"):
                choice = select["options"][labels.index(picked)]
                send_to_agent(choice["reply"], display=choice["display"])
                st.rerun()
    number = question.get("number")
    if number:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            # value=None: the box starts empty, so nobody confirms a made-up default age.
            value = st.number_input(number["label"], min_value=number["min"], max_value=number["max"],
                                    value=None, step=1, placeholder="Type a number", key=f"{key}_number")
            if st.button(number["submit"], key=f"{key}_number_submit", disabled=value is None):
                send_to_agent(str(int(value)), display=number["display"].format(value=int(value)))
                st.rerun()


def as_chat_text(text):
    """Make a reply display like normal chat text in markdown: keep line
    breaks, and escape '$' (Streamlit would treat it as a maths formula)."""
    return text.replace("$", r"\$").replace("\n", "  \n")


def show_details(details):
    """The agent's internals for one reply, collapsed by default."""
    with st.expander("Show details", expanded=False):
        if details["plan"]:
            st.markdown("**Plan (sub-goals in order)**")
            st.dataframe(pd.DataFrame(details["plan"]).rename(columns={"input": "tool input"}),
                         hide_index=True)
        if details["trace"]:
            st.markdown("**Steps run so far**")
            st.dataframe(pd.DataFrame(details["trace"]), hide_index=True)
        if details.get("options"):
            st.markdown("**Options offered**")     # the chat itself shows only what the user picked
            st.dataframe(pd.DataFrame({"option": details["options"]}), hide_index=True)
        st.markdown("**Memory**")
        waiting = details["waiting_for"]
        st.caption(f"Short-term memory: {details['messages_in_memory']} message(s) · context: "
                   f"{details['context'] or 'none'} · waiting for: "
                   f"{waiting.replace('_', ' ') if waiting else 'nothing'}")
        if details["notes"]:
            st.caption("Long-term notes for this patient (latest first):")
            st.dataframe(pd.DataFrame(details["notes"][::-1]), hide_index=True)


def new_conversation():
    for key in ("agent", "chat", "last"):
        st.session_state.pop(key, None)


def show_plan_and_trace():
    last = st.session_state.get("last")
    if not last:
        st.caption("No request yet. The plan and steps appear here after you send a message.")
        return
    if last["plan"]:
        st.markdown("**Plan (sub-goals in order)**")
        st.dataframe(pd.DataFrame(last["plan"]).rename(columns={"input": "tool input"}),
                     hide_index=True)
    if last["trace"]:
        st.markdown("**Steps run so far**")
        st.dataframe(pd.DataFrame(last["trace"]), hide_index=True,
                     column_config={"message": st.column_config.TextColumn(width="large")})
    waiting = last["waiting_for"]
    st.caption(f"Waiting for your answer: **{waiting.replace('_', ' ')}**" if waiting
               else "Not waiting for an answer.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_assistant():
    st.header("Health Assistant Helexa")
    get_agent()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        choice = st.selectbox("Try a sample scenario", ["(type your own below)"] + data.sample_scenarios())
        if st.button("Send scenario", disabled=choice.startswith("(")):
            send_to_agent(choice)
        if st.button("New conversation"):
            new_conversation()
            st.rerun()
    chat = st.session_state.chat
    if not chat:     # empty welcome screen: offer the four bubbles before the user types anything
        from agent.agent import menu_question
        st.markdown("**Hello! I'm Helexa. What would you like to do?**")
        show_question(menu_question(), key="welcome")
    for number, entry in enumerate(chat):
        with st.chat_message(entry["role"]):
            st.markdown(as_chat_text(entry["text"]))
            # Widgets only under the LATEST reply: once answered, older
            # questions show no buttons, so they can't be clicked by mistake.
            if entry.get("question") and number == len(chat) - 1:
                show_question(entry["question"], key=f"q{number}")
            if entry["details"]:
                show_details(entry["details"])
    message = st.chat_input("Type a request or an answer (you can also type yes / no / an ID)")
    if message:
        send_to_agent(message)
        st.rerun()


def page_patients():
    st.header("Patients")
    people = data.patients()
    labels = {f"{r['name']} ({r['patient_id']})": r["patient_id"] for _, r in people.iterrows()}
    patient_id = labels[st.selectbox("Patient", list(labels))]
    p = people.set_index("patient_id").loc[patient_id]

    with st.container(horizontal=True):
        st.metric("Age", p["age"], border=True)
        st.metric("Gender", p["gender"], border=True)
        st.metric("Contact person", p["contact_person"], border=True)
        st.metric("Contact phone", p.get("contact_phone") or "not on file", border=True)
        st.metric("Registered", p["created_date"], border=True)

    summary = data.patient_summary(patient_id)
    st.subheader("Summary")
    if summary:
        st.write(summary["summary"])
        st.caption(f"Written by {'the LLM' if summary['generated_by'] == 'llm' else 'a fixed template'} "
                   f"on {summary['generated_at']} from {summary['record_count']} record(s).")
    else:
        st.info("No stored summary yet. Build the patient store with `python -m memory.vector_store`.")

    st.subheader("Medical records (oldest first)")
    records = data.patient_records(patient_id)
    if records.empty:
        st.info(f"{p['name']} has no medical records on file.")
    else:
        st.dataframe(records.drop(columns=["patient_id"]), hide_index=True)

    st.subheader("Appointments")
    appts = data.appointments_table()
    mine = appts[appts["patient_id"] == patient_id] if not appts.empty else appts
    if mine.empty:
        st.caption("No appointments.")
    else:
        st.dataframe(mine.drop(columns=["patient_id", "patient_name"]), hide_index=True)

    st.subheader("Long-term memory notes")
    notes = data.patient_notes(patient_id)
    if notes:
        st.dataframe(pd.DataFrame(notes[::-1]), hide_index=True)
    else:
        st.caption("No notes yet. The agent saves one after each booking or search for this patient.")


RECORD_FIELDS = [   # (column name, label) of the fields an attendant may edit
    ("diagnosis", "Diagnosis"), ("medications", "Medications"), ("lab_results", "Lab results"),
    ("alerts", "Alerts (allergies, warnings)"), ("doctor_notes", "Doctor's notes")]
SUMMARY_NOTE = ("The patient's stored summary refreshes by itself the next time their history "
                "is requested in the chat.")


def patient_choices():
    """{"Ramesh Kumar (P001)": "P001", ...} for a patient dropdown."""
    return {f"{r['name']} ({r['patient_id']})": r["patient_id"] for _, r in data.patients().iterrows()}


def show_changes(will_write):
    """Show what a record update would change as an old -> new table."""
    st.dataframe(pd.DataFrame([{"field": f, "old value": c["old"], "new value": c["new"]}
                               for f, c in will_write.items()]), hide_index=True)


def page_records():
    st.header("Medical records")
    st.caption("Attendant form. Every change is previewed first and saved only after you confirm. "
               "Patient ID and record ID can never be changed.")
    add_tab, update_tab = st.tabs(["Add a record", "Update a record"])
    with add_tab:
        form_add_record()
    with update_tab:
        form_update_record()


def form_add_record():
    people = patient_choices()
    patient_id = people[st.selectbox("Patient", list(people), key="add_patient")]
    today = config.get_now().date()
    values = {"patient_id": patient_id,
              "visit_date": str(st.date_input("Visit date", value=today, max_value=today, key="add_date")),
              "diagnosis": st.text_input("Diagnosis (required)", key="add_diagnosis"),
              "medications": st.text_input("Medications", key="add_medications"),
              "lab_results": st.text_input("Lab results", key="add_lab_results"),
              "alerts": st.text_input("Alerts (allergies, warnings)", key="add_alerts"),
              "doctor_notes": st.text_area("Doctor's notes", key="add_doctor_notes")}

    if st.button("Preview record", key="add_preview_button"):
        # confirm=False: the tool checks everything and writes NOTHING.
        st.session_state.add_preview = {"values": values, "result": add_record(**values)}
    preview = st.session_state.get("add_preview")
    if not preview:
        return
    result = preview["result"]
    if result["status"] != "needs_confirmation":
        st.error(result["message"])
        return
    st.markdown("**This will be saved:**")
    st.dataframe(pd.DataFrame([result["will_write"]]), hide_index=True)
    if preview["values"] != values:
        st.warning("The form changed after the preview. Press 'Preview record' again.")
    elif st.button("Confirm and save", type="primary", key="add_confirm_button"):
        saved = add_record(**preview["values"], confirm=True)   # saves exactly what was previewed
        st.session_state.pop("add_preview", None)
        if saved["status"] == "success":
            st.success(f"{saved['message']} {SUMMARY_NOTE}")
        else:
            st.error(saved["message"])


def form_update_record():
    people = patient_choices()
    patient_id = people[st.selectbox("Patient", list(people), key="update_patient")]
    records = data.patient_records(patient_id)
    if records.empty:
        st.info("This patient has no records yet. Use 'Add a record' instead.")
        return
    labels = {f"{r['record_id']} · {r['visit_date']} · {r['diagnosis']}": r for _, r in records.iterrows()}
    record = labels[st.selectbox("Record", list(labels), key=f"update_record_{patient_id}")]
    rid = record["record_id"]

    with st.container(horizontal=True):   # the two IDs are shown but cannot be edited
        st.text_input("Record ID (cannot be changed)", value=rid, disabled=True, key=f"update_rid_{rid}")
        st.text_input("Patient ID (cannot be changed)", value=patient_id, disabled=True, key=f"update_pid_{rid}")
    today = config.get_now().date()
    current_date = pd.to_datetime(record["visit_date"]).date()
    # Keys include the record ID, so choosing another record shows ITS values.
    values = {"visit_date": str(st.date_input("Visit date", value=current_date,
                                              max_value=max(today, current_date), key=f"update_date_{rid}"))}
    for field, label in RECORD_FIELDS:
        widget = st.text_area if field == "doctor_notes" else st.text_input
        values[field] = widget(label, value=record[field], key=f"update_{field}_{rid}")

    if st.button("Preview changes", key=f"update_preview_button_{rid}"):
        st.session_state.update_preview = {"record_id": rid, "values": values,
                                           "result": update_record(rid, values)}   # confirm=False
    preview = st.session_state.get("update_preview")
    if not preview or preview["record_id"] != rid:
        return
    result = preview["result"]
    if result["status"] == "no_change":
        st.info("Nothing was changed: the form has the same values as the saved record.")
        return
    if result["status"] != "needs_confirmation":
        st.error(result["message"])
        return
    st.markdown("**These changes will be saved:**")
    show_changes(result["will_write"])
    if preview["values"] != values:
        st.warning("The form changed after the preview. Press 'Preview changes' again.")
    elif st.button("Confirm and save", type="primary", key=f"update_confirm_button_{rid}"):
        saved = update_record(rid, preview["values"], confirm=True)   # saves exactly what was previewed
        st.session_state.pop("update_preview", None)
        if saved["status"] == "success":
            st.success(f"{saved['message']} {SUMMARY_NOTE}")
        else:
            st.error(saved["message"])


def page_doctors():
    st.header("Doctors")
    docs = data.doctors()
    labels = {f"{r['name']} - {r['specialty']}": r["doctor_id"] for _, r in docs.iterrows()}
    doctor_id = labels[st.selectbox("Doctor", list(labels))]
    d = docs.set_index("doctor_id").loc[doctor_id]
    slots = data.slots_with_state()
    mine = slots[slots["doctor_id"] == doctor_id]
    counts = mine["state"].value_counts()

    with st.container(horizontal=True):
        st.metric("Specialty", d["specialty"], border=True)
        st.metric("Clinic", d["clinic"], border=True)
        st.metric("Experience", f"{d['experience_years']} years", border=True)
        st.metric("Open upcoming slots", int(counts.get("available", 0)), border=True)
    if counts.get("available", 0) == 0:
        st.warning(f"{d['name']} has no open upcoming slots. All are booked or in the past.")

    st.subheader("Calendar")
    st.caption("available = can be booked · booked = taken · past = time has passed, cannot be booked")
    appts = data.appointments_table()
    confirmed = appts[appts["status"] == "confirmed"][["slot_id", "patient_name"]] if not appts.empty \
        else pd.DataFrame(columns=["slot_id", "patient_name"])
    calendar = mine.merge(confirmed, on="slot_id", how="left").fillna({"patient_name": ""})
    st.dataframe(calendar[["slot_id", "date", "time", "state", "patient_name"]], hide_index=True)


def page_appointments():
    st.header("Appointments")
    st.caption("Updates automatically every 5 seconds, so bookings made in the Assistant appear here.")

    @st.fragment(run_every="5s")
    def live_table():
        appts = data.appointments_table()
        slots = data.slots_with_state()
        upcoming = appts[(appts["status"] == "confirmed") & (appts["when"] == "upcoming")]
        waiting = data.waitlist_table()
        waiting_now = waiting[waiting["status"] == "waiting"]
        with st.container(horizontal=True):
            st.metric("Upcoming confirmed", len(upcoming), border=True)
            st.metric("Cancelled", int((appts["status"] == "cancelled").sum()), border=True)
            st.metric("Open slots (all doctors)", int((slots["state"] == "available").sum()), border=True)
            st.metric("On the waiting list", len(waiting_now), border=True)
        # Appointment tracker on the left, the waiting list right next to it.
        left, right = st.columns([3, 2])
        with left:
            show = st.segmented_control("Show", ["Upcoming", "All"], default="Upcoming", key="appt_filter")
            st.dataframe(upcoming if show == "Upcoming" else appts, hide_index=True)
        with right:
            st.markdown("**Waiting list**")
            if waiting.empty:
                st.caption("Nobody is on the waiting list.")
            else:
                st.dataframe(waiting.drop(columns=["patient_id"]), hide_index=True)
        st.caption(f"Last updated {pd.Timestamp.now():%H:%M:%S}")
    live_table()

    st.subheader("Cancel an appointment")
    appts = data.appointments_table()
    upcoming = appts[(appts["status"] == "confirmed") & (appts["when"] == "upcoming")]
    if upcoming.empty:
        st.caption("No upcoming appointments to cancel.")
        return
    labels = {f"{r['appointment_id']}: {r['patient_name']} with {r['doctor_name']}, {r['date']} {r['time']}":
              r["appointment_id"] for _, r in upcoming.iterrows()}
    choice = labels[st.selectbox("Appointment", list(labels))]
    from tools.appointment_tool import cancel_appointment
    if st.button("Preview cancellation"):
        st.session_state.cancel_preview = cancel_appointment(choice)   # writes nothing
    preview = st.session_state.get("cancel_preview")
    if preview and preview.get("will_write", {}).get("appointment", {}).get("appointment_id") == choice:
        st.info(preview["message"])
        st.json(preview["will_write"])
        if st.button("Confirm cancellation", type="primary"):
            result = cancel_appointment(choice, confirm=True)
            st.session_state.pop("cancel_preview", None)
            (st.success if result["status"] == "success" else st.error)(result["message"])


def page_medical_info():
    st.header("Medical information")
    st.caption("All disease information comes from MedlinePlus (U.S. National Library of Medicine). "
               "It is general patient information, not the newest research.")
    agent = st.session_state.get("agent")
    latest = agent.last_disease_info if agent else None
    st.subheader("Latest summary in this conversation")
    if latest:
        st.text(latest["answer"])
        with st.expander(f"The {len(latest['passages'])} retrieved passages the summary is based on"):
            for p in latest["passages"]:
                st.markdown(f"**[{p['source_number']}] [{p['title']}]({p['link']})**")
                st.text(p["text"])
    else:
        st.caption("Nothing yet. Ask the Assistant about a disease, e.g. 'What are the treatment options for asthma?'")

    st.subheader("Recent MedlinePlus searches (24-hour cache)")
    searches = data.recent_searches()
    if not searches:
        st.caption("No searches yet.")
    for s in searches:
        titles = ", ".join(f"[{r['title']}]({r['link']})" for r in s["results"]) or "_no results_"
        st.markdown(f"- **{s['query']}** · {data.format_time(s['saved_at'])} · {titles}")


def page_evaluation():
    st.header("Evaluation")
    summary, results = data.latest_evaluation()
    if summary is None:
        st.info("No evaluation results yet. Run `python -m evaluation.evaluate`.")
        return
    m = summary["metrics_from_log"]
    ap, bsr, refusals = m["answer_precision"], m["booking_success_rate"], m.get("booking_refusals", {})

    def pct(value):
        return "n/a" if value is None else f"{value:.0%}"

    with st.container(horizontal=True):
        st.metric("Answer precision", pct(ap["precision"]), border=True,
                  help="Answers the judge graded CORRECT / answers graded")
        st.caption(f"{ap['correct']} of {ap['graded']}")
        st.metric("Booking success rate", pct(bsr["rate"]), border=True,
                  help="Bookings saved / bookings the user said yes to")
        st.caption(f"{bsr['saved']} of {bsr['confirmed']}")
        st.metric("Correct refusals", refusals.get("total", "n/a"), border=True,
                  help="Requests that rightly ended without a booking (fully booked, declined, ...)")
        st.metric("Booking checks", f"{summary['booking_checks_passed']}/{summary['questions']}", border=True,
                  help="Code check of appointment.csv before and after each question")
    st.caption(f"Run {summary['run_at']} · {summary['questions']} questions · agent {summary['agent_model']} · "
               f"judge {summary['judge_model']} · clock frozen at {summary['reference_now']}")
    if refusals.get("by_reason"):
        st.markdown("**Correct refusals by reason:** "
                    + ", ".join(f"{reason}: {n}" for reason, n in refusals["by_reason"].items()))

    st.subheader("Results per question")
    st.dataframe(results[["id", "edge_case", "category", "judge_grade", "booking_check", "seconds", "question"]],
                 hide_index=True)
    for _, r in results.iterrows():
        with st.expander(f"{r['id']} · {r['judge_grade']} · {r['question']}"):
            st.markdown(f"**Expected answer:** {r['expected_answer']}")
            st.markdown("**Judge's reasoning**")
            st.text(r["judge_reasoning"])
            st.markdown("**Conversation**")
            st.text(r["agent_conversation"])

    st.subheader("Time per agent step (top-level calls)")
    per_tool = pd.DataFrame([{"step": tool, "calls": s["calls"], "avg_ms": s["avg_ms"],
                              "no_error_rate": s["no_error_rate"], "statuses": str(s["statuses"])}
                             for tool, s in m["per_tool"].items()])
    chart = (alt.Chart(per_tool)
             .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, size=14)
             .encode(x=alt.X("avg_ms:Q", title="Average time (ms)"),
                     y=alt.Y("step:N", sort="-x", title=None),
                     tooltip=["step", "calls", "avg_ms", "statuses"])
             .properties(height=36 * len(per_tool)))
    st.altair_chart(chart)
    st.dataframe(per_tool, hide_index=True)

    st.subheader("This dashboard's own log")
    live = data.live_metrics()
    st.caption(f"{live['total_logged_calls']} logged steps · booking success rate "
               f"{pct(live['booking_success_rate']['rate'])} · correct refusals "
               f"{live['booking_refusals']['total']}")


def page_memory_and_logs():
    st.header("Memory & logs")
    agent = get_agent()

    st.subheader("Try a scenario")
    text = st.text_input("Send a message to the assistant (same conversation as the Assistant page)",
                         key="scenario_box")
    if st.button("Send", disabled=not text):
        send_to_agent(text)
    if st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(as_chat_text(st.session_state.chat[-1]["text"]))

    st.subheader("Planning breakdown")
    show_plan_and_trace()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Short-term memory")
        st.caption(f"Last {agent.memory.max_messages} messages of this conversation, plus context.")
        st.text(agent.memory.as_text() or "(empty)")
        st.json({k: v for k, v in agent.memory.context.items() if k != "patient"} or {})
    with col2:
        st.subheader("Long-term memory")
        people = data.patients()
        current = agent.memory.get_context("patient_id")
        ids = list(people["patient_id"])
        patient_id = st.selectbox("Patient", ids, index=ids.index(current) if current in ids else 0,
                                  format_func=lambda pid: f"{people.set_index('patient_id').at[pid, 'name']} ({pid})")
        notes = data.patient_notes(patient_id)
        st.dataframe(pd.DataFrame(notes[::-1]), hide_index=True) if notes else st.caption("No notes yet.")

    st.subheader("Tool-call log (newest first)")
    log = data.log_table()
    if log.empty:
        st.caption("No logged steps yet.")
        return
    tools = st.multiselect("Filter by step", sorted(log["tool_name"].unique()))
    st.dataframe(log[log["tool_name"].isin(tools)] if tools else log, hide_index=True)


def show_recheck_banners():
    """At the top of every page: waiting-list entries whose recheck date has come.
    "Check availability" runs the normal slot search (the logged list_available_slots tool)."""
    groups = {}
    for item in data.due_rechecks():
        groups.setdefault((item["doctor_id"], item["specialty"], item["doctor_label"]), []).append(item)
    dismissed = st.session_state.setdefault("recheck_dismissed", set())
    for (doctor_id, specialty, label), items in groups.items():
        key = f"recheck_{doctor_id or specialty}"
        if key in dismissed:
            continue
        with st.container(border=True):
            st.warning(f"{label} may have openings now. Check availability?")
            st.caption("Waiting: " + ", ".join(f"{i['patient_name']} ({i['specialty']}, recheck {i['recheck_on']})"
                                                for i in items))
            with st.container(horizontal=True):
                if st.button("Check availability", key=f"{key}_check"):
                    st.session_state[f"{key}_result"] = (
                        list_available_slots(doctor_id=doctor_id) if doctor_id
                        else list_available_slots(specialty=specialty))
                if st.button("Dismiss", key=f"{key}_dismiss"):
                    dismissed.add(key)
                    st.rerun()
            result = st.session_state.get(f"{key}_result")
            if result and result["status"] == "success":
                st.success(result["message"] + " The earliest ones:")
                st.dataframe(pd.DataFrame(result["slots"][:5])[["slot_id", "date", "time", "doctor_name"]],
                             hide_index=True)
            elif result:
                st.info(result["message"])


# ---------------------------------------------------------------------------
# Layout: sidebar + chosen page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 Health Assistant Helexa")
    st.warning(config.DISCLAIMER)
    page = st.radio("Page", PAGES, key="page")
    st.divider()
    if data.using_original_data():
        st.error("Working on the ORIGINAL data/ folder.")
    else:
        st.caption(f"Working on a demo copy: `{config.DATA_DIR.relative_to(PROJECT_ROOT)}`")
        if st.button("Reset demo data"):
            st.session_state.confirm_reset = True
        if st.session_state.get("confirm_reset"):
            st.caption("Copies the original CSVs back, clears long-term memory and creates "
                       "future slots again. The log is kept.")
            with st.container(horizontal=True):
                if st.button("Yes, reset", type="primary"):
                    data.reset_demo_data()
                    new_conversation()
                    slots_after_reset = ensure_future_slots()   # the originals' slots are in the past
                    st.session_state.confirm_reset = False
                    st.success(f"Demo data reset. {slots_after_reset['message']}")
                if st.button("Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()
    st.caption(f"Future slots: {startup_slots['message']}")
    st.caption(f"Model: {config.LLM_MODEL}")

st.info(f"⚠️ {config.DISCLAIMER}")
show_recheck_banners()
{"Health Assistant Helexa": page_assistant, "Patients": page_patients, "Records": page_records,
 "Doctors": page_doctors,
 "Appointments": page_appointments, "Medical info": page_medical_info,
 "Evaluation": page_evaluation, "Memory & logs": page_memory_and_logs}[page]()
