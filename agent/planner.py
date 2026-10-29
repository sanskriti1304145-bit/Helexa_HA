"""
planner.py - breaks a user request into ordered sub-goals and picks a tool for each.

Two parts:

1. extract_patient_facts(text)   (simple rules, no LLM)
   Pulls "hard facts" about the patient out of the request:
       "My 70-year-old father ..."  ->  {"age": 70, "gender": "male"}
   Rules are used instead of the LLM because these facts filter real patient
   data, so they must be predictable and testable. If something is unclear
   (two different ages, or both "father" and "mother"), we return None for it
   rather than guess.

2. make_plan(request, memory, llm)   (LLM)
   Asks the LLM for a plan in JSON, checks the JSON against a schema
   (pydantic), then fixes the order with simple rules (validate_plan), e.g.
   identify_patient always comes before patient-specific steps.

Plan steps use these tool names (the agent maps each to the real tools):
    identify_patient     -> search_patient_summaries / find_patient + patients.csv check
    get_patient_history  -> get_patient_history + stored FAISS summary + long-term notes
    book_appointment     -> list_available_slots + book_appointment (with confirmation)
    search_disease_info  -> search_medlineplus + RAG summary with citations
"""

import re
from typing import Literal, Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import config
from agent.prompts import PLAN_PROMPT
from memory.conversation_memory import recall_patient_memory
from tools.csv_helpers import read_csv
from tools.logger import log_tool_call

# ---------------------------------------------------------------------------
# 1. Extract age and gender with rules
# ---------------------------------------------------------------------------

# Words that tell us the patient's gender. Pronouns count too ("book for him").
MALE_WORDS = {"father", "dad", "daddy", "husband", "son", "brother", "grandfather",
              "grandpa", "uncle", "nephew", "boyfriend", "man", "boy", "he", "him", "his"}
FEMALE_WORDS = {"mother", "mom", "mum", "mummy", "wife", "daughter", "sister",
                "grandmother", "grandma", "aunt", "niece", "girlfriend", "woman",
                "girl", "she", "her", "hers"}

# Matches "70-year-old", "70 year old", "70 years old", "aged 70", "age 70", "age: 70"
AGE_PATTERNS = [
    r"\b(\d{1,3})\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b",
    r"\baged?\s*:?\s*(\d{1,3})\b",
]


def extract_patient_facts(text):
    """Return {"age": int or None, "gender": "male"/"female"/None} from the request."""
    lowered = str(text or "").lower()

    ages = set()
    for pattern in AGE_PATTERNS:
        ages.update(int(a) for a in re.findall(pattern, lowered))
    ages = {a for a in ages if 0 <= a <= 120}
    age = ages.pop() if len(ages) == 1 else None   # 0 or 2+ different ages -> unclear

    words = set(re.findall(r"[a-z]+", lowered))
    is_male = bool(words & MALE_WORDS)
    is_female = bool(words & FEMALE_WORDS)
    gender = "male" if is_male and not is_female else "female" if is_female and not is_male else None

    return {"age": age, "gender": gender}


# ---------------------------------------------------------------------------
# 2. The plan format (checked by pydantic)
# ---------------------------------------------------------------------------

ToolName = Literal["identify_patient", "get_patient_history", "book_appointment", "search_disease_info"]
PATIENT_STEPS = ("get_patient_history", "book_appointment")   # need a known patient


class PlanStep(BaseModel):
    goal: str = Field(description="One short sentence: what this step achieves")
    tool: ToolName = Field(description="The single tool to use for this step")
    input: str = Field(default="", description="Tool input: name, specialty or search topic")


# The planner first CLASSIFIES every message. Only "in_scope" messages get steps;
# the agent answers all other categories with a fixed template.
Category = Literal["in_scope", "greeting", "cancel", "medical_advice", "unclear", "off_topic"]


class Plan(BaseModel):
    category: Category = Field(default="in_scope", description="What kind of message this is")
    patient_name: Optional[str] = Field(default=None, description="Patient's name if the user gave it, else null")
    condition: Optional[str] = Field(default=None, description="Main condition mentioned, else null")
    steps: list[PlanStep] = Field(description="Ordered sub-goals")


plan_parser = PydanticOutputParser(pydantic_object=Plan)


def validate_plan(plan, patient_known=False):
    """Fix the plan's order with simple rules, so a slightly wrong LLM plan is still safe.

    - remove duplicate steps (same tool and input)
    - if a patient step exists and no patient is known yet, identify_patient goes first
    - get_patient_history comes before book_appointment
    """
    steps, seen = [], set()
    for step in plan.steps:
        key = (step.tool, step.input.strip().lower())
        if key not in seen:
            seen.add(key)
            steps.append(step)

    needs_patient = any(s.tool in PATIENT_STEPS for s in steps)
    identify = [s for s in steps if s.tool == "identify_patient"]
    others = [s for s in steps if s.tool != "identify_patient"]
    if needs_patient and not patient_known and not identify:
        identify = [PlanStep(goal="Identify the patient", tool="identify_patient",
                             input=plan.patient_name or "")]
    if not needs_patient and patient_known:
        identify = []   # nothing patient-specific to do

    # Keep the LLM's order, but make sure history comes before booking.
    history = [s for s in others if s.tool == "get_patient_history"]
    rest = [s for s in others if s.tool != "get_patient_history"]
    if any(s.tool == "book_appointment" for s in rest) and not history:
        history = [PlanStep(goal="Retrieve the patient's medical history", tool="get_patient_history")]
    first_booking = next((i for i, s in enumerate(rest) if s.tool == "book_appointment"), len(rest))
    ordered = identify + rest[:first_booking] + history + rest[first_booking:]

    return Plan(patient_name=plan.patient_name, condition=plan.condition, steps=ordered)


# ---------------------------------------------------------------------------
# 3. Ask the LLM for a plan
# ---------------------------------------------------------------------------

@log_tool_call
def make_plan(request, conversation="", current_patient="none", patient_id=None, llm=None):
    """Ask the LLM to plan the request. Tries twice if the JSON is not valid.

    conversation:    recent messages (short-term memory)
    current_patient: e.g. "Ramesh Kumar (P001, age 70)" if already confirmed
    patient_id:      if given, the patient's long-term notes are looked up
                     and shown to the LLM (memory lookup in the prompt)

    Returns {"status": "success", "plan": Plan, "facts": {...}} or an error.
    """
    patient_known = bool(patient_id)
    patient_notes = "none"
    if patient_id:
        recalled = recall_patient_memory(patient_id, limit=5)
        if recalled["status"] == "success":
            patient_notes = "\n".join(f"- {n['timestamp']} ({n['type']}): {n['text']}"
                                      for n in recalled["notes"])
    if llm is None:
        from agent.llm import get_llm
        llm = get_llm()
    specialties = ", ".join(sorted(read_csv(config.DOCTORS_CSV)["specialty"].unique()))
    chain = PLAN_PROMPT | llm

    last_error = ""
    for _attempt in range(2):
        try:
            reply = chain.invoke({
                "request": request,
                "conversation": conversation or "(none)",
                "current_patient": current_patient,
                "patient_notes": patient_notes,
                "specialties": specialties,
                "format_instructions": plan_parser.get_format_instructions(),
            })
            plan = plan_parser.parse(reply.text)   # raises if not valid JSON for Plan
            if plan.category != "in_scope":
                # Scope guard: not a request we run. Any steps the LLM added are ignored.
                return {"status": "out_of_scope", "category": plan.category,
                        "message": f"The message was classified as {plan.category}.",
                        "plan": plan, "facts": extract_patient_facts(request)}
            plan = validate_plan(plan, patient_known=patient_known)
            if not plan.steps:
                return {"status": "no_steps",
                        "message": "I could not find a task I can help with in that request.",
                        "plan": plan, "facts": extract_patient_facts(request)}
            return {"status": "success",
                    "message": f"Planned {len(plan.steps)} step(s): "
                               + "; ".join(f"{i}. {s.tool}" for i, s in enumerate(plan.steps, 1)),
                    "plan": plan,
                    "facts": extract_patient_facts(request)}
        except Exception as error:
            last_error = str(error)
    return {"status": "error", "message": f"The planner did not return a valid plan: {last_error[:300]}"}
