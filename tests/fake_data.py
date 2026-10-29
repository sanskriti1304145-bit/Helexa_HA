"""
fake_data.py - hand-written patient summaries used by tests instead of real
LLM output, so tests are offline and free. They follow the CSV data.
"""

# Hand-written summaries for P001..P011 (P012 has no records, so no LLM call).
FAKE_SUMMARIES = [
    "Ramesh Kumar (P001), 70-year-old male, has hypertension since 2023 that progressed to chronic kidney disease stage 3b by April 2026 (creatinine 2.4 mg/dL, eGFR 31, potassium 5.2). Current medications are Losartan 100mg and sodium bicarbonate. Alerts: penicillin allergy; avoid NSAIDs; monitor potassium. Nephrology follow-up every 6 months is recommended and dialysis planning was discussed.",
    "Anita Sharma (P002), 34-year-old female, has generalized anxiety disorder treated with Sertraline 50mg. Symptoms were improving in March 2026. No known allergies. Review planned in 6 months.",
    "John Miller (P003), 45-year-old male, has chronic insomnia linked to work stress, taking Melatonin 3mg with little improvement. No known allergies. Referred for cognitive behavioural therapy for insomnia.",
    "Sunita Verma (P004), 62-year-old female, has type 2 diabetes with early diabetic nephropathy (creatinine 1.3 mg/dL, urine albumin-creatinine ratio 85 mg/g). Takes Metformin 500mg and Losartan 25mg. Sulfa drug allergy. Repeat kidney tests advised in 6 months.",
    "Robert Davis (P005), 58-year-old male, has hypertension and major depressive disorder (PHQ-9 score 14). Takes Escitalopram 10mg and Lisinopril 10mg. Alert: monitor blood pressure. Review in 3 months.",
    "Priya Patel (P006), 29-year-old female, has stress-related tension headaches treated with paracetamol as needed. Latex allergy. Relaxation techniques and counselling suggested.",
    "Michael Johnson (P007), 41-year-old male, has panic disorder treated with Escitalopram 10mg; attacks became less frequent by June 2026. No known allergies. Therapy sessions were added.",
    "Linda Garcia (P008), 52-year-old female, has hypothyroidism on Levothyroxine 75mcg with normal TSH 3.1 mIU/L, plus low mood and poor sleep. Aspirin allergy. Referred to psychiatry for screening.",
    "Arvind Iyer (P009), 67-year-old male, has atrial fibrillation with anxiety, taking Apixaban 5mg and Sertraline 25mg. Alerts: bleeding risk; avoid NSAIDs; review antidepressant choice with cardiology. A medication interaction review was requested.",
    "Emma Wilson (P010), 38-year-old female, has chronic insomnia for over 6 months, given a short course of Zolpidem 5mg. No known allergies. Therapy recommended.",
    "Rakesh Kumar (P011), 48-year-old male, has type 2 diabetes and dyslipidemia (LDL 162 mg/dL, HbA1c 7.4%). Takes Atorvastatin 20mg and Metformin 500mg. No known allergies. Diabetes control is improving.",
]
