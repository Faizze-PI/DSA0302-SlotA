"""
Corpus and Resource Generator
DSA03 - Natural Language Processing (Assignment 1)
Generates:
1. 500 de-identified clinical records (>= 25,000 tokens, 4 record types)
2. Morphological Lexicon (>= 300 stems, >= 20 inflectional, >= 15 derivational, irregulars)
3. Hand-annotated set of 200 clinical word forms
4. 40 clinician evaluation queries with document relevance ground truth
5. Day-4 challenge evaluation records (OCR corrupted and code-mixed)
"""

import json
import os
import random
import re

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RECORD_TYPES = [
    "Discharge Summary",
    "Prescription",
    "Operation Note",
    "Nursing Observation"
]

PATIENTS = [f"PID-{10000 + i}" for i in range(1200)]
DOCTORS = ["Dr. K. Ramanathan", "Dr. S. Priya", "Dr. V. Anand", "Dr. M. Deepa", "Dr. R. Karthik", "Dr. N. Meenakshi"]
WARDS = ["Cardiology Ward", "General Surgery", "Orthopaedic Unit", "Pulmonology Bay", "Nephrology Ward", "ICU-A", "ICU-B", "Emergency Care"]

DIAGNOSES = [
    ("Essential hypertension", "I10"),
    ("Type 2 diabetes mellitus", "E11.9"),
    ("Lobar pneumonia", "J18.9"),
    ("Acute appendicitis", "K35.8"),
    ("Gastro-oesophageal reflux disease", "K21.0"),
    ("Closed fracture of right femoral shaft", "S72.3"),
    ("Chronic kidney disease stage 4", "N18.4"),
    ("Acute myocardial infarction", "I21.9"),
    ("Bronchial asthma with acute exacerbation", "J45.9"),
    ("Urinary tract infection", "N39.0"),
    ("Osteoarthritis of left knee", "M17.1"),
    ("Acute cholecystitis", "K81.0"),
    ("Ischaemic stroke", "I63.9"),
    ("Dermatitis due to drugs", "L27.0"),
    ("Lumbar disc herniation with radiculopathy", "M51.1")
]

DRUGS = [
    ("Paracetamol", "500mg", "t.i.d.", "oral"),
    ("Amoxicillin-Clavulanate", "625mg", "b.i.d.", "oral"),
    ("Metformin", "500mg", "b.i.d.", "oral"),
    ("Atorvastatin", "20mg", "once daily", "oral"),
    ("Pantoprazole", "40mg", "once daily before breakfast", "oral"),
    ("Ceftriaxone", "1g", "every 12 hours", "IV"),
    ("Tramadol", "50mg", "every 8 hours prn", "oral"),
    ("Furosemide", "40mg", "once daily morning", "oral"),
    ("Amlodipine", "5mg", "once daily", "oral"),
    ("Insulin Glargine", "10 units", "at bedtime", "subcutaneous"),
    ("Enoxaparin", "40mg", "once daily", "subcutaneous"),
    ("Azithromycin", "500mg", "once daily", "oral"),
    ("Ciprofloxacin", "500mg", "b.i.d.", "oral"),
    ("Salbutamol nebulisation", "2.5mg", "every 6 hours prn", "inhalation"),
    ("Ondansetron", "4mg", "every 8 hours prn", "IV")
]

SURGERIES = [
    ("Laparoscopic appendectomy", "uncomplicated acute appendicitis", "General endotracheal"),
    ("Open reduction and internal fixation", "fracture shaft of femur", "Spinal anaesthesia"),
    ("Right simple nephrectomy", "non-functioning atrophic kidney", "General anaesthesia"),
    ("Diagnostic bronchoscopy and bronchial wash", "persistent pulmonary infiltrate", "Local with sedation"),
    ("Laparoscopic cholecystectomy", "symptomatic cholelithiasis", "General anaesthesia"),
    ("Total knee arthroplasty", "severe tricompartmental osteoarthritis", "Combined spinal-epidural"),
    ("Percutaneous coronary intervention with drug-eluting stent", "coronary artery occlusion", "Local vascular access"),
    ("Debridement and primary closure of traumatic laceration", "contaminated soft tissue injury", "Local field block")
]

SYMPTOMS = [
    ("fever with chills", "for 3 days"),
    ("productive cough with yellowish expectoration", "for 1 week"),
    ("acute stabbing pain in right lower abdomen", "for 18 hours"),
    ("diffuse shortness of breath on exertion", "duration 4 weeks"),
    ("severe pain and inability to bear weight on right leg", "since 2 days"),
    ("burning micturition and increased frequency", "for 5 days"),
    ("throbbing frontal headache associated with nausea", "for 48 hours"),
    ("epigastric discomfort relieved by antacids", "duration 3 months"),
    ("bilateral pedal edema", "since 2 weeks"),
    ("generalised lethargy and loss of appetite", "for 10 days")
]

def make_discharge_summary(rec_id):
    pid = random.choice(PATIENTS)
    age = random.randint(18, 85)
    contact = f"+91-9{random.randint(100000000, 999999999)}"
    diag_desc, icd = random.choice(DIAGNOSES)
    symptom, sym_dur = random.choice(SYMPTOMS)
    doc = random.choice(DOCTORS)
    ward = random.choice(WARDS)
    bed = f"Bed {random.randint(1, 40)}"
    
    adm_day = random.randint(1, 20)
    adm_month = random.randint(1, 10)
    dis_day = adm_day + random.randint(2, 7)
    adm_date = f"2024-{adm_month:02d}-{adm_day:02d}"
    dis_date = f"2024-{adm_month:02d}-{dis_day:02d}"
    
    bp_sys = random.randint(105, 165)
    bp_dia = random.randint(65, 100)
    temp = round(random.uniform(98.0, 102.2), 1)
    hb = round(random.uniform(9.5, 15.8), 1)
    wbc = random.randint(4500, 14800)
    creat = round(random.uniform(0.7, 3.2), 2)
    
    selected_drugs = random.sample(DRUGS, 3)
    drug_lines = " ".join([f"Prescribed {d[0]} {d[1]} {d[2]} via {d[3]} route." for d in selected_drugs])
    
    text = (
        f"DISCHARGE SUMMARY\n"
        f"Patient ID: {pid} | Age: {age} yo | Contact: {contact}\n"
        f"Admission Date: {adm_date} | Discharge Date: {dis_date}\n"
        f"Attending Consultant: {doc} | Location: {ward}, {bed}\n"
        f"Primary Diagnosis: {diag_desc} (ICD-10 Code: {icd}).\n"
        f"History of Present Illness: Patient presented with complaints of {symptom} {sym_dur}. "
        f"Initial physical examination revealed blood pressure {bp_sys}/{bp_dia} mmHg, body temperature {temp} F, "
        f"pulse 82 bpm, and clear breath sounds. Laboratory workup on admission indicated Hb {hb} g/dL, "
        f"WBC {wbc} /uL, and serum Creatinine {creat} mg/dL. "
        f"Hospital Course and Treatment: Patient was admitted under {ward} and commenced on intravenous hydration, "
        f"targeted antimicrobial therapy, and analgesics. Daily clinical evaluations showed satisfactory systemic recovery. "
        f"Vital signs stabilized over the course of hospitalization. Post-treatment laboratory markers returned within acceptable limits. "
        f"Discharge Medications: {drug_lines} "
        f"Follow-up Advice: Patient advised to review in outpatient clinic in 10 days. Return immediately if high fever, worsening pain, or dyspnoea occurs."
    )
    return {
        "record_id": f"REC-{rec_id:04d}",
        "record_type": "Discharge Summary",
        "patient_id": pid,
        "admission_date": adm_date,
        "discharge_date": dis_date,
        "diagnosis": diag_desc,
        "icd_code": icd,
        "age": f"{age} yo",
        "contact": contact,
        "ward_bed": f"{ward}, {bed}",
        "blood_pressure": f"{bp_sys}/{bp_dia} mmHg",
        "temperature": f"{temp} F",
        "lab_values": [f"Hb {hb} g/dL", f"WBC {wbc} /uL", f"Creatinine {creat} mg/dL"],
        "drugs": [f"{d[0]} {d[1]}" for d in selected_drugs],
        "frequencies": [d[2] for d in selected_drugs],
        "duration": sym_dur,
        "text": text
    }

def make_prescription(rec_id):
    pid = random.choice(PATIENTS)
    age = random.randint(18, 75)
    doc = random.choice(DOCTORS)
    contact = f"+91-9{random.randint(100000000, 999999999)}"
    diag_desc, icd = random.choice(DIAGNOSES)
    symptom, sym_dur = random.choice(SYMPTOMS)
    ward = random.choice(WARDS)
    bed = f"Bed {random.randint(1, 35)}"
    
    day = random.randint(1, 28)
    month = random.randint(1, 11)
    presc_date = f"2024-{month:02d}-{day:02d}"
    
    bp_sys = random.randint(110, 150)
    bp_dia = random.randint(70, 95)
    temp = round(random.uniform(98.2, 100.8), 1)
    
    selected_drugs = random.sample(DRUGS, 4)
    med_lines = "\n".join([f"  - Rx: {d[0]} {d[1]} | Frequency: {d[2]} | Route: {d[3]} | Dispense: 10 days" for d in selected_drugs])
    
    text = (
        f"OUTPATIENT PRESCRIPTION & CLINICAL ENCOUNTER\n"
        f"Record ID: REC-{rec_id:04d} | Date: {presc_date}\n"
        f"Patient ID: {pid} | Age: {age} years old | Contact: {contact}\n"
        f"Clinic Unit: {ward}, {bed} | Prescribing Physician: {doc}\n"
        f"Clinical Indication: {diag_desc} (ICD Code: {icd}).\n"
        f"Chief Complaint: Patient reported {symptom} {sym_dur}.\n"
        f"Vitals Monitored: Blood pressure measured at {bp_sys}/{bp_dia} mmHg, oral temperature {temp} F.\n"
        f"Medication Orders:\n{med_lines}\n"
        f"Pharmacy Instructions: Administer oral medications with plenty of water after meals. Monitor for hypersensitivity. "
        f"Patient instructed to maintain hydration, follow diabetic diet restrictions, and report any adverse reactions immediately."
    )
    return {
        "record_id": f"REC-{rec_id:04d}",
        "record_type": "Prescription",
        "patient_id": pid,
        "admission_date": presc_date,
        "discharge_date": presc_date,
        "diagnosis": diag_desc,
        "icd_code": icd,
        "age": f"{age} years old",
        "contact": contact,
        "ward_bed": f"{ward}, {bed}",
        "blood_pressure": f"{bp_sys}/{bp_dia} mmHg",
        "temperature": f"{temp} F",
        "lab_values": [],
        "drugs": [f"{d[0]} {d[1]}" for d in selected_drugs],
        "frequencies": [d[2] for d in selected_drugs],
        "duration": sym_dur,
        "text": text
    }

def make_operation_note(rec_id):
    pid = random.choice(PATIENTS)
    age = random.randint(22, 78)
    contact = f"+91-9{random.randint(100000000, 999999999)}"
    surg_name, indic, anaesth = random.choice(SURGERIES)
    doc = random.choice(DOCTORS)
    ward = "Operating Suite 2"
    bed = f"Bed {random.randint(1, 10)}"
    
    day = random.randint(1, 25)
    month = random.randint(1, 10)
    op_date = f"2024-{month:02d}-{day:02d}"
    
    hb = round(random.uniform(10.2, 14.5), 1)
    wbc = random.randint(5500, 13200)
    creat = round(random.uniform(0.8, 1.8), 2)
    bp_sys = random.randint(110, 140)
    bp_dia = random.randint(70, 88)
    temp = round(random.uniform(97.8, 99.4), 1)
    
    text = (
        f"OPERATION AND SURGICAL THEATRE NOTE\n"
        f"Patient Identifier: {pid} | Age: {age} yo | Contact: {contact}\n"
        f"Date of Procedure: {op_date} | Location: {ward}, {bed}\n"
        f"Lead Surgeon: {doc} | Anaesthetist: Dr. V. Anand\n"
        f"Pre-Operative Diagnosis: Confirmed {indic} (ICD code: K35.8).\n"
        f"Procedure Performed: {surg_name}.\n"
        f"Anaesthesia Administered: {anaesth}.\n"
        f"Clinical History: Patient experienced severe symptoms duration 3 days before surgical decision. "
        f"Pre-incision vitals: blood pressure {bp_sys}/{bp_dia} mmHg, core temperature {temp} F, pre-op lab Hb {hb} g/dL, "
        f"total WBC count {wbc} /uL, and serum Creatinine {creat} mg/dL. "
        f"Surgical Narrative: Patient positioned supine on the operating table. Surgical site prepared with povidone-iodine and draped. "
        f"Incision placed with precise anatomic exposure. Hemostasis achieved electro-surgically. "
        f"Target pathology resected without intra-operative complication. Estimated blood loss minimal (less than 80 mL). "
        f"Post-Operative Orders: Administer Ceftriaxone 1g every 12 hours IV and Tramadol 50mg every 8 hours prn for acute wound pain. "
        f"Monitor urine output and wound drain volume hourly in recovery."
    )
    return {
        "record_id": f"REC-{rec_id:04d}",
        "record_type": "Operation Note",
        "patient_id": pid,
        "admission_date": op_date,
        "discharge_date": op_date,
        "diagnosis": indic,
        "icd_code": "K35.8",
        "age": f"{age} yo",
        "contact": contact,
        "ward_bed": f"{ward}, {bed}",
        "blood_pressure": f"{bp_sys}/{bp_dia} mmHg",
        "temperature": f"{temp} F",
        "lab_values": [f"Hb {hb} g/dL", f"WBC {wbc} /uL", f"Creatinine {creat} mg/dL"],
        "drugs": ["Ceftriaxone 1g", "Tramadol 50mg"],
        "frequencies": ["every 12 hours", "every 8 hours prn"],
        "duration": "duration 3 days",
        "text": text
    }

def make_nursing_observation(rec_id):
    pid = random.choice(PATIENTS)
    age = random.randint(20, 80)
    contact = f"+91-9{random.randint(100000000, 999999999)}"
    diag_desc, icd = random.choice(DIAGNOSES)
    symptom, sym_dur = random.choice(SYMPTOMS)
    ward = random.choice(WARDS)
    bed = f"Bed {random.randint(1, 30)}"
    
    day = random.randint(1, 28)
    month = random.randint(1, 11)
    obs_date = f"2024-{month:02d}-{day:02d}"
    
    bp_sys = random.randint(110, 160)
    bp_dia = random.randint(68, 98)
    temp = round(random.uniform(98.4, 101.6), 1)
    hb = round(random.uniform(9.0, 14.8), 1)
    wbc = random.randint(5000, 14000)
    creat = round(random.uniform(0.6, 2.8), 2)
    
    drug_name, dose, freq, route = random.choice(DRUGS)
    
    text = (
        f"NURSING CLINICAL OBSERVATION & WARD CHART\n"
        f"Encounter ID: REC-{rec_id:04d} | Observation Date: {obs_date}\n"
        f"Patient ID: {pid} | Age: {age} years old | Contact: {contact}\n"
        f"Station: {ward}, {bed} | Primary Problem: {diag_desc} (ICD Code: {icd})\n"
        f"Shift Assessment: Patient admitted with persistent {symptom} {sym_dur}. "
        f"Patient rested comfortably in fowler position throughout the shift. "
        f"Vital Signs Monitored: Recorded blood pressure {bp_sys}/{bp_dia} mmHg, axillary temperature {temp} F, "
        f"respiratory rate 18 breaths per minute, SpO2 98% on room air. "
        f"Daily laboratory profile reviewed: Hb {hb} g/dL, WBC {wbc} /uL, Creatinine {creat} mg/dL. "
        f"Medication Administration: Administered {drug_name} {dose} {freq} via {route} route with good patient tolerance. "
        f"No signs of phlebitis at peripheral IV cannula site. Intake and output charting maintained. "
        f"Call bell within easy reach. Night shift hand-off completed."
    )
    return {
        "record_id": f"REC-{rec_id:04d}",
        "record_type": "Nursing Observation",
        "patient_id": pid,
        "admission_date": obs_date,
        "discharge_date": obs_date,
        "diagnosis": diag_desc,
        "icd_code": icd,
        "age": f"{age} years old",
        "contact": contact,
        "ward_bed": f"{ward}, {bed}",
        "blood_pressure": f"{bp_sys}/{bp_dia} mmHg",
        "temperature": f"{temp} F",
        "lab_values": [f"Hb {hb} g/dL", f"WBC {wbc} /uL", f"Creatinine {creat} mg/dL"],
        "drugs": [f"{drug_name} {dose}"],
        "frequencies": [freq],
        "duration": sym_dur,
        "text": text
    }

def generate_clinical_corpus(total_records=500):
    generators = [
        make_discharge_summary,
        make_prescription,
        make_operation_note,
        make_nursing_observation
    ]
    records = []
    total_tokens = 0
    for i in range(1, total_records + 1):
        gen = generators[(i - 1) % len(generators)]
        rec = gen(i)
        words = re.findall(r"\S+", rec["text"])
        total_tokens += len(words)
        records.append(rec)
    return records, total_tokens

def build_morphological_lexicon():
    stems = [
        "fractur", "nephr", "cardio", "arthr", "gastr", "derm", "pulmon", "hepat", "surg", "infect",
        "inflamm", "oper", "biops", "transfus", "lesion", "patient", "symptom", "tendon", "ligament", "vessel",
        "artery", "vein", "muscl", "bronch", "colon", "pancrea", "cerebr", "cranial", "neuron", "synaps",
        "valv", "atrium", "ventricl", "aort", "capillar", "lymph", "spleen", "thymus", "thyroid", "adrenal",
        "pituitar", "hypothalam", "cortex", "medulla", "glomerul", "tubul", "ureter", "urethr", "vesic", "prostat",
        "ovari", "uter", "cervix", "vagin", "placent", "fetus", "embryo", "oste", "chondr", "myo",
        "fasci", "mening", "pleur", "peritone", "pericard", "endocard", "myocard", "myelin", "axon", "dendrit",
        "erythr", "leuk", "thromb", "macro", "phag", "eosin", "baso", "neutro", "monocyt", "histio",
        "hemato", "coagul", "fibrin", "plasmin", "thrombin", "albumin", "globulin", "ferritin", "bilirubin", "creatin",
        "ure", "ammoni", "lactat", "pyruvat", "glucos", "glycogen", "lipase", "amylase", "protease", "kinas",
        "phosphat", "dehydrogen", "transfer", "polymer", "ribosom", "mitochondr", "lysosom", "cytoplasm", "nucle", "chromosom",
        "gen", "allel", "mutat", "transcript", "translat", "peptid", "protein", "lipid", "cholesterol", "triglycerid",
        "stomach", "esophag", "duoden", "jejun", "ileum", "caecum", "append", "rectum", "anorect", "fistul",
        "abscess", "ulcer", "necros", "gangren", "ischemi", "infarct", "embol", "thrombus", "aneurysm", "stenos",
        "occlus", "perfor", "ruptur", "herni", "fissur", "strictur", "calcul", "lithias", "polyp", "tumour",
        "carcin", "sarcom", "adenom", "melanom", "lymphom", "leukemi", "blastom", "fibrom", "lipom", "angiom",
        "edem", "effus", "exudat", "transudat", "ascit", "icter", "pallor", "cyanot", "rubor", "dolor",
        "calor", "fever", "pyrex", "chill", "rigor", "diaphor", "nause", "vomit", "emesis", "dyspeps",
        "anorex", "asthen", "fatigu", "malaise", "syncope", "vertigo", "dizzin", "headach", "cephal", "migrain",
        "seizur", "epileps", "convuls", "tremor", "ataxi", "aphas", "dysarthr", "dysphag", "dyspne", "apne",
        "tachypne", "bradypne", "cough", "expector", "hemoptys", "stridor", "wheez", "rales", "rhonch", "pleurit",
        "palpitat", "tachycard", "bradycard", "arrhythm", "fibrill", "angina", "claudicat", "hypertens", "hypotens", "shock",
        "oliguri", "anuri", "polyuri", "dysuri", "hematuri", "proteinuri", "azotemi", "uremi", "seps", "bacter",
        "viral", "fungal", "parasit", "protozo", "helminth", "toxin", "septic", "pyogen", "purul", "granul",
        "phlegmon", "cellulit", "erythem", "prurit", "urticar", "macul", "papul", "vesicl", "bull", "pustul",
        "cicatr", "keloid", "atroph", "hypertroph", "hyperplas", "dysplas", "metaplas", "anaplas", "neoplas", "malig",
        "benign", "invas", "metastat", "remiss", "relaps", "recurren", "refractor", "intract", "congenit", "heredit",
        "famili", "idiopath", "iatrogen", "nosocomi", "acut", "subacut", "chronic", "fulmin", "insidi", "terminal",
        "prognos", "diagnos", "therap", "regimen", "protocol", "dosag", "titrat", "infus", "inject", "ingest",
        "absorpt", "excret", "clearanc", "metabol", "catabol", "anabol", "homeostas", "patholog", "physiolog", "anatom"
    ]

    inflectional_affixes = [
        {"affix": "-s", "type": "suffix", "feature": "+N +PL"},
        {"affix": "-es", "type": "suffix", "feature": "+N +PL"},
        {"affix": "-ed", "type": "suffix", "feature": "+V +PAST"},
        {"affix": "-ing", "type": "suffix", "feature": "+V +PROG"},
        {"affix": "-er", "type": "suffix", "feature": "+ADJ +COMP"},
        {"affix": "-est", "type": "suffix", "feature": "+ADJ +SUPERL"},
        {"affix": "-ies", "type": "suffix", "feature": "+N +PL (y-to-i)"},
        {"affix": "-en", "type": "suffix", "feature": "+V +PAST_PART"},
        {"affix": "-a", "type": "suffix", "feature": "+N +PL (Latin -um to -a)"},
        {"affix": "-i", "type": "suffix", "feature": "+N +PL (Latin -us to -i)"},
        {"affix": "-ae", "type": "suffix", "feature": "+N +PL (Latin -a to -ae)"},
        {"affix": "-ices", "type": "suffix", "feature": "+N +PL (Latin -ex/-ix to -ices)"},
        {"affix": "-ces", "type": "suffix", "feature": "+N +PL (Greek/Latin)"},
        {"affix": "-des", "type": "suffix", "feature": "+N +PL"},
        {"affix": "-ves", "type": "suffix", "feature": "+N +PL (f/fe to ves)"},
        {"affix": "-s (verb)", "type": "suffix", "feature": "+V +3SG +PRES"},
        {"affix": "-es (verb)", "type": "suffix", "feature": "+V +3SG +PRES"},
        {"affix": "-ing (participle)", "type": "suffix", "feature": "+ADJ +PART"},
        {"affix": "-ed (participle)", "type": "suffix", "feature": "+ADJ +PART"},
        {"affix": "-d", "type": "suffix", "feature": "+V +PAST (e-retention)"}
    ]

    derivational_affixes = [
        {"affix": "-itis", "type": "suffix", "meaning": "inflammation of", "cat_change": "N -> N"},
        {"affix": "-ectomy", "type": "suffix", "meaning": "surgical excision of", "cat_change": "N -> N"},
        {"affix": "-ology", "type": "suffix", "meaning": "study/branch of", "cat_change": "N -> N"},
        {"affix": "-osis", "type": "suffix", "meaning": "abnormal condition of", "cat_change": "N -> N"},
        {"affix": "-aemia", "type": "suffix", "meaning": "blood condition", "cat_change": "N/Adj -> N"},
        {"affix": "-ation", "type": "suffix", "meaning": "process or action", "cat_change": "V -> N"},
        {"affix": "-able", "type": "suffix", "meaning": "capable of being", "cat_change": "V -> Adj"},
        {"affix": "-ive", "type": "suffix", "meaning": "tending to", "cat_change": "V -> Adj"},
        {"affix": "-scopy", "type": "suffix", "meaning": "visual examination", "cat_change": "N -> N"},
        {"affix": "-plasty", "type": "suffix", "meaning": "surgical repair/reconstruction", "cat_change": "N -> N"},
        {"affix": "-megaly", "type": "suffix", "meaning": "abnormal enlargement", "cat_change": "N -> N"},
        {"affix": "-pathy", "type": "suffix", "meaning": "disease of", "cat_change": "N -> N"},
        {"affix": "-gram", "type": "suffix", "meaning": "recorded tracing/image", "cat_change": "N/V -> N"},
        {"affix": "nephro-", "type": "prefix", "meaning": "relating to kidney", "cat_change": "Root -> Compound"},
        {"affix": "hyper-", "type": "prefix", "meaning": "excessive/elevated", "cat_change": "Adj/N -> Adj/N"},
        {"affix": "hypo-", "type": "prefix", "meaning": "deficient/low", "cat_change": "Adj/N -> Adj/N"},
        {"affix": "re-", "type": "prefix", "meaning": "again/backward", "cat_change": "V -> V"},
        {"affix": "un-", "type": "prefix", "meaning": "not/opposite", "cat_change": "Adj/V -> Adj/V"},
        {"affix": "sub-", "type": "prefix", "meaning": "under/below", "cat_change": "Adj/N -> Adj/N"},
        {"affix": "pre-", "type": "prefix", "meaning": "before", "cat_change": "N/Adj -> Adj"},
        {"affix": "post-", "type": "prefix", "meaning": "after", "cat_change": "N/Adj -> Adj"}
    ]

    irregulars = [
        {"surface": "renal", "canonical_stem": "kidney", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "cardiac", "canonical_stem": "heart", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "hepatic", "canonical_stem": "liver", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "pulmonary", "canonical_stem": "lung", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "gastric", "canonical_stem": "stomach", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "ocular", "canonical_stem": "eye", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "cerebral", "canonical_stem": "brain", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "cutaneous", "canonical_stem": "skin", "relation": "adjectival_suppletion", "pos": "ADJ"},
        {"surface": "vertices", "canonical_stem": "vertex", "relation": "classical_plural", "pos": "N"},
        {"surface": "analyses", "canonical_stem": "analysis", "relation": "classical_plural", "pos": "N"},
        {"surface": "indices", "canonical_stem": "index", "relation": "classical_plural", "pos": "N"},
        {"surface": "thrombi", "canonical_stem": "thrombus", "relation": "classical_plural", "pos": "N"},
        {"surface": "erythrocytes", "canonical_stem": "erythrocyte", "relation": "compound_plural", "pos": "N"},
        {"surface": "bacteria", "canonical_stem": "bacterium", "relation": "classical_plural", "pos": "N"},
        {"surface": "protozoa", "canonical_stem": "protozoon", "relation": "classical_plural", "pos": "N"}
    ]

    return {
        "stems": sorted(list(set(stems))),
        "inflectional_affixes": inflectional_affixes,
        "derivational_affixes": derivational_affixes,
        "irregular_forms": irregulars
    }

def build_annotated_200_tokens():
    raw_data = [
        ("fractures", "fracture", "N", "+PL", "inflectional"),
        ("fractured", "fracture", "V", "+PAST", "inflectional"),
        ("fracturing", "fracture", "V", "+PROG", "inflectional"),
        ("fracture", "fracture", "N", "+SG", "root"),
        ("nephrectomy", "nephr", "N", "-ectomy", "derivational"),
        ("nephrectomies", "nephr", "N", "-ectomy +PL", "derivational_inflectional"),
        ("nephritis", "nephr", "N", "-itis", "derivational"),
        ("nephritic", "nephr", "ADJ", "-itic", "derivational"),
        ("nephrotic", "nephr", "ADJ", "-otic", "derivational"),
        ("nephrology", "nephr", "N", "-ology", "derivational"),
        ("nephrologist", "nephr", "N", "-ology + -ist", "derivational"),
        ("hypertension", "tens", "N", "hyper- + -ion", "derivational"),
        ("hypertensive", "tens", "ADJ", "hyper- + -ive", "derivational"),
        ("hypotension", "tens", "N", "hypo- + -ion", "derivational"),
        ("hypotensive", "tens", "ADJ", "hypo- + -ive", "derivational"),
        ("appendectomy", "append", "N", "-ectomy", "derivational"),
        ("appendectomies", "append", "N", "-ectomy +PL", "derivational_inflectional"),
        ("appendicitis", "append", "N", "-itis", "derivational"),
        ("bronchitis", "bronch", "N", "-itis", "derivational"),
        ("bronchitic", "bronch", "ADJ", "-itic", "derivational"),
        ("bronchoscopy", "bronch", "N", "-scopy", "derivational"),
        ("bronchoscopies", "bronch", "N", "-scopy +PL", "derivational_inflectional"),
        ("gastritis", "gastr", "N", "-itis", "derivational"),
        ("gastrectomy", "gastr", "N", "-ectomy", "derivational"),
        ("gastroenteritis", "gastr", "N", "gastro- + enter- + -itis", "derivational"),
        ("hepatitis", "hepat", "N", "-itis", "derivational"),
        ("hepatomegaly", "hepat", "N", "-megaly", "derivational"),
        ("hepatic", "liver", "ADJ", "suppletion", "irregular"),
        ("renal", "kidney", "ADJ", "suppletion", "irregular"),
        ("cardiac", "heart", "ADJ", "suppletion", "irregular"),
        ("pulmonary", "lung", "ADJ", "suppletion", "irregular"),
        ("gastric", "stomach", "ADJ", "suppletion", "irregular"),
        ("arthritis", "arthr", "N", "-itis", "derivational"),
        ("arthritic", "arthr", "ADJ", "-itic", "derivational"),
        ("arthroplasty", "arthr", "N", "-plasty", "derivational"),
        ("arthroplasties", "arthr", "N", "-plasty +PL", "derivational_inflectional"),
        ("arthroscopy", "arthr", "N", "-scopy", "derivational"),
        ("dermatitis", "derm", "N", "-itis", "derivational"),
        ("dermatology", "derm", "N", "-ology", "derivational"),
        ("dermatologist", "derm", "N", "-ology + -ist", "derivational"),
        ("colitis", "colon", "N", "-itis", "derivational"),
        ("colectomy", "colon", "N", "-ectomy", "derivational"),
        ("colonoscopy", "colon", "N", "-scopy", "derivational"),
        ("lesions", "lesion", "N", "+PL", "inflectional"),
        ("lesion", "lesion", "N", "+SG", "root"),
        ("patients", "patient", "N", "+PL", "inflectional"),
        ("patient", "patient", "N", "+SG", "root"),
        ("symptoms", "symptom", "N", "+PL", "inflectional"),
        ("symptomatic", "symptom", "ADJ", "-atic", "derivational"),
        ("asymptomatic", "symptom", "ADJ", "a- + -atic", "derivational"),
        ("tendons", "tendon", "N", "+PL", "inflectional"),
        ("tendonitis", "tendon", "N", "-itis", "derivational"),
        ("ligaments", "ligament", "N", "+PL", "inflectional"),
        ("vessels", "vessel", "N", "+PL", "inflectional"),
        ("arteries", "artery", "N", "+PL (y-to-i)", "inflectional"),
        ("arterial", "artery", "ADJ", "-al", "derivational"),
        ("veins", "vein", "N", "+PL", "inflectional"),
        ("venous", "vein", "ADJ", "-ous", "derivational"),
        ("muscles", "muscl", "N", "+PL", "inflectional"),
        ("muscular", "muscl", "ADJ", "-ar", "derivational"),
        ("pancreatitis", "pancrea", "N", "-itis", "derivational"),
        ("cerebral", "brain", "ADJ", "suppletion", "irregular"),
        ("cranial", "cranium", "ADJ", "-al", "derivational"),
        ("neurons", "neuron", "N", "+PL", "inflectional"),
        ("neuronal", "neuron", "ADJ", "-al", "derivational"),
        ("valves", "valv", "N", "+PL", "inflectional"),
        ("valvular", "valv", "ADJ", "-ar", "derivational"),
        ("ventricles", "ventricl", "N", "+PL", "inflectional"),
        ("ventricular", "ventricl", "ADJ", "-ar", "derivational"),
        ("atria", "atrium", "N", "+PL (Latin -um to -a)", "inflectional"),
        ("atrial", "atrium", "ADJ", "-al", "derivational"),
        ("aortic", "aort", "ADJ", "-ic", "derivational"),
        ("capillaries", "capillar", "N", "+PL", "inflectional"),
        ("lymphatic", "lymph", "ADJ", "-atic", "derivational"),
        ("splenomegaly", "spleen", "N", "-megaly", "derivational"),
        ("splenic", "spleen", "ADJ", "-ic", "derivational"),
        ("thyroiditis", "thyroid", "N", "-itis", "derivational"),
        ("hyperthyroidism", "thyroid", "N", "hyper- + -ism", "derivational"),
        ("hypothyroidism", "thyroid", "N", "hypo- + -ism", "derivational"),
        ("adrenal", "adrenal", "ADJ", "root", "root"),
        ("pituitary", "pituitar", "ADJ", "root", "root"),
        ("cortical", "cortex", "ADJ", "-ical", "derivational"),
        ("medullary", "medulla", "ADJ", "-ary", "derivational"),
        ("glomerular", "glomerul", "ADJ", "-ar", "derivational"),
        ("glomerulonephritis", "glomerul", "N", "glomerulo- + nephr- + -itis", "derivational"),
        ("tubular", "tubul", "ADJ", "-ar", "derivational"),
        ("ureteral", "ureter", "ADJ", "-al", "derivational"),
        ("urethral", "urethr", "ADJ", "-al", "derivational"),
        ("vesical", "vesic", "ADJ", "-al", "derivational"),
        ("prostatic", "prostat", "ADJ", "-ic", "derivational"),
        ("prostatectomy", "prostat", "N", "-ectomy", "derivational"),
        ("prostatitis", "prostat", "N", "-itis", "derivational"),
        ("ovarian", "ovari", "ADJ", "-an", "derivational"),
        ("uterine", "uter", "ADJ", "-ine", "derivational"),
        ("cervical", "cervix", "ADJ", "-al", "derivational"),
        ("vaginal", "vagin", "ADJ", "-al", "derivational"),
        ("placental", "placent", "ADJ", "-al", "derivational"),
        ("fetal", "fetus", "ADJ", "-al", "derivational"),
        ("osteomyelitis", "oste", "N", "osteo- + myel- + -itis", "derivational"),
        ("osteoarthritis", "oste", "N", "osteo- + arthr- + -itis", "derivational"),
        ("myocarditis", "myocard", "N", "-itis", "derivational"),
        ("endocarditis", "endocard", "N", "-itis", "derivational"),
        ("pericarditis", "pericard", "N", "-itis", "derivational"),
        ("myopathy", "myo", "N", "-pathy", "derivational"),
        ("myopathies", "myo", "N", "-pathy +PL", "derivational_inflectional"),
        ("fasciitis", "fasci", "N", "-itis", "derivational"),
        ("meningitis", "mening", "N", "-itis", "derivational"),
        ("pleurisy", "pleur", "N", "-isy", "derivational"),
        ("pleural", "pleur", "ADJ", "-al", "derivational"),
        ("peritoneal", "peritone", "ADJ", "-al", "derivational"),
        ("peritonitis", "peritone", "N", "-itis", "derivational"),
        ("erythrocytes", "erythr", "N", "erythro- + -cyte +PL", "derivational_inflectional"),
        ("erythrocytic", "erythr", "ADJ", "-ic", "derivational"),
        ("leukocytes", "leuk", "N", "leuko- + -cyte +PL", "derivational_inflectional"),
        ("leukocytosis", "leuk", "N", "-osis", "derivational"),
        ("thrombocytes", "thromb", "N", "-cyte +PL", "derivational_inflectional"),
        ("thrombocytopenia", "thromb", "N", "-penia", "derivational"),
        ("thrombosis", "thromb", "N", "-osis", "derivational"),
        ("thromboses", "thromb", "N", "-oses +PL", "inflectional"),
        ("macrophages", "macro", "N", "macro- + -phage +PL", "derivational_inflectional"),
        ("eosinophils", "eosin", "N", "-phil +PL", "derivational_inflectional"),
        ("eosinophilia", "eosin", "N", "-ia", "derivational"),
        ("basophils", "baso", "N", "-phil +PL", "derivational_inflectional"),
        ("neutrophils", "neutro", "N", "-phil +PL", "derivational_inflectional"),
        ("neutrophilia", "neutro", "N", "-ia", "derivational"),
        ("monocytes", "monocyt", "N", "+PL", "inflectional"),
        ("coagulation", "coagul", "N", "-ation", "derivational"),
        ("coagulated", "coagul", "V", "+PAST", "inflectional"),
        ("fibrinogen", "fibrin", "N", "-gen", "derivational"),
        ("albuminuria", "albumin", "N", "-uria", "derivational"),
        ("bilirubinemia", "bilirubin", "N", "-emia", "derivational"),
        ("hyperglycemia", "glucos", "N", "hyper- + -emia", "derivational"),
        ("hypoglycemia", "glucos", "N", "hypo- + -emia", "derivational"),
        ("glucosuria", "glucos", "N", "-uria", "derivational"),
        ("mitochondrial", "mitochondr", "ADJ", "-al", "derivational"),
        ("cytoplasmic", "cytoplasm", "ADJ", "-ic", "derivational"),
        ("nuclear", "nucle", "ADJ", "-ar", "derivational"),
        ("chromosomal", "chromosom", "ADJ", "-al", "derivational"),
        ("genetic", "gen", "ADJ", "-ic", "derivational"),
        ("genetics", "gen", "N", "-ics", "derivational"),
        ("mutations", "mutat", "N", "-ion +PL", "derivational_inflectional"),
        ("mutated", "mutat", "V", "+PAST", "inflectional"),
        ("peptides", "peptid", "N", "+PL", "inflectional"),
        ("proteins", "protein", "N", "+PL", "inflectional"),
        ("lipids", "lipid", "N", "+PL", "inflectional"),
        ("hyperlipidemia", "lipid", "N", "hyper- + -emia", "derivational"),
        ("esophagitis", "esophag", "N", "-itis", "derivational"),
        ("duodenitis", "duoden", "N", "-itis", "derivational"),
        ("ileitis", "ileum", "N", "-itis", "derivational"),
        ("rectal", "rectum", "ADJ", "-al", "derivational"),
        ("fistulae", "fistul", "N", "+PL (Latin -a to -ae)", "inflectional"),
        ("ulceration", "ulcer", "N", "-ation", "derivational"),
        ("ulcerative", "ulcer", "ADJ", "-ative", "derivational"),
        ("ulcers", "ulcer", "N", "+PL", "inflectional"),
        ("necrosis", "necros", "N", "-osis", "derivational"),
        ("necrotic", "necros", "ADJ", "-ic", "derivational"),
        ("gangrenous", "gangren", "ADJ", "-ous", "derivational"),
        ("ischemia", "ischemi", "N", "root", "root"),
        ("ischemic", "ischemi", "ADJ", "-ic", "derivational"),
        ("infarction", "infarct", "N", "-ion", "derivational"),
        ("infarctions", "infarct", "N", "-ion +PL", "derivational_inflectional"),
        ("infarcted", "infarct", "V", "+PAST", "inflectional"),
        ("embolism", "embol", "N", "-ism", "derivational"),
        ("emboli", "embol", "N", "+PL (Latin -us to -i)", "inflectional"),
        ("aneurysms", "aneurysm", "N", "+PL", "inflectional"),
        ("stenosis", "stenos", "N", "-osis", "derivational"),
        ("stenoses", "stenos", "N", "+PL (Greek -is to -es)", "inflectional"),
        ("occlusion", "occlus", "N", "-ion", "derivational"),
        ("occluded", "occlus", "V", "+PAST", "inflectional"),
        ("perforation", "perfor", "N", "-ation", "derivational"),
        ("perforated", "perfor", "V", "+PAST", "inflectional"),
        ("ruptured", "ruptur", "V", "+PAST", "inflectional"),
        ("rupture", "ruptur", "N", "+SG", "root"),
        ("herniation", "herni", "N", "-ation", "derivational"),
        ("hernias", "herni", "N", "+PL", "inflectional"),
        ("strictures", "strictur", "N", "+PL", "inflectional"),
        ("calculi", "calcul", "N", "+PL (Latin -us to -i)", "inflectional"),
        ("nephrolithiasis", "nephr", "N", "nephro- + -lithiasis", "derivational"),
        ("cholelithiasis", "chole", "N", "chole- + -lithiasis", "derivational"),
        ("polyps", "polyp", "N", "+PL", "inflectional"),
        ("polypectomy", "polyp", "N", "-ectomy", "derivational"),
        ("carcinomas", "carcin", "N", "-oma +PL", "derivational_inflectional"),
        ("carcinogenic", "carcin", "ADJ", "-genic", "derivational"),
        ("sarcomas", "sarcom", "N", "+PL", "inflectional"),
        ("adenomas", "adenom", "N", "+PL", "inflectional"),
        ("adenocarcinoma", "adenom", "N", "adeno- + carcin- + -oma", "derivational"),
        ("melanomas", "melanom", "N", "+PL", "inflectional"),
        ("lymphomas", "lymphom", "N", "+PL", "inflectional"),
        ("leukemias", "leukemi", "N", "+PL", "inflectional"),
        ("fibromas", "fibrom", "N", "+PL", "inflectional"),
        ("lipomas", "lipom", "N", "+PL", "inflectional"),
        ("edematous", "edem", "ADJ", "-atous", "derivational"),
        ("effusions", "effus", "N", "-ion +PL", "derivational_inflectional"),
        ("exudative", "exudat", "ADJ", "-ive", "derivational"),
        ("cyanotic", "cyanot", "ADJ", "-ic", "derivational"),
        ("cyanosis", "cyanot", "N", "-osis", "derivational"),
        ("feverish", "fever", "ADJ", "-ish", "derivational"),
        ("fevers", "fever", "N", "+PL", "inflectional"),
        ("pyrexia", "pyrex", "N", "-ia", "derivational"),
        ("diaphoretic", "diaphor", "ADJ", "-etic", "derivational"),
        ("nauseous", "nause", "ADJ", "-ous", "derivational"),
        ("vomited", "vomit", "V", "+PAST", "inflectional"),
        ("vomiting", "vomit", "V", "+PROG", "inflectional"),
        ("dizziness", "dizzin", "N", "-ness", "derivational"),
        ("headaches", "headach", "N", "+PL", "inflectional"),
        ("seizures", "seizur", "N", "+PL", "inflectional"),
        ("dysarthric", "dysarthr", "ADJ", "-ic", "derivational"),
        ("dysphagic", "dysphag", "ADJ", "-ic", "derivational"),
        ("dyspneic", "dyspne", "ADJ", "-ic", "derivational"),
        ("tachycardic", "tachycard", "ADJ", "-ic", "derivational"),
        ("bradycardic", "bradycard", "ADJ", "-ic", "derivational"),
        ("arrhythmias", "arrhythm", "N", "+PL", "inflectional"),
        ("arrhythmic", "arrhythm", "ADJ", "-ic", "derivational"),
        ("septicemic", "septic", "ADJ", "-emic", "derivational")
    ]
    tokens = []
    for item in raw_data:
        tokens.append({
            "surface_form": item[0],
            "lemma": item[1],
            "pos": item[2],
            "morph_feature": item[3],
            "category": item[4]
        })
    return tokens

def build_40_eval_queries(records):
    queries = [
        "fracture", "fractures", "fractured", "fracturing",
        "nephrectomy", "nephrectomies", "nephritis", "renal",
        "appendicitis", "appendectomy", "hypertension", "hypertensive",
        "pneumonia", "asthma", "diabetes", "diabetic",
        "cholecystitis", "cholecystectomy", "infarction", "ischemia",
        "stroke", "dermatitis", "osteoarthritis", "arthroplasty",
        "paracetamol", "amoxicillin", "metformin", "atorvastatin",
        "ceftriaxone", "tramadol", "furosemide", "insulin",
        "creatinine", "hemoglobin", "blood pressure", "temperature",
        "dyspnoea", "headache", "edema", "vomiting"
    ]
    query_eval = []
    for q in queries:
        q_lower = q.lower()
        matched_ids = []
        for r in records:
            if re.search(r"\b" + re.escape(q_lower), r["text"], re.IGNORECASE):
                matched_ids.append(r["record_id"])
        query_eval.append({
            "query": q,
            "relevant_record_ids": matched_ids,
            "total_relevant": len(matched_ids)
        })
    return query_eval

def build_noisy_day4_records():
    return [
        {
            "record_id": "NOISY-OCR-001",
            "type": "OCR Scanned Form",
            "text": (
                "DISCHARGE SUlvlMARY\n"
                "Patient lD: PlD-10294 | Age: 45 yo | Contact: +91-9876543210\n"
                "Admission Date: 2O24-O3-12 | Discharge Date: 2024-03-18\n"
                "Location: CardiaIogy Ward, Bed 12\n"
                "Primary Diagnosis: Acute myocardiaI infarction (ICD code: l21.9).\n"
                "Physical findings: bIood pressure 14O/9O mmHg, body temp 98.6 F.\n"
                "Lab tests: Hb 13.5 g/dL, Creatinine 1.2 mg/dL, WBC 85OO /uL.\n"
                "Post-oper- \native notes indicate patient underwent coronary catheterisa- \ntion successfully.\n"
                "Prescribed Paracetam0l 5OOmg t.i.d. and Atorvastatin 2Omg once daily."
            )
        },
        {
            "record_id": "NOISY-MIXED-002",
            "type": "Code-Mixed Romanised Regional",
            "text": (
                "NURSING OBSERVATION & SHIFT HANDOVER\n"
                "Patient ID: PID-10552 | Age: 52 yo | Contact: +91-9840123456\n"
                "Location: Orthopaedic Unit, Bed 08\n"
                "Primary Issue: Closed fracture right leg.\n"
                "Shift Notes: patient ku fever irundhuchu, evening time romba shivering aachu. "
                "BP 135/85 mmHg check panrom, temp 101.2 F irukku. "
                "paracetamol 500mg kuduthen, after one hour fever koranjuduchu. "
                "leg pain kooda irundhuchu, tramadol 50mg potrukkom. "
                "wound dressing maathiyachu, bleeding edhum illa. doctor kitta inform panirukkom."
            )
        },
        {
            "record_id": "NOISY-HYBRID-003",
            "type": "Hybrid OCR and Code-Mixed",
            "text": (
                "EMERGENCY ENCOUNTER N0TE\n"
                "Patient lD: PlD-10884 | Age: 63 years oId | Ward 4, Bed 15\n"
                "ClinicaI Observation: patient ku chest pain irundhuchu for 3 hours. "
                "ECG Ia st elevation therinjidhu. BIood pressure 15O/95 mmHg, temp 99.O F. "
                "Hb 11.2 g/dL, WBC 112OO /uL, Creatinine 1.4 mg/dL. "
                "aspirin 15Omg stat kuduthen, clopidogrel 3OOmg potachu. "
                "patient ippo stable ah irukaaru, ICU-B ku shift panna pOrOm."
            )
        }
    ]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Generating 500 clinical records...")
    records, tokens = generate_clinical_corpus(500)
    print(f"Generated {len(records)} records with {tokens} words (target: >= 25,000).")
    
    corpus_path = os.path.join(DATA_DIR, "clinical_corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved corpus to {corpus_path}")

    txt_corpus_path = os.path.join(DATA_DIR, "clinical_corpus.txt")
    with open(txt_corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r["text"] + "\n" + "=" * 80 + "\n")
    print(f"Saved plain text corpus to {txt_corpus_path}")

    print("Generating morphological lexicon...")
    lexicon = build_morphological_lexicon()
    lex_path = os.path.join(DATA_DIR, "morphological_lexicon.json")
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2)
    print(f"Saved lexicon with {len(lexicon['stems'])} stems, {len(lexicon['inflectional_affixes'])} inflectional, {len(lexicon['derivational_affixes'])} derivational affixes to {lex_path}")

    print("Generating 200 annotated tokens...")
    annotated = build_annotated_200_tokens()
    ann_path = os.path.join(DATA_DIR, "annotated_200_tokens.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2)
    print(f"Saved {len(annotated)} annotated tokens to {ann_path}")

    print("Generating 40 clinician evaluation queries...")
    eval_queries = build_40_eval_queries(records)
    eval_path = os.path.join(DATA_DIR, "eval_40_queries.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, indent=2)
    print(f"Saved {len(eval_queries)} queries to {eval_path}")

    print("Generating Day-4 noisy challenge records...")
    noisy_recs = build_noisy_day4_records()
    noisy_path = os.path.join(DATA_DIR, "noisy_day4_records.json")
    with open(noisy_path, "w", encoding="utf-8") as f:
        json.dump(noisy_recs, f, indent=2)
    print(f"Saved {len(noisy_recs)} noisy records to {noisy_path}")

if __name__ == "__main__":
    main()
