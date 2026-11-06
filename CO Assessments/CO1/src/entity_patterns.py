"""
Module: entity_patterns.py
Clinical Entity Extraction and Regular Expression Engine
DSA03 - Natural Language Processing (Assignment 1 - Task 2)
Author: Clinical NLP Language Engineering Team
Defines and evaluates regular expressions for twelve structured clinical entities.
Demonstrates greedy vs. non-greedy quantifier semantics and operator precedence.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

ENTITY_REGEXES: Dict[str, str] = {
    "patient_id": r"\b(?:PID|MRN|P)-\d{4,6}\b",
    "clinical_date": r"\b(?:\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])|(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{4}|(?:0[1-9]|[12]\d|3[01])-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4})\b",
    "drug_strength": r"\b([A-Z][a-z]+(?:-[A-Za-z]+)?)\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|mL|units))\b",
    "dosage_frequency": r"\b(?:b\.i\.d\.|t\.i\.d\.|q\.d\.s\.|once\s+daily(?:\s+morning|\s+before\s+breakfast)?|every\s+\d+\s+hours(?:\s+prn)?|at\s+bedtime|stat|prn)\b",
    "lab_value": r"\b(Hb|WBC|Creatinine|Platelets|Potassium|Sodium|Glucose|Bilirubin|AST|ALT)\s+(\d+(?:\.\d+)?)\s*(mg/dL|g/dL|mmol/L|/uL|U/L|mEq/L|%)\b",
    "blood_pressure": r"\b([89]\d|1\d{2}|2[0-3]\d)/([4-9]\d|1[0-3]\d)\s*mmHg\b",
    "temperature": r"\b(9[5-9](?:\.\d)?|10[0-6](?:\.\d)?|3[5-9](?:\.\d)?|4[0-2](?:\.\d)?)\s*([FC])\b",
    "icd_code": r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b",
    "ward_bed": r"\b((?:[A-Za-z\s]+Ward|ICU-[AB]|Operating\s+Suite\s+\d+|Emergency\s+Care|Orthopaedic\s+Unit|Pulmonology\s+Bay|General\s+Surgery)),\s*(Bed\s+\d+[A-Za-z]?)\b",
    "contact_number": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
    "age": r"\b(\d{1,2}|1[01]\d)\s*(?:yo|years\s+old|-year-old)\b",
    "symptom_duration": r"\b(?:for|duration|since)\s+(\d{1,2}\s+(?:hours|days|weeks|months|years))\b"
}

COMPILED_PATTERNS = {
    k: re.compile(v, re.IGNORECASE if k not in ["icd_code", "drug_strength"] else 0)
    for k, v in ENTITY_REGEXES.items()
}

NEGATIVE_TEST_CASES = {
    "patient_id": ["ID-12", "PID_9999", "PATIENT-88", "12345"],
    "clinical_date": ["2024-13-45", "99/99/9999", "20240312", "yesterday"],
    "drug_strength": ["Paracetamol", "500mg", "Amoxicillin pills", "tablets 20"],
    "dosage_frequency": ["sometimes", "rarely", "twice", "continuous"],
    "lab_value": ["Hb normal", "Creatinine high", "13.5 g/dL", "WBC"],
    "blood_pressure": ["250/150", "120/80", "120/80 mmhg bad", "bp 120"],
    "temperature": ["108 F", "25 C", "temp 98", "hot"],
    "icd_code": ["U99.9", "123.4", "ICD-I10", "A1"],
    "ward_bed": ["Ward Bed", "Room 4", "Bed 12", "ICU"],
    "contact_number": ["1234567890", "+91-1234567890", "98765", "0000000000"],
    "age": ["150 yo", "age young", "born 1980", "200 years old"],
    "symptom_duration": ["since yesterday", "for long time", "duration prolonged", "hours 5"]
}

def extract_entities(text: str) -> Dict[str, List[Any]]:
    extracted: Dict[str, List[Any]] = {}
    for name, pattern in COMPILED_PATTERNS.items():
        matches = pattern.findall(text)
        extracted[name] = matches
    return extracted

def validate_entity(entity_type: str, candidate: str) -> bool:
    if entity_type not in COMPILED_PATTERNS:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return bool(COMPILED_PATTERNS[entity_type].fullmatch(candidate.strip()))

def demonstrate_greedy_vs_nongreedy(sentence: Optional[str] = None) -> Dict[str, Any]:
    if sentence is None:
        sentence = "Administer Paracetamol 500mg oral at 08:00 and Amoxicillin 250mg oral at 12:00."
    
    greedy_pattern = r"Paracetamol.*oral"
    nongreedy_pattern = r"Paracetamol.*?oral"

    greedy_match = re.search(greedy_pattern, sentence)
    nongreedy_match = re.search(nongreedy_pattern, sentence)

    return {
        "sentence": sentence,
        "greedy_regex": greedy_pattern,
        "greedy_span": greedy_match.group(0) if greedy_match else None,
        "nongreedy_regex": nongreedy_pattern,
        "nongreedy_span": nongreedy_match.group(0) if nongreedy_match else None,
        "explanation": "Greene quantifier .* consumes maximal characters across the second drug dosage before matching 'oral', whereas non-greedy .*? stops at the first occurrence of 'oral'."
    }

def demonstrate_precedence_issue() -> Dict[str, Any]:
    misplaced = r"Paracetamol|Amoxicillin\s+500mg"
    intended = r"(?:Paracetamol|Amoxicillin)\s+500mg"
    
    test_1 = "Paracetamol 500mg"
    test_2 = "Paracetamol"
    test_3 = "Amoxicillin"

    return {
        "misplaced_regex": misplaced,
        "intended_regex": intended,
        "test_cases": {
            test_1: {
                "misplaced_matches": bool(re.fullmatch(misplaced, test_1)),
                "intended_matches": bool(re.fullmatch(intended, test_1))
            },
            test_2: {
                "misplaced_matches": bool(re.fullmatch(misplaced, test_2)),
                "intended_matches": bool(re.fullmatch(intended, test_2))
            },
            test_3: {
                "misplaced_matches": bool(re.fullmatch(misplaced, test_3)),
                "intended_matches": bool(re.fullmatch(intended, test_3))
            }
        },
        "linguistic_cause": "Disjunction has lower precedence than concatenation. 'Paracetamol|Amoxicillin 500mg' parses as 'Paracetamol' OR ('Amoxicillin 500mg'), silently matching isolated 'Paracetamol' without dosage."
    }

def evaluate_on_corpus(records_path: str) -> Dict[str, Any]:
    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    evaluation = {}
    for entity in COMPILED_PATTERNS.keys():
        evaluation[entity] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for record in records:
        text = record["text"]
        extracted = extract_entities(text)

        if record.get("patient_id"):
            found = any(record["patient_id"] in str(m) for m in extracted["patient_id"])
            if found:
                evaluation["patient_id"]["tp"] += 1
            else:
                evaluation["patient_id"]["fn"] += 1

        if record.get("blood_pressure"):
            found = any(record["blood_pressure"].split()[0] in str(m) for m in extracted["blood_pressure"])
            if found:
                evaluation["blood_pressure"]["tp"] += 1
            else:
                evaluation["blood_pressure"]["fn"] += 1

        if record.get("temperature"):
            found = any(record["temperature"].split()[0] in str(m) for m in extracted["temperature"])
            if found:
                evaluation["temperature"]["tp"] += 1
            else:
                evaluation["temperature"]["fn"] += 1

        if record.get("icd_code"):
            found = any(record["icd_code"] in str(m) for m in extracted["icd_code"])
            if found:
                evaluation["icd_code"]["tp"] += 1
            else:
                evaluation["icd_code"]["fn"] += 1

        for drug in record.get("drugs", []):
            dname = drug.split()[0]
            found = any(dname in str(m) for m in extracted["drug_strength"])
            if found:
                evaluation["drug_strength"]["tp"] += 1
            else:
                evaluation["drug_strength"]["fn"] += 1

        for freq in record.get("frequencies", []):
            found = any(freq in str(m) for m in extracted["dosage_frequency"])
            if found:
                evaluation["dosage_frequency"]["tp"] += 1
            else:
                evaluation["dosage_frequency"]["fn"] += 1

        if record.get("age"):
            raw_age = re.search(r"\d+", record["age"]).group(0)
            found = any(raw_age in str(m) for m in extracted["age"])
            if found:
                evaluation["age"]["tp"] += 1
            else:
                evaluation["age"]["fn"] += 1

        if record.get("contact"):
            raw_contact = record["contact"][-10:]
            found = any(raw_contact in str(m) for m in extracted["contact_number"])
            if found:
                evaluation["contact_number"]["tp"] += 1
            else:
                evaluation["contact_number"]["fn"] += 1

        if record.get("admission_date"):
            found = any(record["admission_date"] in str(m) for m in extracted["clinical_date"])
            if found:
                evaluation["clinical_date"]["tp"] += 1
            else:
                evaluation["clinical_date"]["fn"] += 1

        for lab in record.get("lab_values", []):
            lname = lab.split()[0]
            found = any(lname in str(m) for m in extracted["lab_value"])
            if found:
                evaluation["lab_value"]["tp"] += 1
            else:
                evaluation["lab_value"]["fn"] += 1

        if record.get("duration"):
            found = any(record["duration"].split()[-1] in str(m) for m in extracted["symptom_duration"])
            if found:
                evaluation["symptom_duration"]["tp"] += 1
            else:
                evaluation["symptom_duration"]["fn"] += 1

        if record.get("ward_bed"):
            found = any("Bed" in str(m) for m in extracted["ward_bed"])
            if found:
                evaluation["ward_bed"]["tp"] += 1
            else:
                evaluation["ward_bed"]["fn"] += 1

    for k, v in evaluation.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        v["precision"] = round(prec, 4)
        v["recall"] = round(rec, 4)
        v["f1"] = round(f1, 4)

    return evaluation
