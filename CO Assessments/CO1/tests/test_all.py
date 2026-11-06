"""
Module: test_all.py
Automated Verification Suite
DSA03 - Natural Language Processing (Assignment 1 - CO1)
Author: Clinical NLP Language Engineering Team
Unit and integration tests for all 12 tasks covering:
Regex patterns, Tokenization, Automata, FST Parsing, Porter Stemming, and Search.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_patterns import (
    COMPILED_PATTERNS,
    NEGATIVE_TEST_CASES,
    demonstrate_greedy_vs_nongreedy,
    demonstrate_precedence_issue,
    extract_entities,
    validate_entity
)
from src.tokenizer_normalizer import (
    compare_with_whitespace_split,
    expand_abbreviations,
    normalise_text,
    tokenize
)
from src.automata_engine import (
    build_blood_pressure_dfa,
    build_icd_code_dfa,
    build_optional_dosage_enfa,
    build_patient_id_dfa,
    build_temperature_dfa,
    minimize_dfa,
    subset_construction
)
from src.morphology_engine import CLINICAL_PARADIGMS, DERIVATIONAL_FAMILIES
from src.transducer_fst import MorphologicalFST
from src.porter_stemmer import PorterStemmer
from src.day4_challenge import NoisyTextEvaluator

class TestClinicalNLP(unittest.TestCase):
    def setUp(self):
        self.fst = MorphologicalFST()
        self.stemmer = PorterStemmer()

    def test_task2_entity_patterns(self):
        sample_text = (
            "Patient PID-10294, age 45 yo, admitted on 2024-03-12. "
            "Prescribed Paracetamol 500mg t.i.d. and Amoxicillin 250mg b.i.d. "
            "Vitals: BP 120/80 mmHg, temp 98.6 F. Lab: Hb 13.5 g/dL. "
            "Diagnosis: K35.8 in General Surgery Ward, Bed 12. "
            "Contact: +91-9876543210 for 3 days."
        )
        extracted = extract_entities(sample_text)
        self.assertTrue(len(extracted["patient_id"]) >= 1)
        self.assertTrue(len(extracted["clinical_date"]) >= 1)
        self.assertTrue(len(extracted["drug_strength"]) >= 2)
        self.assertTrue(len(extracted["blood_pressure"]) >= 1)
        self.assertTrue(len(extracted["temperature"]) >= 1)
        self.assertTrue(len(extracted["lab_value"]) >= 1)
        self.assertTrue(len(extracted["icd_code"]) >= 1)
        self.assertTrue(len(extracted["ward_bed"]) >= 1)
        self.assertTrue(len(extracted["contact_number"]) >= 1)
        self.assertTrue(len(extracted["symptom_duration"]) >= 1)

    def test_task2_negative_cases(self):
        for ent, neg_cases in NEGATIVE_TEST_CASES.items():
            for case in neg_cases:
                self.assertFalse(
                    validate_entity(ent, case),
                    f"Entity {ent} should reject '{case}'"
                )

    def test_task2_greedy_semantics(self):
        res = demonstrate_greedy_vs_nongreedy()
        self.assertIn("Amoxicillin", res["greedy_span"])
        self.assertNotIn("Amoxicillin", res["nongreedy_span"])

    def test_task3_tokenizer(self):
        text = "Post-operative Hb 13.5 g/dL was checked for PID-10024 b.i.d."
        tokens = tokenize(text)
        self.assertIn("Post-operative", tokens)
        self.assertIn("13.5", tokens)
        self.assertIn("PID-10024", tokens)
        self.assertIn("b.i.d.", tokens)

    def test_task3_abbreviation_expansion(self):
        text = "Administer Paracetamol 500mg po bid prn"
        expanded = expand_abbreviations(text)
        self.assertIn("by mouth", expanded)
        self.assertIn("twice daily", expanded)
        self.assertIn("as needed", expanded)

    def test_task4_automata_validators(self):
        pid_dfa = build_patient_id_dfa()
        self.assertTrue(pid_dfa.trace("PID-10294")[0])
        self.assertFalse(pid_dfa.trace("PID-99")[0])
        self.assertFalse(pid_dfa.trace("ID-10294")[0])

        bp_dfa = build_blood_pressure_dfa()
        self.assertTrue(bp_dfa.trace("120/80 mmHg")[0])
        self.assertFalse(bp_dfa.trace("120/80")[0])

        temp_dfa = build_temperature_dfa()
        self.assertTrue(temp_dfa.trace("98.6 F")[0])
        self.assertTrue(temp_dfa.trace("37 C")[0])
        self.assertFalse(temp_dfa.trace("98.6")[0])

        icd_dfa = build_icd_code_dfa()
        self.assertTrue(icd_dfa.trace("I10")[0])
        self.assertTrue(icd_dfa.trace("K35.8")[0])
        self.assertFalse(icd_dfa.trace("123.4")[0])

    def test_task4_enfa_and_subset_construction(self):
        enfa = build_optional_dosage_enfa()
        self.assertTrue(enfa.trace("500mg b.i")[0])
        self.assertTrue(enfa.trace("500mg")[0])
        self.assertFalse(enfa.trace("500m")[0])

        dfa, _ = subset_construction(enfa)
        self.assertTrue(dfa.trace("500mg b.i")[0])
        self.assertTrue(dfa.trace("500mg")[0])
        self.assertFalse(dfa.trace("500m")[0])

        min_dfa, before, after = minimize_dfa(dfa)
        self.assertLessEqual(after, before)
        self.assertTrue(min_dfa.trace("500mg b.i")[0])
        self.assertTrue(min_dfa.trace("500mg")[0])

    def test_task7_fst_bidirectional(self):
        parsed = self.fst.parse("fractures")
        self.assertTrue(any(p["stem"] == "fracture" and "+PL" in p["feature"] for p in parsed))

        generated = self.fst.generate("fracture", "+N +PL")
        self.assertIn("fractures", generated)

    def test_task8_twenty_orthographic_forms(self):
        results = self.fst.test_twenty_orthographic_forms()
        for r in results:
            self.assertEqual(r["status"], "PASS", f"Failed rule for {r['intermediate']}")

    def test_task9_porter_stemmer(self):
        self.assertEqual(self.stemmer.stem("fractures"), "fractur")
        self.assertEqual(self.stemmer.stem("treatment"), "treatment")
        self.assertEqual(self.stemmer.stem("operating"), "oper")
        self.assertEqual(self.stemmer.stem("injected"), "inject")
        self.assertEqual(self.stemmer.stem("scanned"), "scan")

    def test_task12_day4_noise_recovery(self):
        evaluator = NoisyTextEvaluator()
        ocr_text = "Patient lD: PlD-10294, bIood pressure 14O/9O mmHg, temp 98.6 F"
        repaired = evaluator.repair_ocr_substitutions(ocr_text)
        self.assertIn("PID-10294", repaired)
        self.assertIn("blood", repaired)
        self.assertIn("140/90", repaired)

if __name__ == "__main__":
    unittest.main()
