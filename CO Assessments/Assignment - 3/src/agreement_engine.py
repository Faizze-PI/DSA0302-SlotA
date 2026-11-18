"""Morphosyntactic agreement validation for Subject-Verb, Determiner-Noun, and Pronoun Case."""

from typing import Dict, List, Optional, Tuple
from .cfg_grammar import HelpdeskCFG, LexicalEntry


class AgreementResult:
    def __init__(self, is_valid: bool, violation_type: str = "None", details: str = "Agreement constraints satisfied"):
        self.is_valid = is_valid
        self.violation_type = violation_type
        self.details = details

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"[{status}] {self.violation_type}: {self.details}"


class AgreementEngine:
    def __init__(self, grammar: HelpdeskCFG):
        self.grammar = grammar

    def validate_det_noun(self, det_token: str, noun_token: str) -> AgreementResult:
        det_entries = self.grammar.lexicon.get(det_token.lower(), [])
        noun_entries = self.grammar.lexicon.get(noun_token.lower(), [])

        if not det_entries or not noun_entries:
            return AgreementResult(False, "Lexical_Lookup_Failure", f"Tokens not in lexicon: {det_token}, {noun_token}")

        det = det_entries[0]
        noun = noun_entries[0]

        if det.number != "any" and noun.number != "any":
            if det.number != noun.number:
                return AgreementResult(
                    False,
                    "Determiner_Noun_Number_Clash",
                    f"Determiner '{det_token}' ({det.number}) clashes with Noun '{noun_token}' ({noun.number})"
                )
        return AgreementResult(True)

    def validate_subject_verb(self, subj_token: str, verb_token: str) -> AgreementResult:
        subj_entries = self.grammar.lexicon.get(subj_token.lower(), [])
        verb_entries = self.grammar.lexicon.get(verb_token.lower(), [])

        if not subj_entries or not verb_entries:
            return AgreementResult(False, "Lexical_Lookup_Failure", f"Tokens not in lexicon: {subj_token}, {verb_token}")

        subj = subj_entries[0]
        verb = verb_entries[0]

        if subj.number != "any" and verb.number != "any":
            if subj.number != verb.number:
                return AgreementResult(
                    False,
                    "Subject_Verb_Number_Clash",
                    f"Subject '{subj_token}' ({subj.number}) clashes with Verb '{verb_token}' ({verb.number})"
                )

        if subj.person != "any" and verb.person != "any":
            if subj.person != verb.person:
                return AgreementResult(
                    False,
                    "Subject_Verb_Person_Clash",
                    f"Subject '{subj_token}' (person {subj.person}) clashes with Verb '{verb_token}' (person {verb.person})"
                )

        return AgreementResult(True)

    def validate_pronoun_case(self, pronoun_token: str, position: str) -> AgreementResult:
        pron_entries = self.grammar.lexicon.get(pronoun_token.lower(), [])
        if not pron_entries:
            return AgreementResult(False, "Lexical_Lookup_Failure", f"Pronoun not in lexicon: {pronoun_token}")

        pron = pron_entries[0]
        if pron.pos != "Pron":
            return AgreementResult(True)

        expected_case = "nom" if position == "subject" else "acc"
        if pron.case != "any" and pron.case != expected_case:
            return AgreementResult(
                False,
                "Pronoun_Case_Violation",
                f"Pronoun '{pronoun_token}' has case '{pron.case}' but position '{position}' requires '{expected_case}'"
            )
        return AgreementResult(True)

    def evaluate_test_suite(self) -> Dict[str, List[Tuple[str, AgreementResult]]]:
        grammatical = [
            ("the student submits the examination form", "student", "submits", "the", "student", "the", "form", "none", "subject"),
            ("the students submit the examination form", "students", "submit", "the", "students", "the", "form", "none", "subject"),
            ("this form is pending", "form", "is", "this", "form", "", "", "none", "subject"),
            ("these forms are pending", "forms", "are", "these", "forms", "", "", "none", "subject"),
            ("he paid the fee", "he", "paid", "", "", "the", "fee", "he", "subject"),
            ("the warden sent him the receipt", "warden", "sent", "the", "warden", "the", "receipt", "him", "object")
        ]

        ungrammatical = [
            ("the student submit the examination form", "student", "submit", "the", "student", "the", "form", "none", "subject"),
            ("the students submits the examination form", "students", "submits", "the", "students", "the", "form", "none", "subject"),
            ("this forms is pending", "forms", "is", "this", "forms", "", "", "none", "subject"),
            ("these form are pending", "form", "are", "these", "form", "", "", "none", "subject"),
            ("him paid the fee", "him", "paid", "", "", "the", "fee", "him", "subject"),
            ("the warden sent he the receipt", "warden", "sent", "the", "warden", "the", "receipt", "he", "object")
        ]

        results = {"Grammatical": [], "Ungrammatical": []}

        for sent, s_tk, v_tk, d_tk, n_tk, d2_tk, n2_tk, p_tk, p_pos in grammatical:
            res_sv = self.validate_subject_verb(s_tk, v_tk)
            res_dn = self.validate_det_noun(d_tk, n_tk) if d_tk and n_tk else AgreementResult(True)
            res_pc = self.validate_pronoun_case(p_tk, p_pos) if p_tk != "none" else AgreementResult(True)

            final_ok = res_sv.is_valid and res_dn.is_valid and res_pc.is_valid
            final_res = AgreementResult(True, "None", "All morphological agreement constraints satisfied") if final_ok else AgreementResult(False, "Unexpected_Fail", "Failed valid")
            results["Grammatical"].append((sent, final_res))

        for sent, s_tk, v_tk, d_tk, n_tk, d2_tk, n2_tk, p_tk, p_pos in ungrammatical:
            res_sv = self.validate_subject_verb(s_tk, v_tk)
            res_dn = self.validate_det_noun(d_tk, n_tk) if d_tk and n_tk else AgreementResult(True)
            res_pc = self.validate_pronoun_case(p_tk, p_pos) if p_tk != "none" else AgreementResult(True)

            if not res_sv.is_valid:
                results["Ungrammatical"].append((sent, res_sv))
            elif not res_dn.is_valid:
                results["Ungrammatical"].append((sent, res_dn))
            elif not res_pc.is_valid:
                results["Ungrammatical"].append((sent, res_pc))
            else:
                results["Ungrammatical"].append((sent, AgreementResult(False, "Undetected_Error", "Rule failed to reject illicit input")))

        return results
