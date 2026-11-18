"""Verb subcategorization frame validation, complement vs adjunct diagnostics, and valence enforcement."""

from typing import Dict, List, Optional, Tuple
from .cfg_grammar import HelpdeskCFG


class SubcatValidationResult:
    def __init__(self, is_valid: bool, frame_expected: str, frame_observed: str, message: str):
        self.is_valid = is_valid
        self.frame_expected = frame_expected
        self.frame_observed = frame_observed
        self.message = message

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"[{status}] Expected {self.frame_expected}, observed {self.frame_observed}: {self.message}"


class SubcatEngine:
    def __init__(self, grammar: HelpdeskCFG):
        self.grammar = grammar
        self.frames = {
            "V_intr": {"complements": [], "description": "Intransitive (0 objects)"},
            "V_trans": {"complements": ["NP"], "description": "Transitive (1 direct NP object)"},
            "V_ditrans": {"complements": ["NP", "NP"], "description": "Ditransitive (indirect + direct NP)"},
            "V_comp": {"complements": ["Sbar"], "description": "Sentential complement clause"},
            "V_prep": {"complements": ["PP"], "description": "Prepositional complement phrase"}
        }

    def validate_verb_valence(self, verb_token: str, complement_types: List[str]) -> SubcatValidationResult:
        entries = self.grammar.lexicon.get(verb_token.lower(), [])
        if not entries:
            return SubcatValidationResult(False, "Unknown", str(complement_types), f"Verb '{verb_token}' not in lexicon")

        verb_entry = entries[0]
        expected_frame = verb_entry.subcat_frame
        if expected_frame == "none" or expected_frame not in self.frames:
            expected_frame = verb_entry.pos

        req_complements = self.frames.get(expected_frame, {}).get("complements", [])

        if expected_frame == "V_intr":
            if len(complement_types) > 0 and complement_types[0] == "NP":
                return SubcatValidationResult(
                    False, expected_frame, str(complement_types),
                    f"Over-complementation: Intransitive verb '{verb_token}' cannot take direct object NP"
                )
            return SubcatValidationResult(True, expected_frame, str(complement_types), "Valid intransitive frame")

        elif expected_frame == "V_trans":
            if not complement_types or complement_types[0] != "NP":
                return SubcatValidationResult(
                    False, expected_frame, str(complement_types),
                    f"Under-complementation: Transitive verb '{verb_token}' requires an obligatory direct object NP"
                )
            return SubcatValidationResult(True, expected_frame, str(complement_types), "Valid transitive frame")

        elif expected_frame == "V_ditrans":
            if len(complement_types) < 2 or complement_types[:2] != ["NP", "NP"]:
                return SubcatValidationResult(
                    False, expected_frame, str(complement_types),
                    f"Valence error: Ditransitive verb '{verb_token}' requires two NP complements (recipient, theme)"
                )
            return SubcatValidationResult(True, expected_frame, str(complement_types), "Valid ditransitive frame")

        elif expected_frame == "V_comp":
            if not any(c in ("Sbar", "S") for c in complement_types):
                return SubcatValidationResult(
                    False, expected_frame, str(complement_types),
                    f"Valence error: Sentential complement verb '{verb_token}' requires an Sbar clausal complement"
                )
            return SubcatValidationResult(True, expected_frame, str(complement_types), "Valid sentential complement frame")

        elif expected_frame == "V_prep":
            if not any(c == "PP" for c in complement_types):
                return SubcatValidationResult(
                    False, expected_frame, str(complement_types),
                    f"Valence error: Prepositional verb '{verb_token}' requires a PP complement"
                )
            return SubcatValidationResult(True, expected_frame, str(complement_types), "Valid prepositional complement frame")

        return SubcatValidationResult(True, expected_frame, str(complement_types), "Frame accepted")

    def run_subcat_tests(self) -> List[Tuple[str, str, List[str], SubcatValidationResult]]:
        test_cases = [
            ("The portal crashed", "crashed", [], True),
            ("*The portal crashed the form", "crashed", ["NP"], False),
            ("Submit the examination form", "submit", ["NP"], True),
            ("*Submit", "submit", [], False),
            ("Send me the receipt", "send", ["NP", "NP"], True),
            ("*Send me", "send", ["NP"], False),
            ("Tell me that the fee is paid", "tell", ["NP", "Sbar"], True),
            ("*Tell me", "tell", ["NP"], False),
            ("Students apply for a hostel room", "apply", ["PP"], True),
            ("*Students apply a hostel room", "apply", ["NP"], False)
        ]

        results = []
        for text, v_tk, complements, expected_pass in test_cases:
            res = self.validate_verb_valence(v_tk, complements)
            results.append((text, v_tk, complements, res))
        return results
