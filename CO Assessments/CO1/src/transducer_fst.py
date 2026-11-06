"""
Module: transducer_fst.py
Finite-State Transducer (FST) and Two-Level Morphology Cascade
DSA03 - Natural Language Processing (Assignment 1 - Tasks 7 & 8)
Author: Clinical NLP Language Engineering Team
Implements two-tape FST with continuation classes for bidirectional morphological
parsing and generation. Composes sequential orthographic rule transducers.
"""


import re
from typing import Any, Dict, List, Optional, Set, Tuple

class MorphologicalFST:
    def __init__(self):
        self.lexicon: Dict[str, Dict[str, Any]] = {
            "fracture": {"pos": ["N", "V"], "cat": "REG_E_STEM"},
            "inject": {"pos": ["V"], "cat": "REG_C_STEM"},
            "scan": {"pos": ["V", "N"], "cat": "DOUBLE_C_STEM"},
            "biopsy": {"pos": ["N", "V"], "cat": "Y_STEM"},
            "abscess": {"pos": ["N"], "cat": "SIBILANT_STEM"},
            "nephr": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "append": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "bronch": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "gastr": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "derm": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "arthr": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "col": {"pos": ["N"], "cat": "ROOT_DERIV"},
            "tens": {"pos": ["N", "ADJ"], "cat": "ROOT_DERIV"},
            "dilute": {"pos": ["V", "ADJ"], "cat": "REG_E_STEM"},
            "severe": {"pos": ["ADJ"], "cat": "REG_E_STEM"},
            "patient": {"pos": ["N", "ADJ"], "cat": "REG_C_STEM"},
            "ulcer": {"pos": ["N"], "cat": "REG_C_STEM"},
            "lesion": {"pos": ["N"], "cat": "REG_C_STEM"},
            "tendon": {"pos": ["N"], "cat": "REG_C_STEM"},
            "artery": {"pos": ["N"], "cat": "Y_STEM"}
        }

        self.inflectional_rules: Dict[str, List[Tuple[str, str]]] = {
            "N": [
                ("", "+N +SG"),
                ("+s", "+N +PL"),
                ("+es", "+N +PL")
            ],
            "V": [
                ("", "+V +BASE"),
                ("+s", "+V +3SG +PRES"),
                ("+es", "+V +3SG +PRES"),
                ("+ed", "+V +PAST"),
                ("+ing", "+V +PROG")
            ],
            "ADJ": [
                ("", "+ADJ +BASE"),
                ("+er", "+ADJ +COMP"),
                ("+est", "+ADJ +SUPERL")
            ]
        }

        self.derivational_rules: Dict[str, List[Tuple[str, str, str]]] = {
            "ROOT_DERIV": [
                ("+itis", "N", "-itis (inflammation)"),
                ("+ectomy", "N", "-ectomy (surgical excision)"),
                ("+ology", "N", "-ology (study of)"),
                ("+oscopy", "N", "-scopy (visual inspection)"),
                ("+oplasty", "N", "-plasty (surgical repair)"),
                ("+opathy", "N", "-pathy (disease of)")
            ]
        }

    def apply_orthographic_rules(self, intermediate_form: str) -> str:
        s = intermediate_form
        had_e_deletion = bool(re.search(r"e\+[aeiou]", s))
        if had_e_deletion:
            s = re.sub(r"e\+([aeiou])", r"+\1", s)

        if not had_e_deletion:
            s = re.sub(r"\b([bcdfghjklmnpqrstvwxyz]*[aeiou])([bdfgmnpt])\+([aeiou])", r"\1\2\2+\3", s)

        s = re.sub(r"([bcdfghjklmnpqrstvwxyz])y\+([e])", r"\1i+\2", s)
        s = re.sub(r"([sz]|sh|ch|x)\+s\b", r"\1+es", s)
        surface = s.replace("+", "")
        return surface

    def generate(self, stem: str, feature: str) -> List[str]:
        if stem not in self.lexicon:
            return []
        meta = self.lexicon[stem]
        results = []

        if meta["cat"] == "ROOT_DERIV":
            for deriv_aff, target_pos, meaning in self.derivational_rules.get("ROOT_DERIV", []):
                if deriv_aff.strip("+") in feature:
                    interm = f"{stem}{deriv_aff}"
                    surf = self.apply_orthographic_rules(interm)
                    if "+PL" in feature:
                        interm_pl = f"{surf}+s"
                        surf_pl = self.apply_orthographic_rules(interm_pl)
                        results.append(surf_pl)
                    else:
                        results.append(surf)

        for pos in meta["pos"]:
            for infl_aff, infl_feat in self.inflectional_rules.get(pos, []):
                if infl_feat == feature:
                    interm = f"{stem}{infl_aff}"
                    surface = self.apply_orthographic_rules(interm)
                    results.append(surface)

        return sorted(list(set(results)))

    def parse(self, surface_word: str) -> List[Dict[str, Any]]:
        word = surface_word.lower()
        analyses = []

        if word in self.lexicon:
            meta = self.lexicon[word]
            for p in meta["pos"]:
                analyses.append({
                    "surface": surface_word,
                    "stem": word,
                    "pos": p,
                    "feature": f"+{p} +BASE",
                    "intermediate": word
                })

        for stem, meta in self.lexicon.items():
            if meta["cat"] == "ROOT_DERIV":
                for deriv_aff, target_pos, meaning in self.derivational_rules.get("ROOT_DERIV", []):
                    interm = f"{stem}{deriv_aff}"
                    surf = self.apply_orthographic_rules(interm)
                    if surf == word:
                        analyses.append({
                            "surface": surface_word,
                            "stem": stem,
                            "pos": target_pos,
                            "feature": f"+{target_pos} {deriv_aff}",
                            "intermediate": interm
                        })
                    interm_pl = f"{surf}+s"
                    surf_pl = self.apply_orthographic_rules(interm_pl)
                    if surf_pl == word:
                        analyses.append({
                            "surface": surface_word,
                            "stem": stem,
                            "pos": target_pos,
                            "feature": f"+{target_pos} {deriv_aff} +PL",
                            "intermediate": interm_pl
                        })

            for p in meta["pos"]:
                for infl_aff, infl_feat in self.inflectional_rules.get(p, []):
                    if not infl_aff:
                        continue
                    interm = f"{stem}{infl_aff}"
                    surf = self.apply_orthographic_rules(interm)
                    if surf == word:
                        analyses.append({
                            "surface": surface_word,
                            "stem": stem,
                            "pos": p,
                            "feature": infl_feat,
                            "intermediate": interm
                        })

        return analyses

    def trace_hand_analyses(self, surface_forms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if surface_forms is None:
            surface_forms = [
                "fractures",
                "injected",
                "biopsies",
                "scanned",
                "appendectomy",
                "abscesses"
            ]
        traces = []
        for sf in surface_forms:
            parsed = self.parse(sf)
            traces.append({
                "surface_form": sf,
                "analyses_count": len(parsed),
                "analyses": parsed
            })
        return traces

    def test_twenty_orthographic_forms(self) -> List[Dict[str, Any]]:
        test_cases = [
            ("fracture", "+ed", "fractured", "e-deletion"),
            ("fracture", "+ing", "fracturing", "e-deletion"),
            ("dilute", "+ed", "diluted", "e-deletion"),
            ("scan", "+ed", "scanned", "consonant-doubling"),
            ("scan", "+ing", "scanning", "consonant-doubling"),
            ("biopsy", "+es", "biopsies", "y-to-i"),
            ("biopsy", "+ed", "biopsied", "y-to-i"),
            ("artery", "+es", "arteries", "y-to-i"),
            ("abscess", "+s", "abscesses", "sibilant-es"),
            ("inject", "+s", "injects", "regular-concatenation"),
            ("inject", "+ed", "injected", "regular-concatenation"),
            ("inject", "+ing", "injecting", "regular-concatenation"),
            ("ulcer", "+s", "ulcers", "regular-concatenation"),
            ("lesion", "+s", "lesions", "regular-concatenation"),
            ("tendon", "+s", "tendons", "regular-concatenation"),
            ("nephr", "+itis", "nephritis", "derivational-boundary"),
            ("nephr", "+ectomy", "nephrectomy", "derivational-boundary"),
            ("append", "+ectomy", "appendectomy", "derivational-boundary"),
            ("arthr", "+itis", "arthritis", "derivational-boundary"),
            ("arthr", "+oplasty", "arthroplasty", "derivational-boundary")
        ]
        results = []
        for stem, affix, expected, rule_name in test_cases:
            intermediate = f"{stem}{affix}"
            generated = self.apply_orthographic_rules(intermediate)
            matched = (generated == expected)
            results.append({
                "stem": stem,
                "affix": affix,
                "intermediate": intermediate,
                "surface_generated": generated,
                "expected": expected,
                "rule_exercised": rule_name,
                "status": "PASS" if matched else "FAIL"
            })
        return results


