"""Meaning representation comparison, verifiability, canonical form, ambiguity, and expressiveness analysis."""

from typing import Dict, List, Tuple


class MeaningRepresentationEngine:
    def __init__(self):
        self.canonical_map: Dict[str, str] = {
            "does an employee with 10 years service get a pension": "forall x (Employee(x) & ServiceGE(x,10) -> Qualifies(x,Pension))",
            "is a 10-year employee eligible for pension": "forall x (Employee(x) & ServiceGE(x,10) -> Qualifies(x,Pension))",
            "do employees completing ten years of qualifying service receive pension": "forall x (Employee(x) & ServiceGE(x,10) -> Qualifies(x,Pension))",
            "can someone with ten years qualifying service claim a pension": "forall x (Employee(x) & ServiceGE(x,10) -> Qualifies(x,Pension))"
        }
        self.kb_facts = {
            ("Employee", "ramesh"): True,
            ("ServiceGE10", "ramesh"): True,
            ("QualifiesPension", "ramesh"): True,
            ("Employee", "suresh"): True,
            ("ServiceGE10", "suresh"): False,
            ("QualifiesPension", "suresh"): False
        }

    def compare_formalisms(self) -> List[Tuple[str, str, str, str]]:
        return [
            ("First-Order Logic (FOPC)", "High (quantifiers, negation, n-ary)", "Full deductive & resolution", "High"),
            ("Semantic Networks", "Medium (binary isa/has links)", "Inheritance & spreading activation", "Medium"),
            ("Frames", "Medium-High (typed slots)", "Procedural attachment & defaults", "Medium"),
            ("Slot-Filler Structures", "Low (flat key-value)", "Template matching only", "Low")
        ]

    def verify_statement(self, predicate: str, entity: str) -> str:
        val = self.kb_facts.get((predicate, entity.lower()))
        if val is True:
            return "TRUE"
        if val is False:
            return "FALSE"
        return "UNKNOWN"

    def to_canonical(self, query: str) -> str:
        norm = query.strip().lower().rstrip("?")
        return self.canonical_map.get(norm, f"RAW_LF({norm})")
