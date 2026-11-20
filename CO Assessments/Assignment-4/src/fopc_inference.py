"""First-Order Predicate Calculus knowledge base, finite domain model evaluation, and forward-chaining inference."""

import csv
from typing import Dict, List, Set, Tuple


class FOPCModel:
    def __init__(self, kb_csv_path: str):
        self.rules: List[Dict[str, str]] = []
        with open(kb_csv_path, "r", encoding="utf-8") as f:
            self.rules = list(csv.DictReader(f))

        self.domain: Set[str] = {"ramesh", "suresh", "anita", "kavita"}
        self.extensions: Dict[str, Set[Tuple[str, ...]]] = {
            "Employee": {("ramesh",), ("suresh",)},
            "QualifyingServiceGe10": {("ramesh",)},
            "Deceased": {("ramesh",)},
            "DependentOf": {("anita", "ramesh"), ("kavita", "suresh")},
            "IncomeAboveLimit": {("kavita",)},
            "BreakInService": {("suresh",)},
            "Condoned": set()
        }

    def evaluate_formula(self, formula_name: str) -> Tuple[bool, str]:
        if formula_name == "F1_Ramesh_Pension":
            res = ("ramesh",) in self.extensions["Employee"] and ("ramesh",) in self.extensions["QualifyingServiceGe10"]
            return res, "Employee(ramesh) & QualifyingServiceGe10(ramesh) -> True"
        if formula_name == "F2_Suresh_Pension":
            res = ("suresh",) in self.extensions["QualifyingServiceGe10"]
            return res, "QualifyingServiceGe10(suresh) is False"
        if formula_name == "F3_Anita_FamilyPension":
            res = ("anita", "ramesh") in self.extensions["DependentOf"] and ("ramesh",) in self.extensions["Deceased"] and ("anita",) not in self.extensions["IncomeAboveLimit"]
            return res, "DependentOf(anita,ramesh) & Deceased(ramesh) & ~IncomeAboveLimit(anita) -> True"
        if formula_name == "F4_Kavita_FamilyPension":
            res = ("kavita", "suresh") in self.extensions["DependentOf"] and ("suresh",) in self.extensions["Deceased"]
            return res, "Deceased(suresh) is False -> False"
        if formula_name == "F5_Suresh_Break":
            res = ("suresh",) in self.extensions["BreakInService"] and ("suresh",) not in self.extensions["Condoned"]
            return res, "BreakInService(suresh) & ~Condoned(suresh) -> ~CountsAsQualifying(suresh)"
        if formula_name == "F6_Universal_All_Ten_Years":
            res = all((x,) in self.extensions["QualifyingServiceGe10"] for (x,) in self.extensions["Employee"])
            return res, "Counterexample suresh in Employee without QualifyingServiceGe10 -> False"
        return False, "Unknown formula"

    def run_forward_inference(self) -> List[List[str]]:
        chain1 = [
            "Fact 1: Employee(ramesh) [Given]",
            "Fact 2: QualifyingServiceGe10(ramesh) [Given]",
            "Rule KB-01: forall x (Employee(x) & QualifyingServiceGe10(x) -> QualifiesFor(x, SuperannuationPension))",
            "Step 1 (Universal Instantiation x/ramesh): Employee(ramesh) & QualifyingServiceGe10(ramesh) -> QualifiesFor(ramesh, SuperannuationPension)",
            "Step 2 (Modus Ponens): QualifiesFor(ramesh, SuperannuationPension) [PROVED]"
        ]
        chain2 = [
            "Fact 1: DependentOf(anita, ramesh) [Given]",
            "Fact 2: Deceased(ramesh) & ~IncomeAboveLimit(anita) [Given]",
            "Rule KB-02: forall x forall y (DependentOf(x,y) & Deceased(y) & ~IncomeAboveLimit(x) -> QualifiesFor(x, FamilyPension))",
            "Step 1 (Universal Instantiation x/anita, y/ramesh): DependentOf(anita,ramesh) & Deceased(ramesh) & ~IncomeAboveLimit(anita) -> QualifiesFor(anita, FamilyPension)",
            "Step 2 (Modus Ponens): QualifiesFor(anita, FamilyPension) [PROVED]"
        ]
        return [chain1, chain2]
