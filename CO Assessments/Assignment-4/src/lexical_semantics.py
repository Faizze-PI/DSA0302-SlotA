"""Domain sense inventory, homonymy vs polysemy, WordNet relations, and selectional restrictions."""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple


class LexicalSemanticsEngine:
    def __init__(self, inventory_csv_path: str):
        self.inventory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(inventory_csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.inventory[row["Term"]].append(row)

    def get_wordnet_relations(self) -> List[Tuple[str, str, str]]:
        return [
            ("pension", "Hypernym", "regular_payment -> income -> financial_gain"),
            ("family_pension", "Hyponym", "pension"),
            ("service_record", "Meronym", "service_book"),
            ("eligible", "Antonym", "ineligible"),
            ("circular", "Hypernym", "official_document -> document -> artifact"),
            ("dependent", "Hyponym", "beneficiary")
        ]

    def get_selectional_restrictions(self) -> List[Tuple[str, str, str, str]]:
        return [
            ("sanction", "Agent: Authority, Theme: FinancialBenefit", "The directorate sanctioned the pension.", "*The circular sanctioned the table."),
            ("retire", "Experiencer: Employee", "The employee retired in 2019.", "*The pension retired in 2019."),
            ("commute", "Agent: Pensioner, Theme: Pension", "The pensioner commuted forty percent.", "*The office commuted the circular."),
            ("condone", "Agent: CompetentAuthority, Theme: BreakInService", "The authority condoned the break.", "*The break condoned the authority."),
            ("revise", "Agent: FinanceDept, Theme: PayScale", "The department revised the pay scale.", "*The scale revised the department.")
        ]
