"""Formal Context-Free Grammar representation, lexicon indexing, and CNF binarisation."""

import csv
import os
from typing import Dict, List, Set, Tuple, Optional


class CFGRule:
    def __init__(self, rule_id: str, lhs: str, rhs: Tuple[str, ...], rule_type: str = "Standard", is_recursive: bool = False, description: str = ""):
        self.rule_id = rule_id
        self.lhs = lhs
        self.rhs = rhs
        self.rule_type = rule_type
        self.is_recursive = is_recursive
        self.description = description

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CFGRule):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))


class LexicalEntry:
    def __init__(self, token: str, pos: str, number: str = "any", person: str = "any", case: str = "any", subcat_frame: str = "none"):
        self.token = token.lower()
        self.pos = pos
        self.number = number
        self.person = person
        self.case = case
        self.subcat_frame = subcat_frame

    def __repr__(self) -> str:
        return f"{self.token}/{self.pos}[num={self.number},pers={self.person},case={self.case},subcat={self.subcat_frame}]"


class HelpdeskCFG:
    def __init__(self, rules_csv_path: Optional[str] = None, lexicon_csv_path: Optional[str] = None):
        self.start_symbol = "S"
        self.rules: List[CFGRule] = []
        self.lhs_to_rules: Dict[str, List[CFGRule]] = {}
        self.lexicon: Dict[str, List[LexicalEntry]] = {}
        self.pos_to_tokens: Dict[str, Set[str]] = {}
        self.non_terminals: Set[str] = set()
        self.terminals: Set[str] = set()

        if rules_csv_path and os.path.exists(rules_csv_path):
            self.load_rules(rules_csv_path)
        if lexicon_csv_path and os.path.exists(lexicon_csv_path):
            self.load_lexicon(lexicon_csv_path)

    def load_rules(self, csv_path: str):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lhs = row["LHS"].strip()
                rhs = tuple(row["RHS"].strip().split())
                rule_id = row.get("Rule_ID", f"R{len(self.rules)+1:02d}").strip()
                rule_type = row.get("Rule_Type", "Standard").strip()
                is_rec = row.get("Recursive_Flag", "False").strip().lower() in ("true", "1")
                desc = row.get("Description", "").strip()

                rule = CFGRule(rule_id, lhs, rhs, rule_type, is_rec, desc)
                self.rules.append(rule)
                self.non_terminals.add(lhs)

                if lhs not in self.lhs_to_rules:
                    self.lhs_to_rules[lhs] = []
                self.lhs_to_rules[lhs].append(rule)

    def load_lexicon(self, csv_path: str):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                token = row["Token"].strip().lower()
                pos = row["POS"].strip()
                number = row.get("Number", "any").strip()
                person = row.get("Person", "any").strip()
                case = row.get("Case", "any").strip()
                subcat = row.get("Subcat_Frame", "none").strip()

                entry = LexicalEntry(token, pos, number, person, case, subcat)
                if token not in self.lexicon:
                    self.lexicon[token] = []
                self.lexicon[token].append(entry)

                if pos not in self.pos_to_tokens:
                    self.pos_to_tokens[pos] = set()
                self.pos_to_tokens[pos].add(token)
                self.terminals.add(token)

    def get_rules_for_lhs(self, lhs: str) -> List[CFGRule]:
        return self.lhs_to_rules.get(lhs, [])

    def is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminals or symbol.islower()

    def find_direct_left_recursion(self) -> List[CFGRule]:
        direct = []
        for r in self.rules:
            if r.rhs and r.rhs[0] == r.lhs:
                direct.append(r)
        return direct

    def find_indirect_left_recursion(self) -> List[Tuple[str, List[str]]]:
        indirect = []
        for nt in self.non_terminals:
            visited = set()
            stack = [(nt, [nt])]
            while stack:
                curr, path = stack.pop()
                for r in self.get_rules_for_lhs(curr):
                    if r.rhs:
                        first = r.rhs[0]
                        if first == nt and len(path) > 1:
                            indirect.append((nt, path + [first]))
                        elif first in self.non_terminals and first not in visited:
                            visited.add(first)
                            stack.append((first, path + [first]))
        return indirect

    def to_chomsky_normal_form(self) -> List[CFGRule]:
        cnf_rules = []
        rule_counter = 1

        for r in self.rules:
            if len(r.rhs) <= 2:
                cnf_rules.append(r)
            else:
                curr_lhs = r.lhs
                remaining = list(r.rhs)
                step = 1
                while len(remaining) > 2:
                    new_nt = f"{r.lhs}_CNF_{rule_counter}_{step}"
                    first_sym = remaining.pop(0)
                    cnf_rules.append(CFGRule(f"CNF_{rule_counter}_{step}", curr_lhs, (first_sym, new_nt), "Binarized", False, "CNF binarized rule"))
                    curr_lhs = new_nt
                    step += 1
                cnf_rules.append(CFGRule(f"CNF_{rule_counter}_{step}", curr_lhs, (remaining[0], remaining[1]), "Binarized", False, "CNF final binary rule"))
                rule_counter += 1

        return cnf_rules
