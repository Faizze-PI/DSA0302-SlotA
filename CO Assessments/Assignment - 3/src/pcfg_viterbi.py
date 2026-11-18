"""Probabilistic CFG estimation from treebanks and Viterbi CKY / CYK dynamic programming parser."""

import csv
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from .derivation_trees import ParseNode, from_bracketed


class PCFGRule:
    def __init__(self, lhs: str, rhs: Tuple[str, ...], prob: float):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)} [p={self.prob:.4f}]"


class PCFGModel:
    def __init__(self):
        self.rules: Dict[str, List[PCFGRule]] = defaultdict(list)
        self.rule_probs: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.rhs_to_rules: Dict[Tuple[str, ...], List[PCFGRule]] = defaultdict(list)
        self.non_terminals: Set[str] = set()

    def train_from_treebank(self, treebank_csv_path: str, add_k: float = 0.05):
        production_counts: Dict[str, Dict[Tuple[str, ...], float]] = defaultdict(lambda: defaultdict(float))
        lhs_totals: Dict[str, float] = defaultdict(float)

        with open(treebank_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bracketed = row["Bracketed_Parse"].strip()
                tree = from_bracketed(bracketed)
                self._extract_productions(tree, production_counts, lhs_totals)

        for lhs, rhs_dict in production_counts.items():
            self.non_terminals.add(lhs)
            total = lhs_totals[lhs] + len(rhs_dict) * add_k
            for rhs, count in rhs_dict.items():
                prob = (count + add_k) / total
                pcfg_rule = PCFGRule(lhs, rhs, prob)
                self.rules[lhs].append(pcfg_rule)
                self.rule_probs[(lhs, rhs)] = prob
                self.rhs_to_rules[rhs].append(pcfg_rule)

    def _extract_productions(self, node: ParseNode, counts: Dict[str, Dict[Tuple[str, ...], float]], totals: Dict[str, float]):
        if node.token is not None:
            rhs = (node.token.lower(),)
            counts[node.symbol][rhs] += 1.0
            totals[node.symbol] += 1.0
            return

        if not node.children:
            return

        rhs = tuple(c.symbol for c in node.children)
        counts[node.symbol][rhs] += 1.0
        totals[node.symbol] += 1.0

        for child in node.children:
            self._extract_productions(child, counts, totals)

    def get_prob(self, lhs: str, rhs: Tuple[str, ...], default: float = 1e-6) -> float:
        return self.rule_probs.get((lhs, rhs), default)


class ViterbiCKYParser:
    def __init__(self, pcfg: PCFGModel):
        self.pcfg = pcfg

    def parse(self, sentence: str) -> Tuple[Optional[ParseNode], float, Dict[Tuple[int, int, str], float]]:
        tokens = sentence.strip().split()
        n = len(tokens)
        if n == 0:
            return None, 0.0, {}

        table: Dict[Tuple[int, int, str], float] = defaultdict(float)
        back: Dict[Tuple[int, int, str], Any] = {}

        for i, word in enumerate(tokens):
            w_lower = word.lower()
            for r in self.pcfg.rhs_to_rules.get((w_lower,), []):
                table[(i, i + 1, r.lhs)] = r.prob
                back[(i, i + 1, r.lhs)] = ("LEAF", word)

            single_tokens = [(r.lhs, r.prob) for r in self.pcfg.rhs_to_rules.get((w_lower,), [])]
            for pos_tag, pos_prob in single_tokens:
                for r in self.pcfg.rhs_to_rules.get((pos_tag,), []):
                    unary_prob = r.prob * pos_prob
                    if unary_prob > table[(i, i + 1, r.lhs)]:
                        table[(i, i + 1, r.lhs)] = unary_prob
                        back[(i, i + 1, r.lhs)] = ("UNARY", pos_tag)

        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    left_syms = [s for (start, end, s), p in table.items() if start == i and end == k and p > 0]
                    right_syms = [s for (start, end, s), p in table.items() if start == k and end == j and p > 0]

                    for b in left_syms:
                        for c in right_syms:
                            for r in self.pcfg.rhs_to_rules.get((b, c), []):
                                prob = r.prob * table[(i, k, b)] * table[(k, j, c)]
                                if prob > table[(i, j, r.lhs)]:
                                    table[(i, j, r.lhs)] = prob
                                    back[(i, j, r.lhs)] = (k, b, c)

                cur_syms = [s for (start, end, s), p in table.items() if start == i and end == j and p > 0]
                for b in cur_syms:
                    for r in self.pcfg.rhs_to_rules.get((b,), []):
                        unary_prob = r.prob * table[(i, j, b)]
                        if unary_prob > table[(i, j, r.lhs)]:
                            table[(i, j, r.lhs)] = unary_prob
                            back[(i, j, r.lhs)] = ("UNARY", b)

        best_root = "S"
        best_prob = table.get((0, n, best_root), 0.0)
        if best_prob == 0.0:
            for s in ("S", "VP", "NP"):
                if table.get((0, n, s), 0.0) > best_prob:
                    best_prob = table[(0, n, s)]
                    best_root = s

        if best_prob > 0.0:
            tree = self._recover_tree(0, n, best_root, back, tokens)
            tree.prob = best_prob
            return tree, best_prob, table
        return None, 0.0, table

    def _recover_tree(self, i: int, j: int, sym: str, back: Dict[Tuple[int, int, str], Any], tokens: List[str]) -> ParseNode:
        info = back.get((i, j, sym))
        if not info:
            return ParseNode(sym)

        if info[0] == "LEAF":
            return ParseNode(sym, token=info[1])
        elif info[0] == "UNARY":
            child = self._recover_tree(i, j, info[1], back, tokens)
            return ParseNode(sym, children=[child])
        else:
            k, b, c = info
            left_child = self._recover_tree(i, k, b, back, tokens)
            right_child = self._recover_tree(k, j, c, back, tokens)
            return ParseNode(sym, children=[left_child, right_child])

    def calculate_tree_probability(self, node: ParseNode) -> float:
        if node.is_leaf() or not node.children:
            return 1.0
        if all(c.token is not None for c in node.children):
            rhs = tuple(c.token.lower() for c in node.children)
        else:
            rhs = tuple(c.symbol for c in node.children)

        prob = self.pcfg.get_prob(node.symbol, rhs, default=0.01)
        for child in node.children:
            prob *= self.calculate_tree_probability(child)
        return prob
