"""Top-Down (Recursive Descent) Parser with Left-Corner filtering and left-recursion protection."""

from typing import Dict, List, Optional, Set, Tuple
from .cfg_grammar import HelpdeskCFG, CFGRule
from .derivation_trees import ParseNode


class TopDownParser:
    def __init__(self, grammar: HelpdeskCFG, max_depth: int = 25):
        self.grammar = grammar
        self.max_depth = max_depth
        self.left_corners: Dict[str, Set[str]] = {}
        self.build_left_corner_table()

    def build_left_corner_table(self):
        for nt in self.grammar.non_terminals:
            self.left_corners[nt] = set()

        changed = True
        while changed:
            changed = False
            for rule in self.grammar.rules:
                if rule.rhs:
                    first = rule.rhs[0]
                    if first not in self.left_corners[rule.lhs]:
                        self.left_corners[rule.lhs].add(first)
                        changed = True
                    if first in self.left_corners:
                        for descendant in self.left_corners[first]:
                            if descendant not in self.left_corners[rule.lhs]:
                                self.left_corners[rule.lhs].add(descendant)
                                changed = True

        for sym, corners in self.left_corners.items():
            corners.add(sym)

    def can_start_with(self, non_terminal: str, token: str) -> bool:
        token_pos = [entry.pos for entry in self.grammar.lexicon.get(token.lower(), [])]
        corners = self.left_corners.get(non_terminal, set())
        if token in corners:
            return True
        for pos in token_pos:
            if pos in corners:
                return True
        return False

    def parse(self, sentence: str) -> Tuple[Optional[ParseNode], int, List[str]]:
        tokens = sentence.strip().split()
        trace_log: List[str] = []
        step_counter = [0]

        def match_symbol(symbol: str, idx: int, depth: int) -> List[Tuple[ParseNode, int]]:
            step_counter[0] += 1
            if depth > self.max_depth:
                trace_log.append(f"Depth limit ({self.max_depth}) exceeded at symbol {symbol}, index {idx}")
                return []

            if idx < len(tokens):
                curr_tok = tokens[idx]
                if symbol in self.grammar.non_terminals and not self.can_start_with(symbol, curr_tok):
                    return []

            if symbol not in self.grammar.non_terminals:
                if idx < len(tokens) and tokens[idx].lower() == symbol.lower():
                    trace_log.append(f"SCAN matched terminal '{symbol}' at position {idx}")
                    return [(ParseNode(symbol, token=symbol), idx + 1)]
                pos_entries = self.grammar.lexicon.get(tokens[idx].lower(), []) if idx < len(tokens) else []
                for entry in pos_entries:
                    if entry.pos == symbol:
                        trace_log.append(f"SCAN matched lexical category '{symbol}' -> '{tokens[idx]}' at position {idx}")
                        return [(ParseNode(symbol, token=tokens[idx]), idx + 1)]
                return []

            results = []
            rules = self.grammar.get_rules_for_lhs(symbol)
            for rule in rules:
                if rule.rhs and rule.rhs[0] == symbol:
                    continue

                trace_log.append(f"EXPAND {rule.lhs} -> {' '.join(rule.rhs)} at index {idx}")

                def match_rhs(rhs_symbols: Tuple[str, ...], cur_i: int, d: int) -> List[Tuple[List[ParseNode], int]]:
                    if not rhs_symbols:
                        return [([], cur_i)]
                    first_sym = rhs_symbols[0]
                    rest_syms = rhs_symbols[1:]

                    first_matches = match_symbol(first_sym, cur_i, d + 1)
                    combined_matches = []
                    for node1, next_i in first_matches:
                        rest_matches = match_rhs(rest_syms, next_i, d)
                        for rest_nodes, final_i in rest_matches:
                            combined_matches.append(([node1] + rest_nodes, final_i))
                    return combined_matches

                sub_matches = match_rhs(rule.rhs, idx, depth)
                for children, final_pos in sub_matches:
                    results.append((ParseNode(symbol, children=children), final_pos))

            return results

        all_parses = match_symbol(self.grammar.start_symbol, 0, 0)
        valid_parses = [node for node, end_idx in all_parses if end_idx == len(tokens)]

        if valid_parses:
            return valid_parses[0], step_counter[0], trace_log[:50]
        return None, step_counter[0], trace_log[:50]
