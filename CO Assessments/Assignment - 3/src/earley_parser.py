"""Dynamic programming Earley Chart Parser with PREDICTOR, SCANNER, COMPLETER, and tree extraction."""

from typing import Dict, List, Optional, Set, Tuple
from .cfg_grammar import HelpdeskCFG, CFGRule
from .derivation_trees import ParseNode


class EarleyState:
    def __init__(self, rule: CFGRule, dot: int, origin: int, operation: str = "", backpointers: Optional[List[Tuple['EarleyState', Optional[ParseNode]]]] = None):
        self.rule = rule
        self.dot = dot
        self.origin = origin
        self.operation = operation
        self.backpointers = backpointers if backpointers is not None else []

    def is_complete(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    def next_symbol(self) -> Optional[str]:
        if self.is_complete():
            return None
        return self.rule.rhs[self.dot]

    def advance(self, operation: str, child_node: Optional[ParseNode] = None) -> 'EarleyState':
        new_bp = list(self.backpointers)
        new_bp.append((self, child_node))
        return EarleyState(self.rule, self.dot + 1, self.origin, operation, new_bp)

    def key(self) -> Tuple[str, Tuple[str, ...], int, int]:
        return (self.rule.lhs, self.rule.rhs, self.dot, self.origin)

    def __repr__(self) -> str:
        rhs_str = list(self.rule.rhs)
        rhs_str.insert(self.dot, ".")
        return f"{self.rule.lhs} -> {' '.join(rhs_str)} [{self.origin}] ({self.operation})"


class EarleyParser:
    def __init__(self, grammar: HelpdeskCFG):
        self.grammar = grammar

    def parse(self, sentence: str) -> Tuple[List[ParseNode], List[List[EarleyState]], List[str]]:
        tokens = sentence.strip().split()
        n = len(tokens)
        chart: List[List[EarleyState]] = [[] for _ in range(n + 1)]
        state_sets: List[Set[Tuple[str, Tuple[str, ...], int, int]]] = [set() for _ in range(n + 1)]
        trace_logs: List[str] = []

        start_rule = CFGRule("ROOT_INIT", "ROOT", (self.grammar.start_symbol,), "Root", False, "Initial state")
        init_state = EarleyState(start_rule, 0, 0, "INIT")
        chart[0].append(init_state)
        state_sets[0].add(init_state.key())

        def add_state(state: EarleyState, chart_idx: int) -> bool:
            k = state.key()
            if k not in state_sets[chart_idx]:
                state_sets[chart_idx].add(k)
                chart[chart_idx].append(state)
                return True
            else:
                for existing in chart[chart_idx]:
                    if existing.key() == k:
                        for bp in state.backpointers:
                            if bp not in existing.backpointers:
                                existing.backpointers.append(bp)
                return False

        for j in range(n + 1):
            i_idx = 0
            while i_idx < len(chart[j]):
                curr = chart[j][i_idx]
                i_idx += 1

                if not curr.is_complete():
                    next_sym = curr.next_symbol()
                    if next_sym in self.grammar.non_terminals:
                        rules = self.grammar.get_rules_for_lhs(next_sym)
                        for r in rules:
                            new_pred = EarleyState(r, 0, j, f"PREDICTOR from {curr.rule.lhs}")
                            if add_state(new_pred, j):
                                trace_logs.append(f"S[{j}] PREDICT: {new_pred}")
                    else:
                        if j < n:
                            tok = tokens[j].lower()
                            matched = False
                            if next_sym.lower() == tok:
                                matched = True
                            else:
                                for entry in self.grammar.lexicon.get(tok, []):
                                    if entry.pos == next_sym:
                                        matched = True
                                        break
                            if matched:
                                term_node = ParseNode(next_sym, token=tokens[j])
                                scanned = curr.advance(f"SCANNER '{tokens[j]}'", term_node)
                                add_state(scanned, j + 1)
                                trace_logs.append(f"S[{j+1}] SCAN: {scanned}")
                else:
                    completed_sym = curr.rule.lhs
                    origin_j = curr.origin
                    for candidate in chart[origin_j]:
                        if not candidate.is_complete() and candidate.next_symbol() == completed_sym:
                            tree_frag = self._build_node_from_state(curr)
                            completed = candidate.advance(f"COMPLETER from S[{j}]", tree_frag)
                            if add_state(completed, j):
                                trace_logs.append(f"S[{j}] COMPLETE: {completed}")

        parse_trees = []
        for state in chart[n]:
            if state.rule.lhs == "ROOT" and state.is_complete() and state.origin == 0:
                tree = self._build_node_from_state(state)
                if tree.children:
                    parse_trees.append(tree.children[0])

        return parse_trees, chart, trace_logs

    def _build_node_from_state(self, state: EarleyState) -> ParseNode:
        if not state.backpointers:
            return ParseNode(state.rule.lhs)

        children: List[ParseNode] = []
        curr = state
        while curr and curr.backpointers:
            prev_state, child_node = curr.backpointers[-1]
            if child_node is not None:
                children.append(child_node)
            curr = prev_state

        children.reverse()
        return ParseNode(state.rule.lhs, children=children)

    def format_chart_trace(self, chart: List[List[EarleyState]], tokens: List[str]) -> str:
        lines = []
        for j, states in enumerate(chart):
            header = f"=== Chart S[{j}] (token: '{tokens[j-1]}' if j > 0 else 'START') ===" if j > 0 else "=== Chart S[0] (START) ==="
            lines.append(header)
            for idx, s in enumerate(states):
                lines.append(f"  [{idx:02d}] {s}")
            lines.append("")
        return "\n".join(lines)
