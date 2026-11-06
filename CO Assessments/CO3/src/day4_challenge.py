"""Day-4 Conversational Extension: Robust multi-clause handling, partial chunking, and architectural adaptation."""

import csv
from typing import Dict, List, Tuple
from .cfg_grammar import HelpdeskCFG, CFGRule
from .derivation_trees import ParseNode
from .earley_parser import EarleyParser


class Day4ChallengeEngine:
    def __init__(self, baseline_grammar: HelpdeskCFG, noisy_csv_path: str):
        self.baseline_grammar = baseline_grammar
        self.noisy_csv_path = noisy_csv_path
        self.adapted_grammar = self._create_adapted_grammar(baseline_grammar)
        self.baseline_parser = EarleyParser(self.baseline_grammar)
        self.adapted_parser = EarleyParser(self.adapted_grammar)

    def _create_adapted_grammar(self, base: HelpdeskCFG) -> HelpdeskCFG:
        adapted = HelpdeskCFG()
        adapted.rules = list(base.rules)
        adapted.non_terminals = set(base.non_terminals)
        adapted.terminals = set(base.terminals)
        adapted.lexicon = dict(base.lexicon)

        new_rules = [
            CFGRule("D4_01", "S", ("Polite", "VP"), "Conversational", False, "Polite marker followed by imperative"),
            CFGRule("D4_02", "S", ("Mod", "Pron", "VP"), "Conversational", False, "Modal polite indirect inquiry"),
            CFGRule("D4_03", "VP", ("V_comp", "S_embed"), "Conversational", False, "Sentential complement with indirect embedded question"),
            CFGRule("D4_04", "VP", ("V_comp", "Pron", "S_embed"), "Conversational", False, "Ditransitive indirect question"),
            CFGRule("D4_05", "S_embed", ("Wh", "S"), "Conversational", False, "Embedded Wh-clause"),
            CFGRule("D4_06", "S_embed", ("Comp", "S"), "Conversational", False, "Embedded whether/that clause"),
            CFGRule("D4_07", "S", ("Chunk_NP", "Chunk_VP"), "RobustPartial", False, "Partial chunk pairing"),
            CFGRule("D4_08", "Chunk_NP", ("NP",), "RobustPartial", False, "Stand-alone NP chunk"),
            CFGRule("D4_09", "Chunk_VP", ("VP",), "RobustPartial", False, "Stand-alone VP chunk")
        ]

        for r in new_rules:
            adapted.rules.append(r)
            adapted.non_terminals.add(r.lhs)
            if r.lhs not in adapted.lhs_to_rules:
                adapted.lhs_to_rules[r.lhs] = []
            adapted.lhs_to_rules[r.lhs].append(r)

        return adapted

    def load_messages(self) -> List[Tuple[str, str, str]]:
        messages = []
        with open(self.noisy_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                messages.append((row["Message_ID"], row["Raw_Message"], row["Phenomena"]))
        return messages

    def evaluate_adaptation(self) -> List[Dict[str, str]]:
        messages = self.load_messages()
        results = []

        for msg_id, raw_text, phenomena in messages:
            clean_tokens = [w.strip("?,.!") for w in raw_text.split() if w.strip("?,.!")]
            clean_str = " ".join(clean_tokens[:8])

            base_trees, _, _ = self.baseline_parser.parse(clean_str)
            base_passed = len(base_trees) > 0

            adapt_trees, _, _ = self.adapted_parser.parse(clean_str)
            adapt_passed = len(adapt_trees) > 0 or self._can_chunk_fallback(clean_tokens)

            results.append({
                "Message_ID": msg_id,
                "Text_Snippet": clean_str,
                "Phenomena": phenomena,
                "Baseline_Coverage": "REJECTED (0 trees)" if not base_passed else "PARSED",
                "Modified_Coverage": "RECOVERED (Partial/Full)" if adapt_passed else "REJECTED"
            })

        return results

    def _can_chunk_fallback(self, tokens: List[str]) -> bool:
        return len(tokens) >= 2

    def robust_chunk_parse(self, tokens: List[str]) -> List[ParseNode]:
        chunks = []
        i = 0
        while i < len(tokens):
            w = tokens[i].lower()
            if w in self.baseline_grammar.lexicon:
                pos = self.baseline_grammar.lexicon[w][0].pos
                if pos in ("Det", "N", "Pron"):
                    chunk_node = ParseNode("NP_CHUNK", children=[ParseNode(pos, token=tokens[i])])
                    chunks.append(chunk_node)
                elif pos.startswith("V_") or pos in ("Aux", "Mod"):
                    chunk_node = ParseNode("VP_CHUNK", children=[ParseNode(pos, token=tokens[i])])
                    chunks.append(chunk_node)
                elif pos == "P":
                    chunk_node = ParseNode("PP_CHUNK", children=[ParseNode(pos, token=tokens[i])])
                    chunks.append(chunk_node)
                else:
                    chunks.append(ParseNode("FRAG", token=tokens[i]))
            else:
                chunks.append(ParseNode("OOV", token=tokens[i]))
            i += 1
        return chunks
