"""
Module: brill_tagger.py
Transformation-Based (Brill) Learning Part-of-Speech Tagger
DSA03 - Natural Language Processing (Assignment 2 - Task 9)
Author: Transport Speech Analytics & Language Engineering Team
Learns sequential deterministic correction rules from contextual error templates.
Evaluates error reductions, traces rule generalizability, and plots learning curves.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class BrillRule:
    def __init__(self, from_tag: str, to_tag: str, template_type: str, condition_value: Any):
        self.from_tag = from_tag
        self.to_tag = to_tag
        self.template_type = template_type
        self.condition_value = condition_value

    def matches(self, idx: int, words: List[str], current_tags: List[str]) -> bool:
        if current_tags[idx] != self.from_tag:
            return False

        if self.template_type == "PREV_TAG":
            prev_t = current_tags[idx - 1] if idx > 0 else "<s>"
            return prev_t == self.condition_value

        elif self.template_type == "NEXT_TAG":
            next_t = current_tags[idx + 1] if idx < len(words) - 1 else "</s>"
            return next_t == self.condition_value

        elif self.template_type == "PREV_WORD":
            prev_w = words[idx - 1].lower() if idx > 0 else "<s>"
            return prev_w == self.condition_value

        elif self.template_type == "NEXT_WORD":
            next_w = words[idx + 1].lower() if idx < len(words) - 1 else "</s>"
            return next_w == self.condition_value

        elif self.template_type == "SUFFIX":
            w = words[idx].lower()
            return w.endswith(self.condition_value)

        elif self.template_type == "PREV_TWO_TAGS":
            t_minus_2 = current_tags[idx - 2] if idx > 1 else "<s>"
            t_minus_1 = current_tags[idx - 1] if idx > 0 else "<s>"
            return (t_minus_2, t_minus_1) == self.condition_value

        return False

    def __repr__(self):
        return f"{self.from_tag} -> {self.to_tag} WHEN {self.template_type}='{self.condition_value}'"

class BrillTagger:
    def __init__(self, training_sentences: List[List[Tuple[str, str]]]):
        self.sentences = training_sentences
        self.word_mft: Dict[str, str] = {}
        self.rules: List[Tuple[BrillRule, int, int, int]] = []
        self._build_mft()

    def _build_mft(self):
        counts = defaultdict(Counter)
        for sent in self.sentences:
            for w, tag in sent:
                counts[w.lower()][tag] += 1
        for w, tag_counts in counts.items():
            self.word_mft[w] = tag_counts.most_common(1)[0][0]

    def initial_tag_sentence(self, words: List[str]) -> List[str]:
        return [self.word_mft.get(w.lower(), "NN") for w in words]

    def train(self, max_rules: int = 15, min_score: int = 2) -> List[Dict[str, Any]]:
        # Initialize working copy of tags
        words_corpus = [[w for w, _ in s] for s in self.sentences]
        gold_corpus = [[tag for _, tag in s] for s in self.sentences]
        current_corpus = [self.initial_tag_sentence(s) for s in words_corpus]

        learned_summary = []

        for step in range(max_rules):
            best_rule = None
            best_score = 0
            best_fixed = 0
            best_introduced = 0

            # Scan error positions to generate candidate rules
            candidate_rules: Set[Tuple[str, str, str, Any]] = set()
            for s_idx in range(len(words_corpus)):
                words = words_corpus[s_idx]
                gold = gold_corpus[s_idx]
                curr = current_corpus[s_idx]

                for i in range(len(words)):
                    if curr[i] != gold[i]:
                        from_t = curr[i]
                        to_t = gold[i]
                        w = words[i].lower()

                        prev_t = curr[i - 1] if i > 0 else "<s>"
                        next_t = curr[i + 1] if i < len(words) - 1 else "</s>"
                        prev_w = words[i - 1].lower() if i > 0 else "<s>"
                        next_w = words[i + 1].lower() if i < len(words) - 1 else "</s>"

                        candidate_rules.add((from_t, to_t, "PREV_TAG", prev_t))
                        candidate_rules.add((from_t, to_t, "NEXT_TAG", next_t))
                        candidate_rules.add((from_t, to_t, "PREV_WORD", prev_w))
                        candidate_rules.add((from_t, to_t, "NEXT_WORD", next_w))
                        if len(w) > 4:
                            candidate_rules.add((from_t, to_t, "SUFFIX", w[-3:]))
                        if i > 1:
                            candidate_rules.add((from_t, to_t, "PREV_TWO_TAGS", (curr[i - 2], prev_t)))

            # Evaluate each candidate rule
            for from_t, to_t, t_type, cond_val in candidate_rules:
                rule = BrillRule(from_t, to_t, t_type, cond_val)
                fixed = 0
                introduced = 0

                for s_idx in range(len(words_corpus)):
                    words = words_corpus[s_idx]
                    gold = gold_corpus[s_idx]
                    curr = current_corpus[s_idx]

                    for i in range(len(words)):
                        if rule.matches(i, words, curr):
                            if gold[i] == to_t:
                                fixed += 1
                            elif gold[i] == from_t:
                                introduced += 1

                score = fixed - introduced
                if score > best_score:
                    best_score = score
                    best_rule = rule
                    best_fixed = fixed
                    best_introduced = introduced

            if not best_rule or best_score < min_score:
                break

            self.rules.append((best_rule, best_score, best_fixed, best_introduced))
            learned_summary.append({
                "rule_number": step + 1,
                "rule_string": str(best_rule),
                "net_score": best_score,
                "errors_corrected": best_fixed,
                "errors_introduced": best_introduced
            })

            # Apply best rule to update current corpus
            for s_idx in range(len(words_corpus)):
                words = words_corpus[s_idx]
                curr = current_corpus[s_idx]
                for i in range(len(words)):
                    if best_rule.matches(i, words, curr):
                        curr[i] = best_rule.to_tag

        return learned_summary

    def tag_sentence(self, words: List[str]) -> List[str]:
        tags = self.initial_tag_sentence(words)
        for rule, _, _, _ in self.rules:
            for i in range(len(words)):
                if rule.matches(i, words, tags):
                    tags[i] = rule.to_tag
        return tags

    def evaluate_learning_curve(self, test_sentences: List[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        words_corpus = [[w for w, _ in s] for s in test_sentences]
        gold_corpus = [[tag for _, tag in s] for s in test_sentences]
        curr_corpus = [self.initial_tag_sentence(s) for s in words_corpus]

        total_tokens = sum(len(s) for s in words_corpus)
        initial_correct = sum(
            sum(1 for i in range(len(s)) if curr_corpus[idx][i] == gold_corpus[idx][i])
            for idx, s in enumerate(words_corpus)
        )
        curve = [{
            "rules_applied": 0,
            "accuracy_pct": round((initial_correct / total_tokens) * 100, 2)
        }]

        for r_idx, (rule, _, _, _) in enumerate(self.rules, start=1):
            for s_idx in range(len(words_corpus)):
                words = words_corpus[s_idx]
                curr = curr_corpus[s_idx]
                for i in range(len(words)):
                    if rule.matches(i, words, curr):
                        curr[i] = rule.to_tag

            correct = sum(
                sum(1 for i in range(len(s)) if curr_corpus[idx][i] == gold_corpus[idx][i])
                for idx, s in enumerate(words_corpus)
            )
            curve.append({
                "rules_applied": r_idx,
                "accuracy_pct": round((correct / total_tokens) * 100, 2)
            })

        return curve
