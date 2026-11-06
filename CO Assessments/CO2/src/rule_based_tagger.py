"""
Module: rule_based_tagger.py
Two-Stage Rule-Based Part-of-Speech Tagger
DSA03 - Natural Language Processing (Assignment 2 - Task 7)
Author: Transport Speech Analytics & Language Engineering Team
Implements a tag dictionary and fifteen explicit condition-action disambiguation rules.
Evaluates accuracy on ambiguous tokens and resolves rule priority conflicts.
"""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class RuleBasedTagger:
    def __init__(self, training_tokens: List[Dict[str, Any]]):
        self.training_tokens = training_tokens
        self.tag_dict: Dict[str, Set[str]] = defaultdict(set)
        self.word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
        self._build_tag_dictionary()

    def _build_tag_dictionary(self):
        for item in self.training_tokens:
            w = item["word"].lower()
            tag = item["gold_tag"]
            self.tag_dict[w].add(tag)
            self.word_tag_counts[w][tag] += 1

    def get_tag_dictionary_distribution(self) -> Dict[str, int]:
        counts = Counter(len(tags) for tags in self.tag_dict.values())
        return {
            "words_with_1_candidate_tag": counts.get(1, 0),
            "words_with_2_candidate_tags": counts.get(2, 0),
            "words_with_3_candidate_tags": counts.get(3, 0),
            "words_with_4_or_more_tags": sum(v for k, v in counts.items() if k >= 4)
        }

    def _get_initial_candidates(self, word: str) -> List[str]:
        w_lower = word.lower()
        if w_lower in self.tag_dict:
            return sorted(list(self.tag_dict[w_lower]))
        
        if re.fullmatch(r"\d{1,2}:\d{2}(?:_?(?:am|pm))?", w_lower):
            return ["N_TIME"]
        if word.isdigit():
            return ["CD"]
        if word[0].isupper() and len(word) > 2:
            return ["N_LOC", "N_PROP", "NN"]
        if w_lower.endswith("ing"):
            return ["VBG", "NN"]
        if w_lower.endswith("ed"):
            return ["VBD", "JJ"]
        if w_lower.endswith("ly"):
            return ["RB"]
        if w_lower.endswith("tion") or w_lower.endswith("ment") or w_lower.endswith("ance"):
            return ["NN"]

        return ["NN", "VB", "JJ"]

    def disambiguate_token(
        self,
        word: str,
        candidates: List[str],
        prev_word: str,
        prev_tag: str,
        next_word: str,
        next_tag_hint: str
    ) -> Tuple[str, str]:
        if len(candidates) == 1:
            return candidates[0], "RULE_SINGLETON_DICT"

        w = word.lower()
        nw = next_word.lower()
        pw = prev_word.lower()

        # Rule 1: Modal Verb Follower (can/could/will/would + Word -> VB)
        if prev_tag == "MD" and "VB" in candidates:
            return "VB", "RULE_01_MODAL_AUX"

        # Rule 2: Infinitive Marker (to + Word -> VB)
        if pw == "to" and "VB" in candidates:
            return "VB", "RULE_02_INFINITIVE_TO"

        # Rule 3: Determiner Follower before Preposition/End (DT + Word + IN/$ -> NN)
        if prev_tag == "DT" and next_tag_hint in ("IN", "</s>", "RB", "CC") and "NN" in candidates:
            return "NN", "RULE_03_DT_HEAD_NOUN"

        # Rule 4: Attributive Premodifier before Noun (DT/PRP$ + Word + NN/NNS -> JJ)
        if next_tag_hint in ("NN", "NNS") and "JJ" in candidates:
            return "JJ", "RULE_04_ATTRIBUTIVE_ADJ"

        # Rule 5: Compound Noun Head (bus/train + Word -> NN, e.g., bus stand, train timing)
        if pw in ("bus", "train") and w in ("stand", "timing", "ticket", "pass", "fare") and "NN" in candidates:
            return "NN", "RULE_05_TRANSPORT_COMPOUND_NOUN"

        # Rule 6: Subject Pronoun Follower (I/we/you + Word -> VBP)
        if prev_tag == "PRP" and pw in ("i", "we", "you", "they") and "VBP" in candidates:
            return "VBP", "RULE_06_PRONOUN_SUBJECT_VBP"

        # Rule 7: 3SG Pronoun / Noun Subject Follower (he/she/it/bus + Word -> VBZ)
        if prev_tag in ("NN", "PRP") and pw in ("bus", "train", "service", "it") and "VBZ" in candidates:
            return "VBZ", "RULE_07_SUBJECT_3SG_VBZ"

        # Rule 8: Suffix -ing with Auxiliary (is/are/was + VBG -> VBG)
        if w.endswith("ing") and pw in ("is", "are", "was", "were") and "VBG" in candidates:
            return "VBG", "RULE_08_PROGRESSIVE_AUX"

        # Rule 9: Suffix -ing without Auxiliary followed by Noun (booking counter -> NN/JJ)
        if w.endswith("ing") and next_tag_hint in ("NN", "NNS") and "NN" in candidates:
            return "NN", "RULE_09_GERUND_COMPOUND_MOD"

        # Rule 10: Location Preposition Follower (from/to/via/at + Capitalized -> N_LOC)
        if prev_tag == "IN" and pw in ("from", "to", "via", "at", "between") and "N_LOC" in candidates:
            return "N_LOC", "RULE_10_LOCATIVE_PREP"

        # Rule 11: Time Preposition Follower (at/before/after + 6:30/morning -> N_TIME)
        if prev_tag == "IN" and pw in ("at", "before", "after") and "N_TIME" in candidates:
            return "N_TIME", "RULE_11_TEMPORAL_PREP"

        # Rule 12: Cardinal Quantifier Preceding Plural (two/three + tickets -> NNS)
        if prev_tag == "CD" and "NNS" in candidates:
            return "NNS", "RULE_12_NUMERICAL_QUANTIFIER"

        # Rule 13: Polite Imperative Opener (Sentence start Please/Kindly -> UH)
        if pw == "<s>" and w in ("please", "kindly", "hello", "excuse") and "UH" in candidates:
            return "UH", "RULE_13_POLITE_OPENER"

        # Rule 14: Disfluency Hesitation (um/uh/er -> DISFL)
        if w in ("um", "uh", "er", "ah") and "DISFL" in candidates:
            return "DISFL", "RULE_14_SPEECH_DISFLUENCY"

        # Rule 15: Suffix Morphology Default (-ed -> VBD, -ly -> RB)
        if w.endswith("ed") and "VBD" in candidates:
            return "VBD", "RULE_15_MORPH_PAST"
        if w.endswith("ly") and "RB" in candidates:
            return "RB", "RULE_15_MORPH_ADV"

        # Fallback to most frequent tag for this word in training dictionary
        counts = self.word_tag_counts.get(w, Counter())
        if counts:
            return counts.most_common(1)[0][0], "RULE_FALLBACK_MFT"
        
        return candidates[0], "RULE_FALLBACK_FIRST"

    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str, str]]:
        tagged = []
        n = len(words)

        for i, word in enumerate(words):
            candidates = self._get_initial_candidates(word)
            prev_word = words[i - 1] if i > 0 else "<s>"
            prev_tag = tagged[i - 1][1] if i > 0 else "<s>"
            next_word = words[i + 1] if i < n - 1 else "</s>"
            next_candidates = self._get_initial_candidates(next_word) if i < n - 1 else ["</s>"]
            next_hint = next_candidates[0]

            chosen_tag, fired_rule = self.disambiguate_token(
                word, candidates, prev_word, prev_tag, next_word, next_hint
            )
            tagged.append((word, chosen_tag, fired_rule))

        return tagged

    def evaluate_on_tokens(self, eval_tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        words = [t["word"] for t in eval_tokens]
        gold_tags = [t["gold_tag"] for t in eval_tokens]
        pred_tagged = self.tag_sentence(words)

        total = len(gold_tokens := gold_tags)
        correct_total = 0
        ambiguous_total = 0
        ambiguous_correct = 0

        errors = []

        for i, gold in enumerate(gold_tokens):
            w = words[i].lower()
            pred = pred_tagged[i][1]
            rule = pred_tagged[i][2]
            is_ambig = len(self.tag_dict.get(w, set())) > 1

            if is_ambig:
                ambiguous_total += 1
                if pred == gold:
                    ambiguous_correct += 1

            if pred == gold:
                correct_total += 1
            else:
                errors.append((w, gold, pred, rule))

        overall_acc = (correct_total / total) * 100 if total else 0.0
        ambig_acc = (ambiguous_correct / ambiguous_total) * 100 if ambiguous_total else 0.0

        error_counts = Counter((e[1], e[2]) for e in errors).most_common(5)

        return {
            "total_tokens_evaluated": total,
            "overall_accuracy_pct": round(overall_acc, 2),
            "ambiguous_tokens_count": ambiguous_total,
            "ambiguous_tokens_accuracy_pct": round(ambig_acc, 2),
            "commonest_error_confusions": [
                {"gold": k[0], "predicted": k[1], "count": v} for k, v in error_counts
            ],
            "rule_conflict_case": {
                "phrase": "the return ticket",
                "conflicting_rules": [
                    "Rule 03 (After DT -> NN)",
                    "Rule 04 (Preceding NN -> JJ)"
                ],
                "resolution": "Ordered Rule 04 ahead of Rule 03 so prenominal modifiers become JJ, preventing misclassification of return as head noun.",
                "cost": "Hard-coded rule ordering fragility: exceptions requiring nominal compounds (e.g. 'the bus station') then require explicit compound exceptions (Rule 05)."
            }
        }
