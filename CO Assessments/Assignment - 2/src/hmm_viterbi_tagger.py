"""
Module: hmm_viterbi_tagger.py
Hidden Markov Model Stochastic Tagger and Viterbi Decoding
DSA03 - Natural Language Processing (Assignment 2 - Task 8)
Author: Transport Speech Analytics & Language Engineering Team
Estimates transition and emission distributions, decodes optimal tag sequences via Viterbi trellis,
compares complexity against exhaustive search, and handles unknown tokens via morphological priors.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class HMMTagger:
    def __init__(self, tagged_corpus: List[List[Tuple[str, str]]], all_tags: List[str]):
        self.corpus = tagged_corpus
        self.tags = sorted(list(set(all_tags)))
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.idx_to_tag = {i: tag for i, tag in enumerate(self.tags)}
        self.num_tags = len(self.tags)

        self.transition_counts: Counter = Counter()
        self.tag_counts: Counter = Counter()
        self.emission_counts: Counter = Counter()
        self.vocab: Set[str] = set()
        self.word_counts: Counter = Counter()

        self.open_class_tags = {"NN", "NNS", "N_LOC", "N_PROP", "VB", "VBD", "VBG", "VBZ", "VBP", "JJ", "RB"}
        self._train()

    def _train(self):
        for sent in self.corpus:
            prev_tag = "<s>"
            self.tag_counts["<s>"] += 1
            for word, tag in sent:
                w_lower = word.lower()
                self.vocab.add(w_lower)
                self.word_counts[w_lower] += 1
                self.emission_counts[(tag, w_lower)] += 1
                self.transition_counts[(prev_tag, tag)] += 1
                self.tag_counts[tag] += 1
                prev_tag = tag
            self.transition_counts[(prev_tag, "</s>")] += 1

    def transition_prob(self, t1: str, t2: str, alpha: float = 0.01) -> float:
        c_pair = self.transition_counts.get((t1, t2), 0)
        c_t1 = self.tag_counts.get(t1, 0)
        denom = c_t1 + (alpha * self.num_tags)
        return (c_pair + alpha) / denom if denom > 0 else 1.0 / self.num_tags

    def emission_prob(self, tag: str, word: str, alpha: float = 0.001) -> float:
        w_lower = word.lower()
        if w_lower in self.vocab:
            c_tw = self.emission_counts.get((tag, w_lower), 0)
            c_t = self.tag_counts.get(tag, 0)
            denom = c_t + (alpha * len(self.vocab))
            return (c_tw + alpha) / denom if denom > 0 else 1e-6
        else:
            return self._unknown_word_emission(tag, word)

    def _unknown_word_emission(self, tag: str, word: str) -> float:
        w_lower = word.lower()
        if word.isdigit() and tag == "CD":
            return 0.85
        if ":" in word and tag == "N_TIME":
            return 0.90
        if word[0].isupper():
            if tag in ("N_LOC", "N_PROP"):
                return 0.40
            if tag == "NN":
                return 0.15
        if w_lower.endswith("ing") and tag in ("VBG", "NN"):
            return 0.50
        if w_lower.endswith("ed") and tag in ("VBD", "JJ"):
            return 0.50
        if w_lower.endswith("ly") and tag == "RB":
            return 0.70
        if w_lower.endswith("tion") and tag == "NN":
            return 0.75

        if tag in self.open_class_tags:
            c_t = self.tag_counts.get(tag, 1)
            total_open = sum(self.tag_counts.get(t, 1) for t in self.open_class_tags)
            return (c_t / total_open) * 0.1
        
        return 1e-6

    def viterbi_decode(self, words: List[str]) -> Tuple[List[str], float, List[Dict[str, Any]]]:
        t_len = len(words)
        if t_len == 0:
            return [], 0.0, []

        viterbi = [{} for _ in range(t_len)]
        backpointer = [{} for _ in range(t_len)]

        w0 = words[0]
        for tag in self.tags:
            p_trans = self.transition_prob("<s>", tag)
            p_emiss = self.emission_prob(tag, w0)
            viterbi[0][tag] = math.log2(max(p_trans * p_emiss, 1e-30))
            backpointer[0][tag] = None

        for t in range(1, t_len):
            wt = words[t]
            for curr_tag in self.tags:
                p_emiss = self.emission_prob(curr_tag, wt)
                best_score = -float("inf")
                best_prev = None

                for prev_tag in self.tags:
                    p_trans = self.transition_prob(prev_tag, curr_tag)
                    score = viterbi[t - 1][prev_tag] + math.log2(max(p_trans * p_emiss, 1e-30))
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag

                viterbi[t][curr_tag] = best_score
                backpointer[t][curr_tag] = best_prev

        best_last_tag = max(self.tags, key=lambda tag: viterbi[t_len - 1][tag])
        best_log_prob = viterbi[t_len - 1][best_last_tag]

        best_path = [best_last_tag]
        for t in range(t_len - 1, 0, -1):
            prev = backpointer[t][best_path[-1]]
            best_path.append(prev)
        best_path.reverse()

        trellis_trace = []
        for t in range(t_len):
            col = {"step": t, "word": words[t], "scores": {}}
            for tag in ["NN", "VB", "JJ", "N_LOC", "CD"]:
                if tag in viterbi[t]:
                    col["scores"][tag] = round(viterbi[t][tag], 3)
            trellis_trace.append(col)

        return best_path, round(best_log_prob, 4), trellis_trace

    def most_frequent_tag_baseline(self, words: List[str]) -> List[str]:
        word_to_tag = defaultdict(Counter)
        for (tag, w), cnt in self.emission_counts.items():
            word_to_tag[w][tag] += cnt

        preds = []
        for w in words:
            w_lower = w.lower()
            if w_lower in word_to_tag and word_to_tag[w_lower]:
                preds.append(word_to_tag[w_lower].most_common(1)[0][0])
            else:
                preds.append("NN")
        return preds

    def compare_trellis_complexity(self, sentence_len: int = 5) -> Dict[str, Any]:
        s = self.num_tags
        t = sentence_len
        viterbi_ops = t * s * s
        exhaustive_ops = (s ** t) * t

        return {
            "sentence_length": t,
            "tagset_size": s,
            "viterbi_operations_O_T_S2": viterbi_ops,
            "exhaustive_search_operations_O_T_ST": exhaustive_ops,
            "speedup_factor": round(exhaustive_ops / viterbi_ops, 1),
            "explanation": "Dynamic programming collapses path histories at each time step, reducing exponential search complexity to quadratic execution."
        }
