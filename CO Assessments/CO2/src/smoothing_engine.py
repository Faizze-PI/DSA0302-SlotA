"""
Module: smoothing_engine.py
Probability Mass Redistribution, Smoothing, and Discounting
DSA03 - Natural Language Processing (Assignment 2 - Task 3)
Author: Transport Speech Analytics & Language Engineering Team
Implements Add-1 (Laplace), Add-k, and Good-Turing discounting techniques.
Computes frequency-of-frequencies tables, discounted counts, and validation perplexities.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class SmoothingEngine:
    def __init__(self, tokenized_sentences: List[List[str]], vocab: Set[str]):
        self.sentences = tokenized_sentences
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.bigram_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.total_bigrams: int = 0
        self.nr_table: Counter = Counter()
        self._build_counts()
        self._build_good_turing()

    def _build_counts(self):
        for sent in self.sentences:
            cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
            for i in range(len(cleaned) - 1):
                w1, w2 = cleaned[i], cleaned[i+1]
                self.bigram_counts[(w1, w2)] += 1
                self.unigram_counts[w1] += 1
                self.total_bigrams += 1

    def _build_good_turing(self):
        for bg, cnt in self.bigram_counts.items():
            self.nr_table[cnt] += 1

        v2 = self.vocab_size * self.vocab_size
        n0 = v2 - len(self.bigram_counts)
        self.nr_table[0] = max(n0, 1)

    def laplace_prob(self, w1: str, w2: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        count = self.bigram_counts.get((c1, c2), 0)
        denom = self.unigram_counts.get(c1, 0) + self.vocab_size
        return (count + 1) / denom

    def add_k_prob(self, w1: str, w2: str, k: float = 0.05) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        count = self.bigram_counts.get((c1, c2), 0)
        denom = self.unigram_counts.get(c1, 0) + (k * self.vocab_size)
        return (count + k) / denom

    def good_turing_effective_count(self, r: int, k_threshold: int = 5) -> float:
        if r > k_threshold:
            return float(r)
        
        nr = self.nr_table.get(r, 0)
        nr1 = self.nr_table.get(r + 1, 0)
        if nr == 0:
            return 0.0
        if nr1 == 0:
            return float(r)
        
        c_star = (r + 1) * (nr1 / nr)
        return c_star

    def good_turing_prob(self, w1: str, w2: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        r = self.bigram_counts.get((c1, c2), 0)

        if r == 0:
            n1 = self.nr_table.get(1, 1)
            n0 = self.nr_table.get(0, 1)
            p0 = n1 / (self.total_bigrams * n0)
            return p0
        else:
            c_star = self.good_turing_effective_count(r)
            return c_star / self.total_bigrams

    def tabulate_context_smoothing(self, context_word: str, k: float = 0.05) -> Dict[str, Any]:
        c1 = context_word if context_word in self.vocab else "<UNK>"
        context_count = self.unigram_counts.get(c1, 0)
        
        continuations = [(bg[1], cnt) for bg, cnt in self.bigram_counts.items() if bg[0] == c1]
        continuations.sort(key=lambda x: x[1], reverse=True)

        rows = []
        mass_seen_mle = 0.0
        mass_seen_laplace = 0.0
        mass_seen_add_k = 0.0

        for w2, count in continuations[:6]:
            p_mle = count / context_count if context_count > 0 else 0.0
            p_lap = (count + 1) / (context_count + self.vocab_size)
            p_addk = (count + k) / (context_count + k * self.vocab_size)

            c_star_lap = ((count + 1) * context_count) / (context_count + self.vocab_size)
            c_star_addk = ((count + k) * context_count) / (context_count + k * self.vocab_size)

            mass_seen_mle += p_mle
            mass_seen_laplace += p_lap
            mass_seen_add_k += p_addk

            rows.append({
                "word": w2,
                "count": count,
                "p_mle": round(p_mle, 5),
                "c_star_laplace": round(c_star_lap, 3),
                "p_laplace": round(p_lap, 5),
                "c_star_add_k": round(c_star_addk, 3),
                "p_add_k": round(p_addk, 5)
            })

        unseen_count = self.vocab_size - len(continuations)
        p_unseen_laplace = 1.0 / (context_count + self.vocab_size)
        p_unseen_add_k = k / (context_count + k * self.vocab_size)

        return {
            "context_word": c1,
            "context_count": context_count,
            "vocab_size": self.vocab_size,
            "seen_continuations_count": len(continuations),
            "unseen_continuations_count": unseen_count,
            "mass_shaved_laplace": round(1.0 - (p_lap * len(continuations)), 4),
            "p_unseen_laplace": round(p_unseen_laplace, 6),
            "p_unseen_add_k": round(p_unseen_add_k, 6),
            "sample_rows": rows
        }

    def evaluate_perplexity_on_dev(self, dev_sentences: List[List[str]]) -> Dict[str, float]:
        models = {
            "Laplace (Add-1)": lambda w1, w2: self.laplace_prob(w1, w2),
            "Add-0.5": lambda w1, w2: self.add_k_prob(w1, w2, k=0.5),
            "Add-0.1": lambda w1, w2: self.add_k_prob(w1, w2, k=0.1),
            "Add-0.05": lambda w1, w2: self.add_k_prob(w1, w2, k=0.05),
            "Good-Turing": lambda w1, w2: self.good_turing_prob(w1, w2)
        }

        results = {}
        for name, prob_fn in models.items():
            total_log_prob = 0.0
            total_words = 0

            for sent in dev_sentences:
                cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
                for i in range(len(cleaned) - 1):
                    w1, w2 = cleaned[i], cleaned[i+1]
                    p = prob_fn(w1, w2)
                    p = max(p, 1e-12)
                    total_log_prob += math.log2(p)
                    total_words += 1

            cross_entropy = - (total_log_prob / total_words) if total_words > 0 else float("inf")
            perplexity = 2 ** cross_entropy if cross_entropy < 300 else float("inf")
            results[name] = round(perplexity, 2)

        return results
