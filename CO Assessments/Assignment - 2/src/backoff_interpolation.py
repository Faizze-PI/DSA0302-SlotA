"""
Module: backoff_interpolation.py
Katz Backoff, Stupid Backoff, and Deleted Interpolation
DSA03 - Natural Language Processing (Assignment 2 - Task 4)
Author: Transport Speech Analytics & Language Engineering Team
Combines higher and lower order N-gram contexts via backoff cascades and mixture models.
Estimates deleted interpolation lambdas across context-count frequency bands.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class BackoffInterpolation:
    def __init__(self, tokenized_sentences: List[List[str]], vocab: Set[str]):
        self.sentences = tokenized_sentences
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()
        self.bigram_context_counts: Counter = Counter()
        self.trigram_context_counts: Counter = Counter()
        self.total_tokens: int = 0
        self._train()
        self.lambdas: Tuple[float, float, float] = (0.1, 0.3, 0.6)
        self.band_lambdas: Dict[str, Tuple[float, float, float]] = {}

    def _train(self):
        for sent in self.sentences:
            cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
            for w in cleaned:
                self.unigram_counts[w] += 1
                if w != "<s>":
                    self.total_tokens += 1
            for i in range(len(cleaned) - 1):
                w1, w2 = cleaned[i], cleaned[i+1]
                self.bigram_counts[(w1, w2)] += 1
                self.bigram_context_counts[w1] += 1
            for i in range(len(cleaned) - 2):
                w1, w2, w3 = cleaned[i], cleaned[i+1], cleaned[i+2]
                self.trigram_counts[(w1, w2, w3)] += 1
                self.trigram_context_counts[(w1, w2)] += 1

    def mle_unigram(self, w: str) -> float:
        cw = w if w in self.vocab else "<UNK>"
        return self.unigram_counts.get(cw, 0) / self.total_tokens if self.total_tokens > 0 else 0.0

    def mle_bigram(self, w1: str, w2: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        denom = self.bigram_context_counts.get(c1, 0)
        return self.bigram_counts.get((c1, c2), 0) / denom if denom > 0 else 0.0

    def mle_trigram(self, w1: str, w2: str, w3: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        c3 = w3 if w3 in self.vocab else "<UNK>"
        denom = self.trigram_context_counts.get((c1, c2), 0)
        return self.trigram_counts.get((c1, c2, c3), 0) / denom if denom > 0 else 0.0

    def stupid_backoff_score(self, w1: str, w2: str, w3: str, alpha: float = 0.4) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        c3 = w3 if w3 in self.vocab else "<UNK>"

        if self.trigram_counts.get((c1, c2, c3), 0) > 0:
            denom = self.trigram_context_counts.get((c1, c2), 1)
            return self.trigram_counts[(c1, c2, c3)] / denom
        elif self.bigram_counts.get((c2, c3), 0) > 0:
            denom = self.bigram_context_counts.get(c2, 1)
            return alpha * (self.bigram_counts[(c2, c3)] / denom)
        else:
            return alpha * alpha * self.mle_unigram(c3)

    def katz_backoff_prob(self, w1: str, w2: str, w3: str, discount: float = 0.75) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        c3 = w3 if w3 in self.vocab else "<UNK>"

        trig_cnt = self.trigram_counts.get((c1, c2, c3), 0)
        trig_ctx = self.trigram_context_counts.get((c1, c2), 0)

        if trig_cnt > 0 and trig_ctx > 0:
            return (trig_cnt - discount) / trig_ctx

        big_cnt = self.bigram_counts.get((c2, c3), 0)
        big_ctx = self.bigram_context_counts.get(c2, 0)
        
        seen_trig_words = [tg[2] for tg in self.trigram_counts.keys() if tg[0] == c1 and tg[1] == c2]
        mass_reserved_trig = (discount * len(seen_trig_words)) / trig_ctx if trig_ctx > 0 else 1.0

        if big_cnt > 0 and big_ctx > 0:
            p_back = (big_cnt - discount) / big_ctx
        else:
            p_back = self.mle_unigram(c3)

        return mass_reserved_trig * p_back

    def linear_interpolation_prob(self, w1: str, w2: str, w3: str, lambdas: Tuple[float, float, float] = None) -> float:
        l1, l2, l3 = lambdas or self.lambdas
        p_tri = self.mle_trigram(w1, w2, w3)
        p_bi = self.mle_bigram(w2, w3)
        p_uni = self.mle_unigram(w3)
        return (l3 * p_tri) + (l2 * p_bi) + (l1 * p_uni)

    def estimate_deleted_interpolation_weights(self, heldout_sentences: List[List[str]]) -> Dict[str, Any]:
        counts = {"low": [0.0, 0.0, 0.0], "mid": [0.0, 0.0, 0.0], "high": [0.0, 0.0, 0.0]}
        
        for sent in heldout_sentences:
            cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
            for i in range(len(cleaned) - 2):
                w1, w2, w3 = cleaned[i], cleaned[i+1], cleaned[i+2]
                ctx_cnt = self.trigram_context_counts.get((w1, w2), 0)
                
                if ctx_cnt == 0:
                    band = "low"
                elif 1 <= ctx_cnt <= 10:
                    band = "mid"
                else:
                    band = "high"

                p3 = self.mle_trigram(w1, w2, w3)
                p2 = self.mle_bigram(w2, w3)
                p1 = self.mle_unigram(w3)

                max_p = max(p1, p2, p3)
                if max_p == p3 and p3 > 0:
                    counts[band][2] += 1.0
                elif max_p == p2 and p2 > 0:
                    counts[band][1] += 1.0
                else:
                    counts[band][0] += 1.0

        for band, vec in counts.items():
            tot = sum(vec)
            if tot > 0:
                self.band_lambdas[band] = (round(vec[0]/tot, 3), round(vec[1]/tot, 3), round(vec[2]/tot, 3))
            else:
                self.band_lambdas[band] = (0.4, 0.4, 0.2)

        tot_all = [sum(counts[b][i] for b in counts) for i in range(3)]
        s_all = sum(tot_all)
        if s_all > 0:
            self.lambdas = (round(tot_all[0]/s_all, 3), round(tot_all[1]/s_all, 3), round(tot_all[2]/s_all, 3))

        return {
            "global_lambdas": self.lambdas,
            "band_lambdas": {
                "Count=0 (Zero context)": self.band_lambdas.get("low", (0.6, 0.35, 0.05)),
                "Count 1-10 (Sparse context)": self.band_lambdas.get("mid", (0.2, 0.45, 0.35)),
                "Count > 10 (Frequent context)": self.band_lambdas.get("high", (0.05, 0.25, 0.70))
            }
        }

    def trace_unseen_trigram_comparison(self, trigram: Tuple[str, str, str]) -> Dict[str, Any]:
        w1, w2, w3 = trigram
        c_tri = self.trigram_counts.get((w1, w2, w3), 0)
        c_ctx = self.trigram_context_counts.get((w1, w2), 0)
        c_bi = self.bigram_counts.get((w2, w3), 0)
        c_bictx = self.bigram_context_counts.get(w2, 0)
        c_uni = self.unigram_counts.get(w3, 0)

        p_katz = self.katz_backoff_prob(w1, w2, w3)
        p_stupid = self.stupid_backoff_score(w1, w2, w3)
        p_interp = self.linear_interpolation_prob(w1, w2, w3)

        return {
            "trigram": trigram,
            "counts": {
                f"C({w1}, {w2}, {w3})": c_tri,
                f"C({w1}, {w2})": c_ctx,
                f"C({w2}, {w3})": c_bi,
                f"C({w2})": c_bictx,
                f"C({w3})": c_uni
            },
            "katz_backoff_probability": round(p_katz, 6),
            "stupid_backoff_score": round(p_stupid, 6),
            "linear_interpolation_probability": round(p_interp, 6),
            "explanation": "Katz backoff strictly falls back to the bigram context only because trigram count is 0, whereas linear interpolation always incorporates unigram, bigram, and trigram terms weighted by lambda."
        }
