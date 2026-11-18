"""
Module: unsmoothed_ngram.py
Maximum Likelihood Estimators and Unsmoothed N-gram Language Models
DSA03 - Natural Language Processing (Assignment 2 - Task 2)
Author: Transport Speech Analytics & Language Engineering Team
Builds unigram, bigram, and trigram MLE language models from corpus frequencies.
Computes utterance probabilities, quantifies matrix sparsity, and samples text.
"""

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

class UnsmoothedNgram:
    def __init__(self, tokenized_sentences: List[List[str]], vocab: Set[str] = None):
        self.sentences = tokenized_sentences
        self.vocab = vocab or set()
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()
        self.bigram_context_counts: Counter = Counter()
        self.trigram_context_counts: Counter = Counter()
        self.total_tokens: int = 0
        self._train()

    def _train(self):
        for sent in self.sentences:
            cleaned = [w if (not self.vocab or w in self.vocab or w in ("<s>", "</s>")) else "<UNK>" for w in sent]
            
            for w in cleaned:
                self.unigram_counts[w] += 1
                if w != "<s>":
                    self.total_tokens += 1
            
            for i in range(len(cleaned) - 1):
                bg = (cleaned[i], cleaned[i+1])
                self.bigram_counts[bg] += 1
                self.bigram_context_counts[cleaned[i]] += 1
            
            for i in range(len(cleaned) - 2):
                tg = (cleaned[i], cleaned[i+1], cleaned[i+2])
                self.trigram_counts[tg] += 1
                self.trigram_context_counts[(cleaned[i], cleaned[i+1])] += 1

        if not self.vocab:
            self.vocab = set(self.unigram_counts.keys())

    def unigram_prob(self, word: str) -> float:
        w = word if word in self.vocab else "<UNK>"
        return self.unigram_counts[w] / self.total_tokens if self.total_tokens > 0 else 0.0

    def bigram_prob(self, w1: str, w2: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        denom = self.bigram_context_counts[c1]
        return self.bigram_counts[(c1, c2)] / denom if denom > 0 else 0.0

    def trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        c1 = w1 if w1 in self.vocab else "<UNK>"
        c2 = w2 if w2 in self.vocab else "<UNK>"
        c3 = w3 if w3 in self.vocab else "<UNK>"
        denom = self.trigram_context_counts[(c1, c2)]
        return self.trigram_counts[(c1, c2, c3)] / denom if denom > 0 else 0.0

    def score_sentence(self, tokens: List[str], order: int = 2) -> Dict[str, Any]:
        if tokens[0] != "<s>":
            tokens = ["<s>"] + tokens
        if tokens[-1] != "</s>":
            tokens = tokens + ["</s>"]

        factors = []
        prob = 1.0
        log_prob = 0.0
        zero_cause = None

        if order == 1:
            for w in tokens[1:]:
                p = self.unigram_prob(w)
                factors.append((w, p))
                prob *= p
                if p > 0:
                    log_prob += math.log2(p)
                else:
                    zero_cause = f"Unseen unigram: {w}"
        elif order == 2:
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                p = self.bigram_prob(w1, w2)
                factors.append(((w1, w2), p))
                prob *= p
                if p > 0:
                    log_prob += math.log2(p)
                else:
                    if not zero_cause:
                        zero_cause = f"Unseen bigram: ({w1}, {w2})"
        elif order == 3:
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                p = self.trigram_prob(w1, w2, w3)
                factors.append(((w1, w2, w3), p))
                prob *= p
                if p > 0:
                    log_prob += math.log2(p)
                else:
                    if not zero_cause:
                        zero_cause = f"Unseen trigram: ({w1}, {w2}, {w3})"

        return {
            "order": order,
            "sentence": " ".join(tokens),
            "factors": factors,
            "probability": prob,
            "log2_probability": log_prob if prob > 0 else -float("inf"),
            "zero_probability": prob == 0.0,
            "zero_culprit": zero_cause
        }

    def compute_sparsity_statistics(self, dev_sentences: List[List[str]]) -> Dict[str, Any]:
        v_size = len(self.vocab)
        possible_bigrams = v_size * v_size
        possible_trigrams = v_size * v_size * v_size

        observed_bigrams = len(self.bigram_counts)
        observed_trigrams = len(self.trigram_counts)

        dev_bigrams = set()
        dev_trigrams = set()
        for sent in dev_sentences:
            cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
            for i in range(len(cleaned) - 1):
                dev_bigrams.add((cleaned[i], cleaned[i+1]))
            for i in range(len(cleaned) - 2):
                dev_trigrams.add((cleaned[i], cleaned[i+1], cleaned[i+2]))

        unseen_dev_bigrams = sum(1 for bg in dev_bigrams if bg not in self.bigram_counts)
        unseen_dev_trigrams = sum(1 for tg in dev_trigrams if tg not in self.trigram_counts)

        return {
            "vocab_size": v_size,
            "possible_bigrams": possible_bigrams,
            "observed_bigrams": observed_bigrams,
            "bigram_sparsity_pct": round((1.0 - (observed_bigrams / possible_bigrams)) * 100, 6),
            "possible_trigrams": possible_trigrams,
            "observed_trigrams": observed_trigrams,
            "trigram_sparsity_pct": round((1.0 - (observed_trigrams / possible_trigrams)) * 100, 8),
            "dev_unique_bigrams": len(dev_bigrams),
            "dev_unseen_bigrams_pct": round((unseen_dev_bigrams / len(dev_bigrams)) * 100, 4) if dev_bigrams else 0.0,
            "dev_unique_trigrams": len(dev_trigrams),
            "dev_unseen_trigrams_pct": round((unseen_dev_trigrams / len(dev_trigrams)) * 100, 4) if dev_trigrams else 0.0
        }

    def generate_utterances(self, order: int = 2, n_samples: int = 10, max_len: int = 20, seed: int = 42) -> List[str]:
        random.seed(seed)
        generated = []

        for _ in range(n_samples):
            tokens = ["<s>"]
            
            if order == 1:
                words = [w for w in self.unigram_counts.keys() if w not in ("<s>", "<UNK>")]
                weights = [self.unigram_counts[w] for w in words]
                for _ in range(max_len):
                    nxt = random.choices(words, weights=weights, k=1)[0]
                    tokens.append(nxt)
                    if nxt == "</s>":
                        break
            elif order == 2:
                for _ in range(max_len):
                    cur = tokens[-1]
                    candidates = [(bg[1], cnt) for bg, cnt in self.bigram_counts.items() if bg[0] == cur and bg[1] != "<UNK>"]
                    if not candidates:
                        tokens.append("</s>")
                        break
                    words, weights = zip(*candidates)
                    nxt = random.choices(words, weights=weights, k=1)[0]
                    tokens.append(nxt)
                    if nxt == "</s>":
                        break
            elif order == 3:
                tokens.append("please" if ("<s>", "please") in self.bigram_counts else list(self.vocab)[0])
                for _ in range(max_len):
                    ctx = (tokens[-2], tokens[-1])
                    candidates = [(tg[2], cnt) for tg, cnt in self.trigram_counts.items() if (tg[0], tg[1]) == ctx and tg[2] != "<UNK>"]
                    if not candidates:
                        tokens.append("</s>")
                        break
                    words, weights = zip(*candidates)
                    nxt = random.choices(words, weights=weights, k=1)[0]
                    tokens.append(nxt)
                    if nxt == "</s>":
                        break

            generated.append(" ".join([t for t in tokens if t not in ("<s>", "</s>")]))

        return generated
