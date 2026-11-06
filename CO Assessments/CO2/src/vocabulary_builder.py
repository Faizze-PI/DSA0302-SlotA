"""
Module: vocabulary_builder.py
Vocabulary Construction, Word Counting, and Lexical Dynamics
DSA03 - Natural Language Processing (Assignment 2 - Task 1)
Author: Transport Speech Analytics & Language Engineering Team
Computes types, tokens, lemmas, and word forms across transcribed utterances.
Implements tokenization policies, evaluates Zipf's law, and measures OOV rates.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", re.IGNORECASE)

def tokenize_utterance(text: str, preserve_case: bool = False, keep_time_intact: bool = True) -> List[str]:
    raw = text.strip()
    if not preserve_case:
        raw = raw.lower()

    if keep_time_intact:
        time_tokens = {}
        for idx, match in enumerate(TIME_PATTERN.finditer(raw)):
            placeholder = f"__TIME_{idx}__"
            time_tokens[placeholder] = match.group(0).replace(" ", "_")
            raw = raw.replace(match.group(0), placeholder)
        
        words = re.findall(r"\b[A-Za-z0-9_:]+\b", raw)
        tokens = [time_tokens.get(w, w) for w in words]
    else:
        tokens = re.findall(r"\b[A-Za-z0-9]+\b|[:]", raw)

    return ["<s>"] + tokens + ["</s>"]

class VocabularyBuilder:
    def __init__(self, train_records: List[Dict[str, Any]]):
        self.train_records = train_records
        self.raw_tokens: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.word_counts: Counter = Counter()
        self._process_corpus()

    def _process_corpus(self):
        for rec in self.train_records:
            toks = tokenize_utterance(rec["utterance_text"], preserve_case=False, keep_time_intact=True)
            self.tokenized_corpus.append(toks)
            for t in toks:
                if t not in ("<s>", "</s>"):
                    self.raw_tokens.append(t)
        self.word_counts = Counter(self.raw_tokens)

    def compute_lexical_statistics(self) -> Dict[str, Any]:
        total_tokens = len(self.raw_tokens)
        types = set(self.raw_tokens)
        
        case_preserved_tokens = []
        for rec in self.train_records:
            toks = tokenize_utterance(rec["utterance_text"], preserve_case=True, keep_time_intact=True)
            case_preserved_tokens.extend([t for t in toks if t not in ("<s>", "</s>")])
        case_preserved_types = set(case_preserved_tokens)

        split_time_tokens = []
        for rec in self.train_records:
            toks = tokenize_utterance(rec["utterance_text"], preserve_case=False, keep_time_intact=False)
            split_time_tokens.extend([t for t in toks if t not in ("<s>", "</s>")])
        split_time_types = set(split_time_tokens)

        lemmas = set()
        for w in types:
            if w.endswith("ing") and len(w) > 5:
                lemmas.add(w[:-3])
            elif w.endswith("ed") and len(w) > 4:
                lemmas.add(w[:-2])
            elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
                lemmas.add(w[:-1])
            else:
                lemmas.add(w)

        return {
            "total_tokens": total_tokens,
            "unique_types_case_folded": len(types),
            "unique_types_case_preserved": len(case_preserved_types),
            "unique_types_split_time": len(split_time_types),
            "estimated_unique_lemmas": len(lemmas),
            "case_folding_reduction": len(case_preserved_types) - len(types),
            "time_tokenization_delta": len(split_time_types) - len(types)
        }

    def compute_zipf_distribution(self, top_n: int = 100) -> Dict[str, Any]:
        sorted_counts = self.word_counts.most_common()
        ranks = []
        frequencies = []
        words = []
        
        c = sorted_counts[0][1] if sorted_counts else 1
        deviations = []

        for rank, (word, count) in enumerate(sorted_counts[:top_n], start=1):
            ranks.append(rank)
            frequencies.append(count)
            words.append(word)
            
            theoretical_count = c / rank
            rel_diff = abs(count - theoretical_count) / theoretical_count
            deviations.append((word, rank, count, round(theoretical_count, 1), round(rel_diff * 100, 2)))

        deviations.sort(key=lambda x: x[4], reverse=True)

        return {
            "ranks": ranks,
            "frequencies": frequencies,
            "words": words,
            "constant_c": c,
            "top_deviations": deviations[:5]
        }

    def evaluate_unknown_thresholds(self, thresholds: List[int] = [1, 2, 3, 5]) -> Dict[int, Dict[str, Any]]:
        total_tokens = len(self.raw_tokens)
        results = {}

        for th in thresholds:
            vocab = {word for word, count in self.word_counts.items() if count >= th}
            vocab.add("<UNK>")
            
            unk_tokens = sum(count for word, count in self.word_counts.items() if count < th)
            prop = (unk_tokens / total_tokens) * 100 if total_tokens else 0.0

            results[th] = {
                "threshold": th,
                "vocab_size": len(vocab),
                "words_mapped_to_unk": sum(1 for _, count in self.word_counts.items() if count < th),
                "unk_tokens_count": unk_tokens,
                "unk_token_proportion_pct": round(prop, 4)
            }

        return results

    def measure_development_oov(self, dev_records: List[Dict[str, Any]], threshold: int = 2) -> Dict[str, Any]:
        train_vocab = {word for word, count in self.word_counts.items() if count >= threshold}
        
        dev_tokens = []
        for rec in dev_records:
            toks = tokenize_utterance(rec["utterance_text"], preserve_case=False, keep_time_intact=True)
            dev_tokens.extend([t for t in toks if t not in ("<s>", "</s>")])

        total_dev_tokens = len(dev_tokens)
        dev_types = set(dev_tokens)
        oov_tokens = sum(1 for t in dev_tokens if t not in train_vocab)
        oov_types = sum(1 for t in dev_types if t not in train_vocab)

        return {
            "threshold_used": threshold,
            "train_vocab_size": len(train_vocab),
            "total_dev_tokens": total_dev_tokens,
            "dev_unique_types": len(dev_types),
            "oov_token_count": oov_tokens,
            "oov_token_rate_pct": round((oov_tokens / total_dev_tokens) * 100, 4) if total_dev_tokens else 0.0,
            "oov_type_rate_pct": round((oov_types / len(dev_types)) * 100, 4) if dev_types else 0.0
        }
