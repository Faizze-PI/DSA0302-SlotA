"""IBM Model 1 Expectation-Maximisation word alignment and statistical translation evaluation."""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class IBMModel1Translator:
    def __init__(self, parallel_csv: str, target_lang: str = "Tamil"):
        self.pairs: List[Tuple[List[str], List[str]]] = []
        self.test_pairs: List[Tuple[List[str], List[str]]] = []
        with open(parallel_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row["English"].lower().strip(".").split()
                tgt = row[target_lang].strip(".").split()
                if row["Split"] == "train":
                    self.pairs.append((src, tgt))
                else:
                    self.test_pairs.append((src, tgt))
        self.t_probs: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.25)

    def train_em(self, iterations: int = 3):
        for _ in range(iterations):
            count: Dict[Tuple[str, str], float] = defaultdict(float)
            total: Dict[str, float] = defaultdict(float)
            for src, tgt in self.pairs[:300]:
                for e in tgt:
                    s_total = sum(self.t_probs[(e, f)] for f in src) or 1e-9
                    for f in src:
                        delta = self.t_probs[(e, f)] / s_total
                        count[(e, f)] += delta
                        total[f] += delta
            for (e, f), c in count.items():
                self.t_probs[(e, f)] = c / total[f]

    def translate(self, src_tokens: List[str]) -> List[str]:
        out = []
        for f in src_tokens:
            cands = [(p, e) for (e, src_w), p in self.t_probs.items() if src_w == f]
            if cands:
                best_e = max(cands)[1]
                out.append(best_e)
            else:
                out.append(f)
        return out

    def evaluate_bleu(self) -> float:
        smooth = SmoothingFunction().method1
        scores = []
        for src, ref in self.test_pairs[:50]:
            hyp = self.translate(src)
            scores.append(sentence_bleu([ref], hyp, smoothing_function=smooth))
        return sum(scores) / len(scores) if scores else 0.0
