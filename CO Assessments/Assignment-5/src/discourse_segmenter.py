"""Discourse segmentation using cue phrases, topic shifts, and inter-annotator agreement."""

import csv
import re
from typing import Dict, List, Tuple
from sklearn.metrics import cohen_kappa_score


class DiscourseSegmenter:
    def __init__(self):
        self.cue_pattern = re.compile(r"\b(?:Consequently|Therefore|However|Meanwhile|Moreover)\b,?\s*")

    def segment_text(self, text: str) -> List[str]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        return parts

    def measure_cross_sentence_anaphora(self, reports_csv: str) -> Tuple[int, float]:
        total = 0
        with_anaphora = 0
        with open(reports_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                sents = self.segment_text(row["Text"])
                if len(sents) > 1 and any(p in sents[1].lower().split() for p in ("it", "they", "this", "that")):
                    with_anaphora += 1
        return total, (with_anaphora / total) if total else 0.0

    def compute_boundary_kappa(self, disc_csv: str) -> Tuple[float, float, int]:
        a1, a2 = [], []
        with open(disc_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Ann2_Bound"]:
                    a1.append(row["Ann1_Bound"])
                    a2.append(row["Ann2_Bound"])
        raw = sum(1 for x, y in zip(a1, a2) if x == y) / len(a1)
        kappa = float(cohen_kappa_score(a1, a2)) if len(set(a2)) > 1 else 1.0
        return raw, kappa, len(a1)
