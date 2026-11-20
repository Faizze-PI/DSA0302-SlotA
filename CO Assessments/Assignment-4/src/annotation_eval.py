"""Inter-annotator agreement (Cohen's kappa) and gold-standard evaluation diagnostics."""

import csv
from typing import Dict, Tuple
from sklearn.metrics import cohen_kappa_score


class AnnotationEvaluator:
    def compute_sense_kappa(self, annotations_csv: str) -> Tuple[float, float, int]:
        a1, a2 = [], []
        with open(annotations_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Annotator_2"]:
                    a1.append(row["Annotator_1"])
                    a2.append(row["Annotator_2"])
        raw = sum(1 for x, y in zip(a1, a2) if x == y) / len(a1)
        kappa = float(cohen_kappa_score(a1, a2))
        return raw, kappa, len(a1)

    def compute_relevance_kappa(self, relevance_csv: str) -> Tuple[float, float, int]:
        a1, a2 = [], []
        with open(relevance_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Annotator_2_Rel"]:
                    a1.append(int(row["Annotator_1_Rel"]))
                    a2.append(int(row["Annotator_2_Rel"]))
        raw = sum(1 for x, y in zip(a1, a2) if x == y) / len(a1)
        kappa = float(cohen_kappa_score(a1, a2)) if len(set(a2)) > 1 else 1.0
        return raw, kappa, len(a1)
