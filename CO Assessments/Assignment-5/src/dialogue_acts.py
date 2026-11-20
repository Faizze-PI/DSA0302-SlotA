"""Dialogue act classification using scikit-learn, Cohen's kappa, and confusion matrix."""

import csv
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split


class DialogueActClassifier:
    def __init__(self, dialogues_csv: str):
        self.rows: List[Dict[str, str]] = []
        with open(dialogues_csv, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def compute_kappa(self) -> Tuple[float, float]:
        a1, a2 = [], []
        for r in self.rows:
            if r["Ann2_Act"]:
                a1.append(r["Gold_Act"])
                a2.append(r["Ann2_Act"])
        raw = sum(1 for x, y in zip(a1, a2) if x == y) / len(a1)
        kappa = float(cohen_kappa_score(a1, a2))
        return raw, kappa

    def train_and_evaluate(self) -> Tuple[float, List[List[int]]]:
        X_text = [f"{r['Utterance']} POS_{r['Turn_Pos']} PREV_{r['Prev_Act']}" for r in self.rows]
        y = [r["Gold_Act"] for r in self.rows]
        X_tr, X_te, y_tr, y_te = train_test_split(X_text, y, test_size=0.25, random_state=42)
        vec = CountVectorizer()
        X_tr_v = vec.fit_transform(X_tr)
        X_te_v = vec.transform(X_te)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr_v, y_tr)
        preds = clf.predict(X_te_v)
        acc = float(accuracy_score(y_te, preds))
        cm = confusion_matrix(y_te, preds).tolist()
        return acc, cm
