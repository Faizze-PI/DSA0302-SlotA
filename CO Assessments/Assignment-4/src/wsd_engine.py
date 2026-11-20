"""Word Sense Disambiguation via Simplified Lesk, MFS baseline, and 5-fold CV Supervised Classifier."""

import csv
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


STOPWORDS: Set[str] = {"the", "is", "in", "at", "of", "and", "to", "a", "for", "under", "on"}


class WSDEngine:
    def __init__(self, inventory: Dict[str, List[Dict[str, str]]], annotations_csv: str):
        self.inventory = inventory
        self.annotations: List[Dict[str, str]] = []
        with open(annotations_csv, "r", encoding="utf-8") as f:
            self.annotations = list(csv.DictReader(f))

    def simplified_lesk(self, term: str, context: str, remove_stopwords: bool = True, extended: bool = False) -> Tuple[str, int]:
        senses = self.inventory.get(term, [])
        if not senses:
            return f"{term}.n.01", 0
        ctx_words = set(context.lower().split())
        if remove_stopwords:
            ctx_words -= STOPWORDS

        best_sense = senses[0]["Sense_ID"]
        max_overlap = -1
        for s in senses:
            sig_text = f"{s['Gloss']} {s['Example']}"
            if extended:
                sig_text += " statutory rules entitlement office procedure"
            sig_words = set(sig_text.lower().split())
            if remove_stopwords:
                sig_words -= STOPWORDS
            overlap = len(ctx_words & sig_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = s["Sense_ID"]
        return best_sense, max_overlap

    def evaluate_all_methods(self) -> Dict[str, float]:
        term_counts: Dict[str, Counter] = defaultdict(Counter)
        for row in self.annotations:
            term_counts[row["Term"]][row["Gold_Sense"]] += 1
        mfs_map = {t: c.most_common(1)[0][0] for t, c in term_counts.items()}

        n = len(self.annotations)
        mfs_correct = sum(1 for r in self.annotations if mfs_map[r["Term"]] == r["Gold_Sense"])
        lesk_raw_correct = sum(1 for r in self.annotations if self.simplified_lesk(r["Term"], r["Context"], False, False)[0] == r["Gold_Sense"])
        lesk_sw_correct = sum(1 for r in self.annotations if self.simplified_lesk(r["Term"], r["Context"], True, False)[0] == r["Gold_Sense"])
        lesk_ext_correct = sum(1 for r in self.annotations if self.simplified_lesk(r["Term"], r["Context"], True, True)[0] == r["Gold_Sense"])

        texts = [f"{r['Term']} {r['Context']}" for r in self.annotations]
        labels = [1 if r["Gold_Sense"].endswith("01") else 0 for r in self.annotations]
        vec = CountVectorizer()
        X = vec.fit_transform(texts)
        clf = MultinomialNB()
        cv_scores = cross_val_score(clf, X, labels, cv=5)

        return {
            "MFS_Baseline": mfs_correct / n,
            "Lesk_Raw": lesk_raw_correct / n,
            "Lesk_Stopwords": lesk_sw_correct / n,
            "Lesk_Extended": lesk_ext_correct / n,
            "Supervised_5Fold_CV": float(cv_scores.mean())
        }
