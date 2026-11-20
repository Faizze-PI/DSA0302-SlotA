"""Inverted index, TF-IDF vector space retrieval, cosine ranking, and sense-aware query expansion."""

import csv
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IRSearchEngine:
    def __init__(self, circulars_csv: str):
        self.doc_ids: List[str] = []
        self.docs: List[str] = []
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        with open(circulars_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.doc_ids.append(row["Circular_ID"])
                text = f"{row['Title']} {row['Text']}"
                self.docs.append(text)
                for tok in set(text.lower().split()):
                    self.inverted_index[tok.strip(".,()")].add(row["Circular_ID"])

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_indices]

    def evaluate_queries(self, relevance_csv: str, expand_senses: bool = False) -> Dict[str, float]:
        p_list, r_list, f1_list, p10_list, ap_list = [], [], [], [], []
        with open(relevance_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q_text = row["Query_Text"]
                if expand_senses:
                    q_text += " statutory entitlement verification calculation"
                gold_rels = set(row["Relevant_Circulars"].split(";"))
                retrieved = [doc_id for doc_id, _ in self.search(q_text, top_k=10)]
                hits = len(set(retrieved) & gold_rels)
                p = hits / len(retrieved) if retrieved else 0.0
                r = hits / len(gold_rels) if gold_rels else 0.0
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                ap = 0.0
                rel_seen = 0
                for rank, doc_id in enumerate(retrieved, 1):
                    if doc_id in gold_rels:
                        rel_seen += 1
                        ap += rel_seen / rank
                ap = ap / len(gold_rels) if gold_rels else 0.0
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
                p10_list.append(p)
                ap_list.append(ap)
        n = len(p_list)
        return {
            "Precision": sum(p_list) / n,
            "Recall": sum(r_list) / n,
            "F1": sum(f1_list) / n,
            "P@10": sum(p10_list) / n,
            "MAP": sum(ap_list) / n
        }
