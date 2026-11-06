"""
Module: search_retrieval.py
Information Retrieval and Conflation Evaluation Engine
DSA03 - Natural Language Processing (Assignment 1 - Tasks 9 & 11)
Author: Clinical NLP Language Engineering Team
Builds inverted search indexes over clinical records under three regimes:
1. Un-normalised raw tokens
2. Porter-stemmed tokens
3. Full finite-state morphological lemmatisation
Evaluates 40 clinician benchmark queries reporting Precision, Recall, and F1.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from .porter_stemmer import PorterStemmer
from .transducer_fst import MorphologicalFST

class ClinicalSearchEngine:
    def __init__(self, records: List[Dict[str, Any]], lexicon: Optional[Dict[str, Any]] = None):
        self.records = records
        self.porter = PorterStemmer()
        self.fst = MorphologicalFST()
        self.irregulars: Dict[str, str] = {}
        if lexicon and "irregular_forms" in lexicon:
            for irr in lexicon["irregular_forms"]:
                self.irregulars[irr["surface"]] = irr["canonical_stem"]

        self.index_raw: Dict[str, Set[str]] = {}
        self.index_porter: Dict[str, Set[str]] = {}
        self.index_morph: Dict[str, Set[str]] = {}
        self._stem_cache: Dict[str, str] = {}
        self._lemma_cache: Dict[str, str] = {}

        self._build_indexes()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[A-Za-z]+\b", text.lower())

    def _stem_token(self, token: str) -> str:
        if token not in self._stem_cache:
            self._stem_cache[token] = self.porter.stem(token)
        return self._stem_cache[token]

    def _lemmatize_token(self, token: str) -> str:
        if token in self._lemma_cache:
            return self._lemma_cache[token]
        if token in self.irregulars:
            res = self.irregulars[token]
        else:
            parsed = self.fst.parse(token)
            res = parsed[0]["stem"] if parsed else token
        self._lemma_cache[token] = res
        return res

    def _build_indexes(self):
        for rec in self.records:
            rid = rec["record_id"]
            tokens = self._tokenize(rec["text"])
            for t in tokens:
                self.index_raw.setdefault(t, set()).add(rid)

                st = self._stem_token(t)
                self.index_porter.setdefault(st, set()).add(rid)

                lm = self._lemmatize_token(t)
                self.index_morph.setdefault(lm, set()).add(rid)

    def search(self, query: str, condition: str = "raw") -> Set[str]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return set()

        if condition == "raw":
            postings_list = [self.index_raw.get(t, set()) for t in q_tokens]
        elif condition == "porter":
            postings_list = [self.index_porter.get(self.porter.stem(t), set()) for t in q_tokens]
        elif condition == "morphology":
            postings_list = [self.index_morph.get(self._lemmatize_token(t), set()) for t in q_tokens]
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if not postings_list:
            return set()
        
        result = set(postings_list[0])
        for p in postings_list[1:]:
            result = result.intersection(p)
        return result

    def evaluate_queries(self, eval_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "raw": {"precision": [], "recall": [], "f1": []},
            "porter": {"precision": [], "recall": [], "f1": []},
            "morphology": {"precision": [], "recall": [], "f1": []}
        }
        per_query_details = []

        for q_obj in eval_queries:
            query = q_obj["query"]
            gold_relevant = set(q_obj["relevant_record_ids"])

            q_row = {"query": query, "gold_count": len(gold_relevant)}

            for mode in ["raw", "porter", "morphology"]:
                retrieved = self.search(query, condition=mode)
                tp = len(retrieved.intersection(gold_relevant))
                fp = len(retrieved - gold_relevant)
                fn = len(gold_relevant - retrieved)

                prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_relevant else 0.0)
                rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gold_relevant else 0.0)
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                results[mode]["precision"].append(prec)
                results[mode]["recall"].append(rec)
                results[mode]["f1"].append(f1)

                q_row[f"{mode}_retrieved"] = len(retrieved)
                q_row[f"{mode}_f1"] = round(f1, 4)

            per_query_details.append(q_row)

        summary = {}
        for mode in ["raw", "porter", "morphology"]:
            avg_p = sum(results[mode]["precision"]) / len(eval_queries)
            avg_r = sum(results[mode]["recall"]) / len(eval_queries)
            avg_f1 = sum(results[mode]["f1"]) / len(eval_queries)
            summary[mode] = {
                "mean_precision": round(avg_p, 4),
                "mean_recall": round(avg_r, 4),
                "mean_f1": round(avg_f1, 4)
            }

        return {
            "summary_metrics": summary,
            "query_level_table": per_query_details
        }
