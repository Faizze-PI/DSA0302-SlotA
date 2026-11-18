"""
Module: porter_stemmer.py
The Porter Stemmer Algorithm (1980) Implementation and Analysis Engine
DSA03 - Natural Language Processing (Assignment 1 - Task 9)
Author: Clinical NLP Language Engineering Team
Implements the 5-step Porter stemmer from first principles:
- Consonant and vowel classification (handling contextual 'y')
- Measure m calculation: [C](VC)^m[V]
- Condition predicates: *v*, *d, *o
- Traceable 5-step rule execution (1a-1c, 2, 3, 4, 5a-5b)
- Over-stemming and under-stemming error quantification
"""

from typing import Any, Dict, List, Optional, Tuple

class PorterStemmer:
    def __init__(self):
        self.vowels = {"a", "e", "i", "o", "u"}

    def is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in self.vowels:
            return False
        if ch == "y":
            if i == 0:
                return True
            return not self.is_consonant(word, i - 1)
        return True

    def is_vowel(self, word: str, i: int) -> bool:
        return not self.is_consonant(word, i)

    def get_measure(self, word: str) -> int:
        if not word:
            return 0
        form = []
        for i in range(len(word)):
            if self.is_consonant(word, i):
                if not form or form[-1] != "C":
                    form.append("C")
            else:
                if not form or form[-1] != "V":
                    form.append("V")
        form_str = "".join(form)
        return form_str.count("VC")

    def contains_vowel(self, word: str) -> bool:
        for i in range(len(word)):
            if self.is_vowel(word, i):
                return True
        return False

    def ends_double_consonant(self, word: str) -> bool:
        if len(word) < 2:
            return False
        if word[-1] == word[-2]:
            return self.is_consonant(word, len(word) - 1)
        return False

    def ends_cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        i = len(word) - 1
        if self.is_consonant(word, i) and self.is_vowel(word, i - 1) and self.is_consonant(word, i - 2):
            if word[i] not in {"w", "x", "y"}:
                return True
        return False

    def step_1a(self, word: str) -> Tuple[str, str]:
        if word.endswith("sses"):
            return word[:-2], "sses -> ss"
        if word.endswith("ies"):
            return word[:-2], "ies -> i"
        if word.endswith("ss"):
            return word, "ss -> ss"
        if word.endswith("s"):
            return word[:-1], "s -> null"
        return word, "none"

    def step_1b(self, word: str) -> Tuple[str, str]:
        if word.endswith("eed"):
            stem = word[:-3]
            if self.get_measure(stem) > 0:
                return word[:-1], "eed -> ee (m > 0)"
            return word, "none"

        flag = False
        rule_fired = "none"
        if word.endswith("ed"):
            stem = word[:-2]
            if self.contains_vowel(stem):
                word = stem
                flag = True
                rule_fired = "ed -> null (*v*)"
        elif word.endswith("ing"):
            stem = word[:-3]
            if self.contains_vowel(stem):
                word = stem
                flag = True
                rule_fired = "ing -> null (*v*)"

        if flag:
            if word.endswith("at"):
                return word + "e", f"{rule_fired} + at -> ate"
            if word.endswith("bl"):
                return word + "e", f"{rule_fired} + bl -> ble"
            if word.endswith("iz"):
                return word + "e", f"{rule_fired} + iz -> ize"
            if self.ends_double_consonant(word) and not (word.endswith("l") or word.endswith("s") or word.endswith("z")):
                return word[:-1], f"{rule_fired} + double consonant cleanup"
            if self.get_measure(word) == 1 and self.ends_cvc(word):
                return word + "e", f"{rule_fired} + cvc cleanup -> +e"

        return word, rule_fired

    def step_1c(self, word: str) -> Tuple[str, str]:
        if word.endswith("y"):
            stem = word[:-1]
            if self.contains_vowel(stem):
                return stem + "i", "y -> i (*v*)"
        return word, "none"

    def step_2(self, word: str) -> Tuple[str, str]:
        rules = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")
        ]
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.get_measure(stem) > 0:
                    return stem + replacement, f"{suffix} -> {replacement} (m > 0)"
                return word, "none"
        return word, "none"

    def step_3(self, word: str) -> Tuple[str, str]:
        rules = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", "")
        ]
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.get_measure(stem) > 0:
                    return stem + replacement, f"{suffix} -> {replacement} (m > 0)"
                return word, "none"
        return word, "none"

    def step_4(self, word: str) -> Tuple[str, str]:
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.get_measure(stem) > 1:
                    return stem, f"{suffix} -> null (m > 1)"
                return word, "none"
        if word.endswith("sion") or word.endswith("tion"):
            stem = word[:-3]
            if self.get_measure(stem) > 1:
                return stem, "ion -> null (m > 1 and s/t preceding)"
        return word, "none"

    def step_5a(self, word: str) -> Tuple[str, str]:
        if word.endswith("e"):
            stem = word[:-1]
            m = self.get_measure(stem)
            if m > 1:
                return stem, "e -> null (m > 1)"
            if m == 1 and not self.ends_cvc(stem):
                return stem, "e -> null (m = 1 and not *o)"
        return word, "none"

    def step_5b(self, word: str) -> Tuple[str, str]:
        if self.get_measure(word) > 1 and self.ends_double_consonant(word) and word.endswith("l"):
            return word[:-1], "ll -> l (m > 1)"
        return word, "none"

    def stem(self, word: str) -> str:
        w = word.lower()
        if len(w) <= 2:
            return w
        w, _ = self.step_1a(w)
        w, _ = self.step_1b(w)
        w, _ = self.step_1c(w)
        w, _ = self.step_2(w)
        w, _ = self.step_3(w)
        w, _ = self.step_4(w)
        w, _ = self.step_5a(w)
        w, _ = self.step_5b(w)
        return w

    def trace_stemming(self, word: str) -> Dict[str, Any]:
        w = word.lower()
        history = [{"step": "original", "form": w, "rule": "input"}]
        if len(w) <= 2:
            return {"word": word, "final_stem": w, "history": history}

        w, r1a = self.step_1a(w)
        history.append({"step": "Step 1a", "form": w, "rule": r1a})

        w, r1b = self.step_1b(w)
        history.append({"step": "Step 1b", "form": w, "rule": r1b})

        w, r1c = self.step_1c(w)
        history.append({"step": "Step 1c", "form": w, "rule": r1c})

        w, r2 = self.step_2(w)
        history.append({"step": "Step 2", "form": w, "rule": r2})

        w, r3 = self.step_3(w)
        history.append({"step": "Step 3", "form": w, "rule": r3})

        w, r4 = self.step_4(w)
        history.append({"step": "Step 4", "form": w, "rule": r4})

        w, r5a = self.step_5a(w)
        history.append({"step": "Step 5a", "form": w, "rule": r5a})

        w, r5b = self.step_5b(w)
        history.append({"step": "Step 5b", "form": w, "rule": r5b})

        return {
            "word": word,
            "final_stem": w,
            "history": history
        }

    def measure_conditions_on_words(self, words: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if words is None:
            words = ["fracture", "treatment", "operating", "condition", "biopsy", "severe"]
        results = []
        for w in words:
            m = self.get_measure(w)
            has_v = self.contains_vowel(w)
            ends_d = self.ends_double_consonant(w)
            ends_o = self.ends_cvc(w)
            results.append({
                "word": w,
                "measure_m": m,
                "contains_vowel (*v*)": has_v,
                "double_consonant (*d)": ends_d,
                "cvc_condition (*o)": ends_o
            })
        return results

    def compare_with_lemmas(self, annotated_tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        agreed = 0
        diverged = 0
        comparison_table = []
        over_stemmed = 0
        under_stemmed = 0

        for item in annotated_tokens:
            surface = item["surface_form"]
            lemma = item["lemma"]
            stem = self.stem(surface)

            match = (stem == lemma) or (stem in lemma and len(stem) >= 3)
            if match:
                agreed += 1
            else:
                diverged += 1

            if len(stem) < len(lemma) - 2:
                over_stemmed += 1
                error_class = "over-stemming"
            elif stem != lemma and surface.endswith("s") and stem == surface:
                under_stemmed += 1
                error_class = "under-stemming"
            else:
                error_class = "conflation_divergence" if not match else "agreement"

            comparison_table.append({
                "surface": surface,
                "gold_lemma": lemma,
                "porter_stem": stem,
                "agrees": match,
                "classification": error_class
            })

        total = len(annotated_tokens)
        return {
            "total_evaluated": total,
            "agreement_count": agreed,
            "divergence_count": diverged,
            "agreement_rate": round(agreed / total, 4) if total > 0 else 0,
            "divergence_rate": round(diverged / total, 4) if total > 0 else 0,
            "over_stemming_rate": round(over_stemmed / total, 4) if total > 0 else 0,
            "under_stemming_rate": round(under_stemmed / total, 4) if total > 0 else 0,
            "table_sample": comparison_table[:25]
        }
