"""
Module: tokenizer_normalizer.py
Maximal-Munch Regular Tokenizer and Normalisation Engine
DSA03 - Natural Language Processing (Assignment 1 - Task 3)
Author: Clinical NLP Language Engineering Team
Implements single-regex maximal munch tokenisation, abbreviation expansion,
and format canonicalisation for clinical text records.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

TOKEN_REGEX = re.compile(
    r"(?P<ABBREV>\b(?:[A-Za-z]\.){2,})"
    r"|(?P<NUMBER_WITH_UNIT>\b\d+(?:\.\d+)?(?:mg|g|mcg|mL|mmol/L|g/dL|/uL|mmHg|F|C)\b)"
    r"|(?P<DECIMAL>\b\d+\.\d+\b)"
    r"|(?P<DATE>\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b)"
    r"|(?P<HYPHEN_WORD>\b[A-Za-z]+(?:-[A-Za-z]+)+\b)"
    r"|(?P<ALPHANUM_ID>\b[A-Z]+-\d+\b)"
    r"|(?P<WORD>\b[A-Za-z]+\b)"
    r"|(?P<INTEGER>\b\d+\b)"
    r"|(?P<PUNCT>[^\w\s])",
    re.UNICODE
)

ABBREVIATION_EXPANSIONS: Dict[str, str] = {
    "prn": "as needed",
    "p.r.n.": "as needed",
    "stat": "immediately",
    "po": "by mouth",
    "p.o.": "by mouth",
    "bid": "twice daily",
    "b.i.d.": "twice daily",
    "tid": "three times daily",
    "t.i.d.": "three times daily",
    "qid": "four times daily",
    "q.i.d.": "four times daily",
    "qds": "four times daily",
    "od": "once daily",
    "o.d.": "once daily",
    "hs": "at bedtime",
    "h.s.": "at bedtime",
    "iv": "intravenous",
    "i.v.": "intravenous",
    "im": "intramuscular",
    "i.m.": "intramuscular",
    "sc": "subcutaneous",
    "s.c.": "subcutaneous",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "temp": "temperature",
    "hb": "hemoglobin",
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "plt": "platelet count",
    "cr": "creatinine",
    "bun": "blood urea nitrogen",
    "cxr": "chest x-ray",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "usg": "ultrasonography",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "icu": "intensive care unit",
    "opd": "outpatient department",
    "dx": "diagnosis",
    "rx": "prescription",
    "hx": "history",
    "sx": "symptoms",
    "tx": "treatment",
    "yo": "years old",
    "y.o.": "years old"
}

def tokenize(text: str) -> List[str]:
    tokens = []
    for match in TOKEN_REGEX.finditer(text):
        tokens.append(match.group(0))
    return tokens

def compare_with_whitespace_split(examples: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if examples is None:
        examples = [
            "Administer Paracetamol 500mg, oral b.i.d. after meals.",
            "Post-operative Hb 13.5 g/dL was checked for PID-10024.",
            "Patient age: 45 yo, admitted on 14/05/2023 with BP 140/90 mmHg."
        ]
    
    comparisons = []
    for sent in examples:
        naive = sent.split()
        regex_tokens = tokenize(sent)
        comparisons.append({
            "input_sentence": sent,
            "naive_whitespace_tokens": naive,
            "regex_maximal_munch_tokens": regex_tokens,
            "naive_token_count": len(naive),
            "regex_token_count": len(regex_tokens)
        })
    return comparisons

def canonicalise_dates(text: str) -> str:
    def _convert_slash_date(m: re.Match) -> str:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{int(mth):02d}-{int(d):02d}"
    
    pattern = r"\b(\d{2})/(\d{2})/(\d{4})\b"
    return re.sub(pattern, _convert_slash_date, text)

def canonicalise_units(text: str) -> str:
    substitutions = [
        (r"\bmg/dl\b", "mg/dL"),
        (r"\bg/dl\b", "g/dL"),
        (r"\bmmol/l\b", "mmol/L"),
        (r"\bmmhg\b", "mmHg"),
        (r"\b([0-9]+)\s*([mM][gG])\b", r"\1 mg"),
        (r"\b([0-9]+)\s*([mM][lL])\b", r"\1 mL"),
    ]
    cur = text
    for pat, rep in substitutions:
        cur = re.sub(pat, rep, cur, flags=re.IGNORECASE)
    return cur

def expand_abbreviations(text: str) -> str:
    words = re.findall(r"\b[A-Za-z]+(?:\.[A-Za-z]+)*\.?\b|[^\w\s]", text)
    result = []
    for w in words:
        w_clean = w.lower()
        if w_clean in ABBREVIATION_EXPANSIONS:
            result.append(ABBREVIATION_EXPANSIONS[w_clean])
        else:
            result.append(w)
    return " ".join(result)

def normalise_text(text: str, fold_case: bool = True, expand_abbrev: bool = True) -> str:
    step1 = canonicalise_dates(text)
    step2 = canonicalise_units(step1)
    if expand_abbrev:
        step2 = expand_abbreviations(step2)
    if fold_case:
        step2 = step2.lower()
    step2 = re.sub(r"\s+", " ", step2).strip()
    return step2

def demonstrate_record_normalisation(raw_record: str) -> Dict[str, str]:
    normalised = normalise_text(raw_record, fold_case=True, expand_abbrev=True)
    return {
        "before": raw_record,
        "after": normalised
    }
