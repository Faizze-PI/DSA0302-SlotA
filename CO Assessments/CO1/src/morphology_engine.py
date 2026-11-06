"""
Module: morphology_engine.py
Inflectional and Derivational Morphology Analysis Engine
DSA03 - Natural Language Processing (Assignment 1 - Tasks 5 & 6)
Author: Clinical NLP Language Engineering Team
Implements inflectional paradigms, orthographic rewrite rules, irregular lexicons,
derivational family hierarchies, affix productivity counters, and type-to-lemma metrics.
"""

import json
import re
from typing import Any, Dict, List, Set, Tuple

CLINICAL_PARADIGMS: Dict[str, Dict[str, str]] = {
    "fracture": {
        "POS": "Noun/Verb",
        "Base (Lemma)": "fracture",
        "Noun Plural": "fractures",
        "Verb 3SG Present": "fractures",
        "Verb Past Tense": "fractured",
        "Verb Past Participle": "fractured",
        "Verb Progressive/Participle": "fracturing"
    },
    "inject": {
        "POS": "Verb",
        "Base (Lemma)": "inject",
        "Verb 3SG Present": "injects",
        "Verb Past Tense": "injected",
        "Verb Past Participle": "injected",
        "Verb Progressive/Participle": "injecting"
    },
    "biopsy": {
        "POS": "Noun/Verb",
        "Base (Lemma)": "biopsy",
        "Noun Plural": "biopsies",
        "Verb 3SG Present": "biopsies",
        "Verb Past Tense": "biopsied",
        "Verb Past Participle": "biopsied",
        "Verb Progressive/Participle": "biopsying"
    },
    "dilute": {
        "POS": "Verb/Adjective",
        "Base (Lemma)": "dilute",
        "Verb 3SG Present": "dilutes",
        "Verb Past Tense": "diluted",
        "Verb Past Participle": "diluted",
        "Verb Progressive": "diluting",
        "Adjective Comparative": "more dilute",
        "Adjective Superlative": "most dilute"
    },
    "severe": {
        "POS": "Adjective",
        "Base (Lemma)": "severe",
        "Adjective Comparative": "severer",
        "Adjective Superlative": "severest",
        "Adverbial Form": "severely"
    }
}

DERIVATIONAL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "nephr": {
        "root": "nephr (Greek nephros - kidney)",
        "members": [
            ("nephritis", "nephr + -itis", "N -> N", "inflammation of kidney"),
            ("nephrectomy", "nephr + -ectomy", "N -> N", "surgical removal of kidney"),
            ("nephrology", "nephr + -ology", "N -> N", "branch of medicine studying kidneys"),
            ("nephrologist", "nephr + -ology + -ist", "N -> N", "kidney specialist physician"),
            ("nephrotic", "nephr + -otic", "N -> ADJ", "relating to nephrosis/kidney damage"),
            ("nephrolithiasis", "nephro- + lith + -iasis", "Root -> N", "kidney calculi/stones")
        ],
        "illegal_orderings": ["*nephr-ed-itis", "*nephr-ectomy-able-ed"]
    },
    "append": {
        "root": "append (Latin appendere - hang upon)",
        "members": [
            ("appendix", "append + -ix", "V -> N", "vermiform anatomical appendage"),
            ("appendicitis", "appendic- + -itis", "N -> N", "acute inflammation of appendix"),
            ("appendectomy", "append- + -ectomy", "N -> N", "surgical resection of appendix"),
            ("appendiceal", "appendic- + -al", "N -> ADJ", "pertaining to the appendix")
        ],
        "illegal_orderings": ["*append-ing-itis", "*append-itis-ectomy"]
    },
    "cardio": {
        "root": "cardio (Greek kardia - heart)",
        "members": [
            ("carditis", "card- + -itis", "N -> N", "inflammation of heart tissue"),
            ("cardiomegaly", "cardio- + -megaly", "N -> N", "pathological enlargement of heart"),
            ("cardiomyopathy", "cardio- + myo- + -pathy", "N -> N", "disease of heart myocardium"),
            ("cardiology", "cardio- + -ology", "N -> N", "study of heart disorders"),
            ("cardiologist", "cardio- + -ology + -ist", "N -> N", "heart disease specialist"),
            ("cardiovascular", "cardio- + vascul- + -ar", "N -> ADJ", "relating to heart and blood vessels")
        ],
        "illegal_orderings": ["*cardio-ed-ology", "*cardio-s-megaly"]
    },
    "gastr": {
        "root": "gastr (Greek gaster - stomach)",
        "members": [
            ("gastritis", "gastr- + -itis", "N -> N", "inflammation of stomach lining"),
            ("gastrectomy", "gastr- + -ectomy", "N -> N", "surgical excision of stomach"),
            ("gastric", "gastr- + -ic", "N -> ADJ", "relating to stomach"),
            ("gastroenteritis", "gastro- + enter- + -itis", "N -> N", "inflammation of stomach and intestine"),
            ("gastroscopy", "gastro- + -scopy", "N -> N", "endoscopic examination of stomach")
        ],
        "illegal_orderings": ["*gastr-ing-ectomy", "*gastr-itis-ic-itis"]
    },
    "derm": {
        "root": "derm (Greek derma - skin)",
        "members": [
            ("dermatitis", "dermat- + -itis", "N -> N", "inflammation of skin"),
            ("dermatology", "dermato- + -ology", "N -> N", "study of skin disorders"),
            ("dermatologist", "dermato- + -ology + -ist", "N -> N", "skin disease specialist"),
            ("dermatosis", "dermat- + -osis", "N -> N", "non-inflammatory skin lesion"),
            ("hypodermic", "hypo- + derm- + -ic", "Prefix + N -> ADJ", "beneath the skin")
        ],
        "illegal_orderings": ["*derm-ed-itis", "*derm-itis-ed-ectomy"]
    },
    "arthr": {
        "root": "arthr (Greek arthron - joint)",
        "members": [
            ("arthritis", "arthr- + -itis", "N -> N", "inflammation of joints"),
            ("arthroplasty", "arthro- + -plasty", "N -> N", "reconstruction/replacement of joint"),
            ("arthroscopy", "arthro- + -scopy", "N -> N", "minimally invasive joint inspection"),
            ("arthropathy", "arthro- + -pathy", "N -> N", "any disease of joints")
        ],
        "illegal_orderings": ["*arthr-ing-plasty", "*arthr-itis-plasty"]
    }
}

def measure_affix_productivity(corpus_words: List[str], affixes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    productivity_data = []
    text_lower = " ".join(corpus_words).lower()

    for aff_dict in affixes:
        raw_aff = aff_dict["affix"]
        clean_aff = raw_aff.strip("-")
        is_suffix = raw_aff.startswith("-")

        if is_suffix:
            pattern = re.compile(rf"\b[a-z]{{3,}}{re.escape(clean_aff)}\b")
        else:
            pattern = re.compile(rf"\b{re.escape(clean_aff)}[a-z]{{3,}}\b")

        matched_tokens = pattern.findall(text_lower)
        unique_types = set(matched_tokens)
        token_count = len(matched_tokens)
        type_count = len(unique_types)

        classification = "Productive" if type_count >= 5 else "Fossilised / Low-Productivity"
        productivity_data.append({
            "affix": raw_aff,
            "type": "suffix" if is_suffix else "prefix",
            "type_count": type_count,
            "token_count": token_count,
            "sample_words": sorted(list(unique_types))[:5],
            "classification": classification
        })
    return productivity_data

def compute_corpus_morphology_metrics(corpus_records: List[Dict[str, Any]], lexicon: Dict[str, Any]) -> Dict[str, Any]:
    all_tokens = []
    for r in corpus_records:
        tokens = re.findall(r"\b[A-Za-z]+\b", r["text"].lower())
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    unique_types = len(set(all_tokens))
    stems = set(lexicon.get("stems", []))

    mapped_lemmas = set()
    for word in set(all_tokens):
        found = False
        for s in stems:
            if word.startswith(s) or s in word:
                mapped_lemmas.add(s)
                found = True
                break
        if not found:
            mapped_lemmas.add(word)

    type_to_lemma_ratio = round(unique_types / len(mapped_lemmas), 4)

    return {
        "total_word_tokens": total_tokens,
        "unique_word_types": unique_types,
        "estimated_lemmas": len(mapped_lemmas),
        "type_to_lemma_ratio": type_to_lemma_ratio,
        "inflection_density_note": "A type-to-lemma ratio > 1.0 indicates morphological inflection and derivation inflating vocabulary surface variety."
    }
