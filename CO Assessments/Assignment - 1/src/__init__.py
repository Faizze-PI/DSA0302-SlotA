"""
Package: src
Morphology-Aware Clinical Text Normalisation and Search Layer
DSA03 - Natural Language Processing (Assignment 1 - CO1)
"""

from .entity_patterns import extract_entities, validate_entity
from .tokenizer_normalizer import tokenize, normalise_text
from .automata_engine import (
    FiniteAutomaton,
    subset_construction,
    minimize_dfa,
    build_patient_id_dfa,
    build_blood_pressure_dfa,
    build_temperature_dfa,
    build_icd_code_dfa
)
from .morphology_engine import CLINICAL_PARADIGMS, DERIVATIONAL_FAMILIES
from .transducer_fst import MorphologicalFST
from .porter_stemmer import PorterStemmer
from .search_retrieval import ClinicalSearchEngine
from .day4_challenge import NoisyTextEvaluator
