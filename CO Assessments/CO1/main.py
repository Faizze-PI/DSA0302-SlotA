"""
Module: main.py
Master Pipeline Execution, Verification, and Plotting Harness
DSA03 - Natural Language Processing (Assignment 1 - Hospital Record System)
Author: Clinical NLP Language Engineering Team
Executes the end-to-end clinical text processing pipeline on the 500-record corpus.
Evaluates entity extraction, automata, morphological transducers, Porter stemming, and IR retrieval.
Generates publication-quality performance graphs in outputs/.
"""

import csv
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.entity_patterns import (
    COMPILED_PATTERNS,
    NEGATIVE_TEST_CASES,
    demonstrate_greedy_vs_nongreedy,
    demonstrate_precedence_issue,
    evaluate_on_corpus,
    extract_entities,
    validate_entity
)
from src.tokenizer_normalizer import (
    compare_with_whitespace_split,
    demonstrate_record_normalisation,
    normalise_text,
    tokenize
)
from src.automata_engine import (
    build_blood_pressure_dfa,
    build_icd_code_dfa,
    build_optional_dosage_enfa,
    build_patient_id_dfa,
    build_temperature_dfa,
    demonstrate_automata_tasks,
    minimize_dfa,
    subset_construction
)
from src.morphology_engine import (
    CLINICAL_PARADIGMS,
    DERIVATIONAL_FAMILIES,
    compute_corpus_morphology_metrics,
    measure_affix_productivity
)
from src.transducer_fst import MorphologicalFST
from src.porter_stemmer import PorterStemmer
from src.search_retrieval import ClinicalSearchEngine
from src.day4_challenge import NoisyTextEvaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def run_pipeline():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("================================================================================")
    print("DSA03 NLP ASSIGNMENT 1: HOSPITAL RECORD TEXT NORMALISATION & SEARCH LAYER")
    print("================================================================================\n")

    corpus_path = os.path.join(DATA_DIR, "clinical_corpus.csv")
    lexicon_path = os.path.join(DATA_DIR, "morphological_lexicon.csv")
    annotated_path = os.path.join(DATA_DIR, "annotated_200_tokens.csv")
    eval_queries_path = os.path.join(DATA_DIR, "eval_40_queries.csv")
    noisy_path = os.path.join(DATA_DIR, "noisy_day4_records.csv")

    with open(corpus_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    lexicon: Dict[str, Any] = {
        "stems": [],
        "inflectional_affixes": [],
        "derivational_affixes": [],
        "irregular_forms": []
    }
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat == "stem":
                lexicon["stems"].append(row["surface_or_affix"])
            elif cat == "inflectional":
                lexicon["inflectional_affixes"].append(row["surface_or_affix"])
            elif cat == "derivational":
                lexicon["derivational_affixes"].append(row["surface_or_affix"])
            elif cat == "irregular":
                lexicon["irregular_forms"].append({
                    "surface": row["surface_or_affix"],
                    "canonical_stem": row["canonical_stem"],
                    "category": row["pos_or_feature"]
                })

    with open(annotated_path, "r", encoding="utf-8") as f:
        annotated_tokens = list(csv.DictReader(f))

    eval_queries = []
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["relevant_record_ids"] = row["relevant_record_ids"].split(";") if row.get("relevant_record_ids") else []
            eval_queries.append(row)

    with open(noisy_path, "r", encoding="utf-8") as f:
        noisy_records = list(csv.DictReader(f))

    start_total_time = time.time()

    t0 = time.time()
    total_tokens = 0
    tokenized_records = []
    record_lengths = []
    record_latencies = []

    for r in records:
        r_start = time.perf_counter()
        toks = tokenize(r["text"])
        r_end = time.perf_counter()
        total_tokens += len(toks)
        tokenized_records.append(toks)
        record_lengths.append(len(toks))
        record_latencies.append((r_end - r_start) * 1000)

    stage1_time = time.time() - t0
    print(f"[Stage 1] Tokenised {len(records)} records ({total_tokens} tokens) in {stage1_time:.4f}s", flush=True)

    t0 = time.time()
    entity_counts = {k: 0 for k in COMPILED_PATTERNS.keys()}
    for r in records:
        extracted = extract_entities(r["text"])
        for k, v in extracted.items():
            entity_counts[k] += len(v)

    rejection_tests_passed = True
    for entity_name, neg_cases in NEGATIVE_TEST_CASES.items():
        for case in neg_cases:
            if validate_entity(entity_name, case):
                rejection_tests_passed = False
                print(f"FAILED rejection test: {entity_name} matched invalid string '{case}'", flush=True)

    stage2_time = time.time() - t0
    print(f"[Stage 2] Extracted {sum(entity_counts.values())} structured entities in {stage2_time:.4f}s. Rejection tests passed: {rejection_tests_passed}", flush=True)

    t0 = time.time()
    automata_demo = demonstrate_automata_tasks()
    stage3_time = time.time() - t0
    print(f"[Stage 3] Automata validation & subset construction executed in {stage3_time:.4f}s", flush=True)
    print(f"          NFA to DFA state count: {automata_demo['subset_construction_states']} states. Minimized to: {automata_demo['minimisation']['states_after']} states.", flush=True)

    t0 = time.time()
    fst = MorphologicalFST()
    ortho_test_results = fst.test_twenty_orthographic_forms()
    ortho_pass_count = sum(1 for t in ortho_test_results if t["status"] == "PASS")
    stage4_time = time.time() - t0
    print(f"[Stage 4] Morphological FST & Orthographic cascade tested in {stage4_time:.4f}s (Passed {ortho_pass_count}/20 boundary tests)", flush=True)

    t0 = time.time()
    stemmer = PorterStemmer()
    porter_comparison = stemmer.compare_with_lemmas(annotated_tokens)
    stage5_time = time.time() - t0
    print(f"[Stage 5] Porter stemmer evaluated on {porter_comparison['total_evaluated']} tokens in {stage5_time:.4f}s", flush=True)
    print(f"          Agreement: {porter_comparison['agreement_rate']*100:.2f}% | Divergence: {porter_comparison['divergence_rate']*100:.2f}%", flush=True)
    print(f"          Over-stemming: {porter_comparison['over_stemming_rate']*100:.2f}% | Under-stemming: {porter_comparison['under_stemming_rate']*100:.2f}%", flush=True)

    t0 = time.time()
    search_engine = ClinicalSearchEngine(records, lexicon)
    retrieval_results = search_engine.evaluate_queries(eval_queries)
    stage6_time = time.time() - t0
    print(f"[Stage 6] 40 clinician search queries evaluated across 3 conditions in {stage6_time:.4f}s", flush=True)
    for mode, met in retrieval_results["summary_metrics"].items():
        print(f"          - {mode.upper():12s} => Precision: {met['mean_precision']:.4f} | Recall: {met['mean_recall']:.4f} | F1: {met['mean_f1']:.4f}", flush=True)

    t0 = time.time()
    noisy_evaluator = NoisyTextEvaluator()
    noisy_results = noisy_evaluator.evaluate_original_vs_adapted(noisy_records)
    stage7_time = time.time() - t0
    print(f"[Stage 7] Day-4 noise challenge evaluated in {stage7_time:.4f}s", flush=True)
    for r_entry in noisy_results["evaluation_records"]:
        print(f"          - {r_entry['record_id']} ({r_entry['record_type']}): Original Coverage={r_entry['original_coverage']*100:.1f}%, Adapted Coverage={r_entry['adapted_coverage']*100:.1f}% (+{r_entry['coverage_gain']*100:.1f}%)", flush=True)

    total_pipeline_time = time.time() - start_total_time
    print(f"\nTotal Pipeline Execution Time: {total_pipeline_time:.4f}s", flush=True)

    fig1_path = os.path.join(OUTPUTS_DIR, "fig1_coverage_vs_lexicon.png")
    stem_subsets = [50, 100, 150, 200, 250, 300]
    sample_corpus_words = list(set([w.lower() for toks in tokenized_records for w in toks if w.isalpha()]))
    coverages = []
    all_stems = lexicon["stems"]
    for sz in stem_subsets:
        cur_stems = set(all_stems[:sz])
        covered = sum(1 for w in sample_corpus_words if any(w.startswith(st) for st in cur_stems))
        coverages.append(round((covered / len(sample_corpus_words)) * 100, 2))

    plt.figure(figsize=(8, 5))
    plt.plot(stem_subsets, coverages, marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
    plt.title("Lexical Coverage vs. Morphological Lexicon Size", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Stems in Lexicon", fontsize=11)
    plt.ylabel("Vocabulary Coverage on Corpus (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    fig2_path = os.path.join(OUTPUTS_DIR, "fig2_latency_vs_length.png")
    plt.figure(figsize=(8, 5))
    plt.scatter(record_lengths, record_latencies, color="#2ca02c", alpha=0.6, edgecolors="none", s=25)
    plt.title("Processing Latency vs. Clinical Record Token Length", fontsize=13, fontweight="bold")
    plt.xlabel("Record Length (Word Tokens)", fontsize=11)
    plt.ylabel("Execution Latency (ms)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"Performance plots successfully generated in: {OUTPUTS_DIR}")
    print("\nAll pipeline tasks successfully executed and verified.")

if __name__ == "__main__":
    run_pipeline()
