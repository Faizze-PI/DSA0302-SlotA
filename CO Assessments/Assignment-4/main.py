"""Master CLI pipeline executing all 12 tasks of DSA0302 Assignment-4 (CO4) Government Pension Helpdesk."""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.meaning_representation import MeaningRepresentationEngine
from src.fopc_inference import FOPCModel
from src.linguistic_concepts import LinguisticConceptEngine
from src.lambda_semantics import LambdaSemanticAnalyzer
from src.lexical_semantics import LexicalSemanticsEngine
from src.wsd_engine import WSDEngine
from src.ir_search_engine import IRSearchEngine
from src.annotation_eval import AnnotationEvaluator
from src.day4_challenge import Day4PensionChallenge


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t_start = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("SIMATS ENGINEERING | DSA0302 - NATURAL LANGUAGE PROCESSING | ASSIGNMENT - 4")
    print("TOPIC: Meaning Representation, FOPC, WSD & IR for Government Pension Helpdesk")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("TASK 1 & 2: MEANING REPRESENTATION, VERIFIABILITY & CANONICAL FORM")
    print("=" * 80)
    mr = MeaningRepresentationEngine()
    for name, expr, inf, eff in mr.compare_formalisms():
        print(f"  - {name:<26} | Expr: {expr:<32} | Inf: {inf}")
    print(f"Verifiability Check -> Ramesh: {mr.verify_statement('QualifiesPension', 'ramesh')} | Suresh: {mr.verify_statement('QualifiesPension', 'suresh')} | Mohan: {mr.verify_statement('QualifiesPension', 'mohan')}")
    print(f"Canonical Mapping   -> {mr.to_canonical('does an employee with 10 years service get a pension?')}")

    print("\n" + "=" * 80)
    print("TASK 3: FIRST-ORDER PREDICATE CALCULUS (FOPC) & INFERENCE")
    print("=" * 80)
    fopc = FOPCModel(os.path.join(data_dir, "fopc_knowledge_base_40.csv"))
    print(f"Loaded {len(fopc.rules)} FOPC rules over domain {sorted(list(fopc.domain))}")
    for fname in ["F1_Ramesh_Pension", "F2_Suresh_Pension", "F3_Anita_FamilyPension", "F4_Kavita_FamilyPension", "F5_Suresh_Break", "F6_Universal_All_Ten_Years"]:
        val, reason = fopc.evaluate_formula(fname)
        print(f"  {fname:<26} => {str(val):<5} ({reason})")
    for idx, chain in enumerate(fopc.run_forward_inference(), 1):
        print(f"Inference Chain {idx}: {chain[-1]}")

    print("\n" + "=" * 80)
    print("TASK 4: LINGUISTICALLY RELEVANT CONCEPTS & NER OVER CIRCULARS")
    print("=" * 80)
    ling = LinguisticConceptEngine()
    ent_counts, ner_acc = ling.extract_entities(os.path.join(data_dir, "pension_circulars_300.csv"))
    print(f"NER Accuracy (200 hand-checked mentions): {ner_acc*100:.1f}%")
    print(f"Extracted Entity Counts across 300 Circulars: {ent_counts}")
    for ev_name, ev_lf in ling.get_event_examples()[:3]:
        print(f"  Event [{ev_name}]: {ev_lf}")

    print("\n" + "=" * 80)
    print("TASK 5: SYNTAX-DRIVEN SEMANTIC ANALYSIS (LAMBDA CALCULUS)")
    print("=" * 80)
    lam = LambdaSemanticAnalyzer(os.path.join(data_dir, "semantic_grammar_50.csv"))
    ok, lf, steps = lam.derive_logical_form("does every employee get a pension?")
    print(f"Loaded {len(lam.rules)} semantic grammar rules. Derivation: {lf}")
    cov, fails = lam.evaluate_coverage(os.path.join(data_dir, "helpdesk_enquiries_200.csv"))
    print(f"Coverage over 200 enquiries: {cov*100:.1f}% | Failures: {fails}")

    print("\n" + "=" * 80)
    print("TASK 6: LEXEMES, SENSES & SELECTIONAL RESTRICTIONS")
    print("=" * 80)
    lex = LexicalSemanticsEngine(os.path.join(data_dir, "sense_inventory_25.csv"))
    print(f"Sense Inventory: {len(lex.inventory)} ambiguous domain terms")
    for term, rel, target in lex.get_wordnet_relations()[:4]:
        print(f"  {term:<15} --[{rel}]--> {target}")

    print("\n" + "=" * 80)
    print("TASK 7: WORD SENSE DISAMBIGUATION (LESK, MFS & SUPERVISED CV)")
    print("=" * 80)
    wsd = WSDEngine(lex.inventory, os.path.join(data_dir, "sense_annotations_600.csv"))
    wsd_scores = wsd.evaluate_all_methods()
    for method, acc in wsd_scores.items():
        print(f"  {method:<22}: {acc*100:.2f}%")

    print("\n" + "=" * 80)
    print("TASK 8: INFORMATION RETRIEVAL OVER 300 CIRCULARS (60 QUERIES)")
    print("=" * 80)
    ir = IRSearchEngine(os.path.join(data_dir, "pension_circulars_300.csv"))
    ir_base = ir.evaluate_queries(os.path.join(data_dir, "ir_relevance_60.csv"), expand_senses=False)
    ir_exp = ir.evaluate_queries(os.path.join(data_dir, "ir_relevance_60.csv"), expand_senses=True)
    print(f"  Lexical TF-IDF Baseline : P={ir_base['Precision']:.3f} R={ir_base['Recall']:.3f} F1={ir_base['F1']:.3f} MAP={ir_base['MAP']:.3f}")
    print(f"  Sense-Expanded Retrieval: P={ir_exp['Precision']:.3f} R={ir_exp['Recall']:.3f} F1={ir_exp['F1']:.3f} MAP={ir_exp['MAP']:.3f}")

    print("\n" + "=" * 80)
    print("TASK 9: ANNOTATION GOLD STANDARD & COHEN'S KAPPA")
    print("=" * 80)
    ann = AnnotationEvaluator()
    s_raw, s_kap, s_n = ann.compute_sense_kappa(os.path.join(data_dir, "sense_annotations_600.csv"))
    r_raw, r_kap, r_n = ann.compute_relevance_kappa(os.path.join(data_dir, "ir_relevance_60.csv"))
    print(f"  Sense Annotation (n={s_n})    : Raw={s_raw*100:.1f}%, Cohen's kappa={s_kap:.3f}")
    print(f"  Relevance Judgements (n={r_n}): Raw={r_raw*100:.1f}%, Cohen's kappa={r_kap:.3f}")

    print("\n" + "=" * 80)
    print("TASK 10 & 11: PIPELINE BENCHMARK & DESIGN RECOMMENDATION")
    print("=" * 80)
    print("  Recommended Architecture: Hybrid Sense-Aware IR + FOPC Rule Engine + Fallback Clarification")

    print("\n" + "=" * 80)
    print("TASK 12: DAY-4 DESIGN CHALLENGE (TEMPORAL & AGGREGATION)")
    print("=" * 80)
    d4 = Day4PensionChallenge(os.path.join(data_dir, "day4_new_scheme_enquiries.csv"))
    for res in d4.evaluate_adaptation():
        print(f"  [{res['Enquiry_ID']}] {res['Challenge_Type']:<32} | Base: {res['Baseline_FOPC']} -> Mod: {res['Modified_Hybrid']}")

    plt.figure(figsize=(7, 4.5), dpi=220)
    m_names = list(wsd_scores.keys())
    m_vals = [v * 100 for v in wsd_scores.values()]
    bars = plt.bar(m_names, m_vals, color=["#6c757d", "#1f77b4", "#17becf", "#2ca02c", "#9467bd"], width=0.55)
    plt.ylim(0, 115)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.title("WSD Accuracy Comparison (600 Annotated Occurrences)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=8.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_wsd_accuracy_comparison.png"))
    plt.close()

    plt.figure(figsize=(7, 4.5), dpi=220)
    metrics = ["Precision", "Recall", "F1", "MAP"]
    base_v = [ir_base[m] for m in metrics]
    exp_v = [ir_exp[m] for m in metrics]
    x = range(len(metrics))
    plt.bar([i - 0.18 for i in x], base_v, width=0.35, label="Lexical TF-IDF", color="#1f77b4")
    plt.bar([i + 0.18 for i in x], exp_v, width=0.35, label="Sense-Expanded", color="#2ca02c")
    plt.xticks(list(x), metrics)
    plt.ylim(0, max(max(base_v), max(exp_v)) * 1.35 + 0.1)
    plt.title("IR Performance: Lexical vs. Sense-Aware Expansion", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_ir_lexical_vs_sense_aware.png"))
    plt.close()

    elapsed = time.time() - t_start
    print(f"\nASSIGNMENT-4 PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")


if __name__ == "__main__":
    main()
