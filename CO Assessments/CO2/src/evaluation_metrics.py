"""
Module: evaluation_metrics.py
Information-Theoretic Evaluation, Perplexity, and ASR Hypothesis Rescoring
DSA03 - Natural Language Processing (Assignment 2 - Task 5)
Author: Transport Speech Analytics & Language Engineering Team
Computes Shannon entropy, cross-entropy, per-word and per-utterance perplexity.
Rescores 200 competing ASR acoustic hypotheses to evaluate Word Error Rate (WER) reduction.
"""

import math
from typing import Any, Callable, Dict, List, Set, Tuple

def compute_levenshtein_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[r_len][h_len]

class LanguageModelEvaluator:
    def __init__(self, vocab: Set[str]):
        self.vocab = vocab

    def compute_distribution_entropy(self, probabilities: List[float]) -> float:
        h = 0.0
        for p in probabilities:
            if p > 0:
                h -= p * math.log2(p)
        return round(h, 4)

    def demonstrate_three_entropies(self) -> Dict[str, Any]:
        uniform_dist = [0.1] * 10
        h_uniform = self.compute_distribution_entropy(uniform_dist)

        peaked_dist = [0.90, 0.05, 0.03, 0.02]
        h_peaked = self.compute_distribution_entropy(peaked_dist)

        moderate_dist = [0.40, 0.25, 0.15, 0.10, 0.10]
        h_moderate = self.compute_distribution_entropy(moderate_dist)

        return {
            "uniform": {
                "description": "Uniform distribution over 10 transport choices (Maximum uncertainty)",
                "probabilities": uniform_dist,
                "entropy_bits": h_uniform,
                "perplexity": round(2 ** h_uniform, 2)
            },
            "peaked": {
                "description": "Sharply peaked distribution following 'Chennai' -> 'Central' (High certainty)",
                "probabilities": peaked_dist,
                "entropy_bits": h_peaked,
                "perplexity": round(2 ** h_peaked, 2)
            },
            "moderate": {
                "description": "Moderate domain distribution across bus categories (Real-world certainty)",
                "probabilities": moderate_dist,
                "entropy_bits": h_moderate,
                "perplexity": round(2 ** h_moderate, 2)
            }
        }

    def evaluate_model_perplexity(
        self,
        test_sentences: List[List[str]],
        prob_fn: Callable[[str, str, str], float],
        model_name: str
    ) -> Dict[str, Any]:
        total_log_prob = 0.0
        total_words = 0
        utterance_perplexities = []

        for sent in test_sentences:
            cleaned = [w if w in self.vocab or w in ("<s>", "</s>") else "<UNK>" for w in sent]
            sent_log_prob = 0.0
            sent_words = 0

            for i in range(len(cleaned) - 2):
                w1, w2, w3 = cleaned[i], cleaned[i+1], cleaned[i+2]
                p = max(prob_fn(w1, w2, w3), 1e-12)
                lp = math.log2(p)
                total_log_prob += lp
                sent_log_prob += lp
                total_words += 1
                sent_words += 1

            if sent_words > 0:
                sent_cross_ent = - (sent_log_prob / sent_words)
                utterance_perplexities.append(2 ** min(sent_cross_ent, 20.0))

        cross_entropy = - (total_log_prob / total_words) if total_words > 0 else float("inf")
        per_word_pp = 2 ** min(cross_entropy, 20.0)
        mean_utterance_pp = sum(utterance_perplexities) / len(utterance_perplexities) if utterance_perplexities else float("inf")

        return {
            "model_name": model_name,
            "total_tokens_evaluated": total_words,
            "cross_entropy_bits": round(cross_entropy, 4),
            "per_word_perplexity": round(per_word_pp, 2),
            "per_utterance_perplexity": round(mean_utterance_pp, 2)
        }

    def demonstrate_unfair_comparison(self, test_sentences: List[List[str]], full_prob_fn: Callable) -> Dict[str, Any]:
        fair_metrics = self.evaluate_model_perplexity(test_sentences, full_prob_fn, "Standard Model (Vocab=Full)")
        
        total_lp = 0.0
        words_count = 0
        for sent in test_sentences:
            cleaned = [w for w in sent if w not in ("<s>", "</s>")]
            for i in range(len(cleaned) - 2):
                w1, w2, w3 = cleaned[i], cleaned[i+1], cleaned[i+2]
                p = max(full_prob_fn(w1, w2, w3), 0.05)
                total_lp += math.log2(p)
                words_count += 1
        
        artificially_low_pp = 2 ** (- (total_lp / words_count))

        return {
            "fair_evaluation": {
                "vocab_treatment": "Strict full vocabulary with <UNK> penalty",
                "perplexity": fair_metrics["per_word_perplexity"]
            },
            "unfair_evaluation": {
                "vocab_treatment": "Artificially filtered small vocabulary discarding rare words",
                "distorted_perplexity": round(artificially_low_pp, 2),
                "explanation": "Perplexity is inversely proportional to vocabulary entropy. Artificially shrinking vocabulary or ignoring rare/unseen tokens artificially deflates perplexity, rendering cross-paper comparisons invalid unless vocabulary is strictly controlled."
            }
        }

    def rescore_asr_hypotheses(
        self,
        asr_records: List[Dict[str, Any]],
        lm_scorer: Callable[[str], float],
        beta: float = 0.4
    ) -> Dict[str, Any]:
        baseline_errors = 0
        rescored_errors = 0
        total_ref_words = 0
        
        rescored_results = []

        for rec in asr_records:
            ref = rec["reference_transcript"].strip()
            ref_words = ref.split()
            total_ref_words += len(ref_words)

            candidates = []
            for i in range(1, 6):
                hyp_txt = rec[f"hyp{i}"].strip()
                ac_score = float(rec[f"score{i}"])
                lm_score = lm_scorer(hyp_txt)
                combined = ac_score + (beta * lm_score)
                candidates.append((hyp_txt, ac_score, lm_score, combined))

            baseline_pick = max(candidates, key=lambda x: x[1])[0]
            rescored_pick = max(candidates, key=lambda x: x[3])[0]

            d_base = compute_levenshtein_distance(ref_words, baseline_pick.split())
            d_resc = compute_levenshtein_distance(ref_words, rescored_pick.split())

            baseline_errors += d_base
            rescored_errors += d_resc

            rescored_results.append({
                "utterance_id": rec["utterance_id"],
                "reference": ref,
                "baseline_1best": baseline_pick,
                "rescored_best": rescored_pick,
                "baseline_wer": round(d_base / len(ref_words), 4),
                "rescored_wer": round(d_resc / len(ref_words), 4)
            })

        baseline_wer = (baseline_errors / total_ref_words) * 100 if total_ref_words else 0.0
        rescored_wer = (rescored_errors / total_ref_words) * 100 if total_ref_words else 0.0

        return {
            "total_utterances": len(asr_records),
            "total_reference_words": total_ref_words,
            "baseline_wer_pct": round(baseline_wer, 2),
            "rescored_wer_pct": round(rescored_wer, 2),
            "absolute_wer_reduction": round(baseline_wer - rescored_wer, 2),
            "relative_wer_reduction_pct": round(((baseline_wer - rescored_wer) / baseline_wer) * 100, 2) if baseline_wer else 0.0,
            "sample_rescored": rescored_results[:5]
        }
