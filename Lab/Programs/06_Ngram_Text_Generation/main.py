"""
Experiment 06: N-gram Language Model for Text Generation
Course: DSA0302 - Natural Language Processing
"""

import os
import random
from collections import defaultdict, Counter

def load_corpus():
    corpus_file = os.path.join(os.path.dirname(__file__), "training_corpus.txt")
    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    return """
    natural language processing is a branch of artificial intelligence.
    artificial intelligence allows computers to understand human language.
    natural language processing systems analyze text and speech.
    computers process natural language to perform sentiment analysis and translation.
    language modeling is a foundational task in natural language processing.
    language models predict the next word in a sentence using statistical probabilities.
    artificial intelligence is transforming modern technological computing.
    """

class BigramLanguageModel:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.unigrams = Counter()

    def train(self, text):
        words = [w.strip(".,!?:;\"\'").lower() for w in text.split() if w.strip(".,!?:;\"\'")]
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.transitions[w1][w2] += 1
            self.unigrams[w1] += 1
        if words:
            self.unigrams[words[-1]] += 1

    def get_probability(self, w1, w2):
        if w1 not in self.transitions or self.unigrams[w1] == 0:
            return 0.0
        return self.transitions[w1][w2] / self.unigrams[w1]

    def generate_text(self, seed_word, max_length=15, deterministic=False):
        current_word = seed_word.lower()
        generated = [current_word]
        
        for _ in range(max_length - 1):
            next_words = self.transitions.get(current_word)
            if not next_words:
                break
                
            if deterministic:
                # Pick highest probability word
                next_word = next_words.most_common(1)[0][0]
            else:
                # Weighted random selection
                candidates = list(next_words.keys())
                weights = list(next_words.values())
                next_word = random.choices(candidates, weights=weights, k=1)[0]
                
            generated.append(next_word)
            current_word = next_word
            
        return " ".join(generated)

def main():
    random.seed(42)  # For reproducible output
    print("=" * 75)
    print("   EXPERIMENT 06: N-GRAM (BIGRAM) LANGUAGE MODEL FOR TEXT GENERATION   ")
    print("=" * 75)
    
    corpus = load_corpus()
    print("\n[1] TRAINING CORPUS SNIPPET:")
    print("-" * 60)
    print(corpus.strip()[:260] + "...")
    print("-" * 60)
    
    model = BigramLanguageModel()
    model.train(corpus)
    
    print(f"\n[2] MODEL VOCABULARY STATISTICS:")
    print(f"    - Unique vocabulary words: {len(model.unigrams)}")
    print(f"    - Total bigram transitions: {sum(len(v) for v in model.transitions.values())}")
    
    print("\n[3] SAMPLE CONDITIONAL TRANSITION PROBABILITIES:")
    print("-" * 65)
    sample_context = "natural"
    print(f"Transitions following '{sample_context}':")
    for next_w, count in model.transitions[sample_context].items():
        prob = model.get_probability(sample_context, next_w)
        print(f"    P('{next_w}' | '{sample_context}') = {count}/{model.unigrams[sample_context]} = {prob:.4f}")
        
    sample_context_2 = "language"
    print(f"\nTransitions following '{sample_context_2}':")
    for next_w, count in model.transitions[sample_context_2].items():
        prob = model.get_probability(sample_context_2, next_w)
        print(f"    P('{next_w}' | '{sample_context_2}') = {count}/{model.unigrams[sample_context_2]} = {prob:.4f}")
    print("-" * 65)

    print("\n[4] TEXT GENERATION EXPERIMENTS:")
    seeds = ["natural", "artificial", "language", "computers"]
    for seed in seeds:
        gen_det = model.generate_text(seed, max_length=12, deterministic=True)
        gen_sto = model.generate_text(seed, max_length=12, deterministic=False)
        print(f"\n[*] Seed: '{seed}'")
        print(f"    Greedy (Top-1): {gen_det}")
        print(f"    Stochastic    : {gen_sto}")
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
