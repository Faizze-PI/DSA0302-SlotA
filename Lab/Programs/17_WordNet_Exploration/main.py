"""
Experiment 17: Lexical Database Exploration using WordNet
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.corpus import wordnet as wn

def ensure_wordnet():
    try:
        wn.synsets('dog')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

def main():
    ensure_wordnet()
    print("=" * 75)
    print("   EXPERIMENT 17: LEXICAL DATABASE EXPLORATION USING WORDNET   ")
    print("=" * 75)
    
    target_word = "bank"
    synsets = wn.synsets(target_word)
    
    print(f"\n[1] SYNSETS FOR POLYSEMOUS WORD '{target_word}' (Total: {len(synsets)}):")
    print("-" * 75)
    for idx, syn in enumerate(synsets[:5], 1):
        pos_name = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 'r': 'Adverb'}.get(syn.pos(), syn.pos())
        print(f"{idx}. Synset : {syn.name()} ({pos_name})")
        print(f"   Gloss  : {syn.definition()}")
        examples = syn.examples()
        print(f"   Example: \"{examples[0]}\"" if examples else "   Example: None")
        print()
    print("-" * 75)
    
    # Hypernym Hierarchy Exploration
    concept = wn.synset("car.n.01")
    print(f"\n[2] TAXONOMIC HYPERNYM HIERARCHY FOR '{concept.name()}':")
    print("-" * 65)
    hypernym_paths = concept.hypernym_paths()[0]
    for depth, node in enumerate(hypernym_paths):
        indent = "  " * depth + "|-- "
        print(f"{indent}{node.name()} : {node.definition()[:50]}...")
    print("-" * 65)
    
    # Antonym Extraction
    print("\n[3] ANTONYM IDENTIFICATION:")
    pairs = [("good", "a"), ("increase", "v"), ("happy", "a")]
    for word, pos in pairs:
        for lemma in wn.lemmas(word, pos=pos):
            if lemma.antonyms():
                ant_name = lemma.antonyms()[0].name()
                print(f"    Word: '{word}' <---> Antonym: '{ant_name}'")
                break
    print("-" * 65)
    
    # Semantic Similarity Metrics
    print("\n[4] SEMANTIC SIMILARITY COMPUTATION (Wu-Palmer Metric):")
    w1 = wn.synset("dog.n.01")
    comparisons = [
        ("cat.n.01", "Cat"),
        ("wolf.n.01", "Wolf"),
        ("car.n.01", "Car"),
        ("computer.n.01", "Computer")
    ]
    print(f"Base Concept: {w1.name()} ('Dog')")
    for syn_name, label in comparisons:
        w2 = wn.synset(syn_name)
        wup_score = w1.wup_similarity(w2)
        path_score = w1.path_similarity(w2)
        print(f"    vs {label:<10} ({syn_name:<15}) => Wu-Palmer: {wup_score:.4f} | Path Similarity: {path_score:.4f}")
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
