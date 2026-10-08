"""
Experiment 15: Probabilistic Context-Free Grammar (PCFG) Parsing
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk import PCFG, ViterbiParser

def main():
    print("=" * 75)
    print("   EXPERIMENT 15: PROBABILISTIC CONTEXT-FREE GRAMMAR (PCFG) PARSING   ")
    print("=" * 75)
    
    # Define PCFG with assigned rule probabilities
    pcfg_rules = """
    S -> NP VP [1.0]
    VP -> V NP [0.7] | VP PP [0.3]
    PP -> P NP [1.0]
    NP -> Det N [0.6] | NP PP [0.2] | 'I' [0.2]
    Det -> 'the' [0.6] | 'a' [0.4]
    N -> 'astronomer' [0.4] | 'telescope' [0.3] | 'star' [0.3]
    V -> 'saw' [1.0]
    P -> 'with' [1.0]
    """
    
    grammar = PCFG.fromstring(pcfg_rules)
    parser = ViterbiParser(grammar)
    
    sentence = "I saw the star with a telescope"
    tokens = sentence.split()
    
    print("\n[1] PROBABILISTIC GRAMMAR PRODUCTIONS:")
    print("-" * 55)
    for prod in grammar.productions():
        print(f"   {prod}")
    print("-" * 55)
    
    print(f"\n[2] TARGET SENTENCE: '{sentence}'")
    print(f"Tokens: {tokens}\n")
    
    parses = list(parser.parse(tokens))
    
    if parses:
        best_tree = parses[0]
        prob = best_tree.prob()
        print(f"[3] MOST PROBABLE (VITERBI) PARSE TREE:")
        print(f"    Joint Derivation Probability : {prob:.8e}")
        print(f"    Log Probability               : {best_tree.logprob():.4f}")
        print("-" * 75)
        print("Visual Parse Tree:")
        best_tree.pretty_print()
    else:
        print("[-] Sentence could not be parsed with the given PCFG.")
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
