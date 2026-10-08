# Experiment 15: Probabilistic Context-Free Grammar (PCFG) Parsing

## Aim
To implement Probabilistic Context-Free Grammar (PCFG) parsing in Python and compute the most probable parse tree using NLTK's Viterbi parser.

## Software / Tools Required
Python 3.11+, NLTK Library (`PCFG`, `ViterbiParser`).

## Theory & Syntax
A Probabilistic Context-Free Grammar (PCFG) augments standard CFGs by assigning a conditional probability P(A -> beta) to each production rule, such that for every non-terminal A:
  sum_{beta} P(A -> beta) = 1.0

The probability of a complete parse tree T is the product of the probabilities of all production rules used in its derivation:
  P(T) = prod_{r in T} P(r)

The Viterbi PCFG Parser uses dynamic programming to discover the single highest probability tree (Viterbi parse) among competing ambiguous derivations.

## Procedure
1. Import `PCFG` and `ViterbiParser` from NLTK.
2. Define production rules with assigned probabilities ensuring rule distributions sum to 1.0.
3. Instantiate `ViterbiParser`.
4. Parse ambiguous sentences (e.g., telescope sentence).
5. Compute joint tree probabilities.
6. Print the optimal parse tree and associated likelihood.

## Program / Source Code
```python
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
```

## Output Block
```text
(Refer to Lab/outputs/Exp15_Output.txt for execution log and Lab/outputs/Exp15_15_PCFG_Parsing.png for terminal screenshot)
```

## Result
PCFG parsing was successfully executed using NLTK's ViterbiParser, accurately ranking and selecting the globally optimal parse tree based on rule probability distributions.
