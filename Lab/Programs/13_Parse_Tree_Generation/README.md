# Experiment 13: Parse Tree Generation for Context-Free Grammars

## Aim
To generate syntactic parse trees for given natural language sentences using context-free grammars and NLTK's ChartParser.

## Software / Tools Required
Python 3.11+, NLTK Library (`CFG`, `ChartParser`, `Tree`).

## Theory & Syntax
A Parse Tree (or concrete syntax tree) is an ordered, rooted tree that represents the syntactic structure of a string according to a Context-Free Grammar.
- Root node: The start symbol S.
- Interior nodes: Non-terminal symbols (NP, VP, PP).
- Leaf nodes: Terminal words from the input sentence.
- Ambiguity: A grammar is syntactically ambiguous if a sentence admits two or more distinct parse trees (e.g., Prepositional Phrase attachment in 'I saw the man with a telescope').

## Procedure
1. Import `CFG`, `ChartParser`, and `Tree` from NLTK.
2. Define a Context-Free Grammar using `CFG.fromstring()` handling noun phrases, verb phrases, and prepositional phrases.
3. Create a `ChartParser` instance.
4. Tokenize target sentences exhibiting syntactic ambiguity.
5. Generate all valid parse trees for the sentence.
6. Display the parse trees in ASCII graphical format and bracketed format.

## Program / Source Code
```python
"""
Experiment 13: Parse Tree Generation for Context-Free Grammars
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk import CFG, ChartParser

def main():
    print("=" * 75)
    print("   EXPERIMENT 13: PARSE TREE GENERATION USING CFG IN NLTK   ")
    print("=" * 75)
    
    # Define Context-Free Grammar exhibiting PP-attachment ambiguity
    grammar_rules = """
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'a' | 'the' | 'my'
    N -> 'man' | 'telescope' | 'park' | 'dog'
    V -> 'saw' | 'walked'
    P -> 'in' | 'with'
    """
    
    cfg = CFG.fromstring(grammar_rules)
    parser = ChartParser(cfg)
    
    sentence = "I saw the man with a telescope"
    tokens = sentence.split()
    
    print("\n[1] CONTEXT-FREE GRAMMAR RULES:")
    print("-" * 55)
    for prod in cfg.productions():
        print("   ", prod)
    print("-" * 55)
    
    print(f"\n[2] TARGET SENTENCE: '{sentence}'")
    print(f"Tokens: {tokens}\n")
    
    trees = list(parser.parse(tokens))
    print(f"[3] GENERATED PARSE TREES (Total Derivations: {len(trees)}):")
    print("=" * 75)
    
    for idx, tree in enumerate(trees, 1):
        print(f"\n--- Parse Tree #{idx} ---")
        print("Bracketed Structure:")
        print(tree)
        print("\nASCII Graphical Tree:")
        tree.pretty_print()
        print("-" * 75)
        
    print("\n[4] SYNTACTIC AMBIGUITY INTERPRETATION:")
    print("   - Tree #1: 'with a telescope' attaches to NP ('the man had the telescope').")
    print("   - Tree #2: 'with a telescope' attaches to VP ('saw using the instrument of a telescope').")
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp13_Output.txt for execution log and Lab/outputs/Exp13_13_Parse_Tree_Generation.png for terminal screenshot)
```

## Result
Syntactic parse trees were successfully generated using NLTK's CFG and ChartParser, clearly illustrating structural constituency and resolving prepositional phrase ambiguity.
