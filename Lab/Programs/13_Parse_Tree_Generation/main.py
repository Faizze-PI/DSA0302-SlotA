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
