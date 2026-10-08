# Experiment 14: Subject-Verb and Number Agreement Checking

## Aim
To create a Python program that checks for subject-verb and number agreement in sentences based on feature-based context-free grammar constraints.

## Software / Tools Required
Python 3.11+, NLTK Library (`FeatureChartParser`, `grammar.FeatureGrammar`).

## Theory & Syntax
In natural languages, grammatical agreement requires that related words share matching morphological features, such as number (singular vs plural) and person (1st, 2nd, 3rd).
Examples:
- 'The dog barks' (Singular Subject + Singular Verb: AGREE)
- 'The dogs bark' (Plural Subject + Plural Verb: AGREE)
- 'The dog bark' (Singular Subject + Plural Verb: VIOLATION)
- 'The dogs barks' (Plural Subject + Singular Verb: VIOLATION)
Feature-based grammars incorporate feature structures and unification: S -> NP[NUM=?n] VP[NUM=?n].

## Procedure
1. Import feature grammar utilities from `nltk.grammar`.
2. Construct a feature-based CFG incorporating the `[NUM=sg]` and `[NUM=pl]` constraints.
3. Instantiate `nltk.parse.FeatureChartParser`.
4. Parse sample sentences with both valid and invalid agreement structures.
5. Flag sentences that fail unification and diagnose the grammatical mismatch.
6. Output agreement verification reports.

## Program / Source Code
```python
"""
Experiment 14: Subject-Verb and Number Agreement Checking
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk import grammar, parse

def build_agreement_grammar():
    feature_grammar_str = """
    % start S
    # Sentence requires matching number feature between NP and VP
    S -> NP[NUM=?n] VP[NUM=?n]
    
    # Noun Phrases
    NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
    NP[NUM=sg] -> 'he' | 'she' | 'it'
    NP[NUM=pl] -> 'they' | 'we'
    
    # Verb Phrases
    VP[NUM=?n] -> V[NUM=?n] NP
    VP[NUM=?n] -> V[NUM=?n]
    
    # Lexicon with Number Features
    Det[NUM=sg] -> 'this' | 'a' | 'the'
    Det[NUM=pl] -> 'these' | 'all' | 'the'
    
    N[NUM=sg] -> 'dog' | 'cat' | 'student' | 'system'
    N[NUM=pl] -> 'dogs' | 'cats' | 'students' | 'systems'
    
    V[NUM=sg] -> 'barks' | 'runs' | 'studies' | 'works'
    V[NUM=pl] -> 'bark' | 'run' | 'study' | 'work'
    """
    return grammar.FeatureGrammar.fromstring(feature_grammar_str)

def main():
    print("=" * 75)
    print("   EXPERIMENT 14: SUBJECT-VERB AND NUMBER AGREEMENT CHECKING   ")
    print("=" * 75)
    
    feat_grammar = build_agreement_grammar()
    parser = parse.FeatureChartParser(feat_grammar)
    
    test_sentences = [
        ("the dog barks", True, "Singular NP + Singular VP"),
        ("the dogs bark", True, "Plural NP + Plural VP"),
        ("this student studies", True, "Singular NP + Singular VP"),
        ("these students study", True, "Plural NP + Plural VP"),
        ("the dog bark", False, "Agreement Violation: Singular NP with Plural VP"),
        ("the dogs barks", False, "Agreement Violation: Plural NP with Singular VP"),
        ("this students study", False, "Determiner-Noun Mismatch: 'this' (sg) with 'students' (pl)"),
        ("these student studies", False, "Determiner-Noun Mismatch: 'these' (pl) with 'student' (sg)")
    ]
    
    print(f"\n{'Test Sentence':<26} | {'Expected':<10} | {'Parser Result':<14} | {'Diagnosis'}")
    print("-" * 75)
    
    for sentence, expected_valid, desc in test_sentences:
        tokens = sentence.split()
        parses = list(parser.parse(tokens))
        is_valid = len(parses) > 0
        
        status = "AGREED" if is_valid else "REJECTED"
        expected_str = "VALID" if expected_valid else "INVALID"
        
        print(f"{sentence:<26} | {expected_str:<10} | {status:<14} | {desc}")
        
    print("-" * 75)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp14_Output.txt for execution log and Lab/outputs/Exp14_14_Agreement_Checking.png for terminal screenshot)
```

## Result
The agreement checking program was successfully implemented, correctly enforcing subject-verb and determiner-noun number unification constraints.
