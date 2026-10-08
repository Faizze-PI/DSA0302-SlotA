# Experiment 03: Morphological Analysis using NLTK

## Aim
To write a Python program demonstrating how to perform morphological analysis, lemmatization, and morpheme decomposition using the NLTK library.

## Software / Tools Required
Python 3.11+, NLTK Library (`wordnet`, `punkt`).

## Theory & Syntax
Morphology is the study of the structure of words and how words are formed from smaller units called morphemes (stems, prefixes, and suffixes).
- Inflectional Morphology: Changes grammatical category or tense without changing core lexical meaning (e.g., walk -> walks, walked, walking).
- Derivational Morphology: Creates new words or parts of speech (e.g., happy -> happiness, unhappy).
- Lemmatization: Uses lexical databases (such as WordNet) and context (POS tag) to reduce words to their base dictionary lemma (e.g., 'better' with POS 'a' -> 'good').

## Procedure
1. Import NLTK, `WordNetLemmatizer`, and `word_tokenize`.
2. Download required corpora (`wordnet`, `punkt`, `omw-1.4`).
3. Instantiate the `WordNetLemmatizer`.
4. Perform lemmatization on nouns, verbs, adjectives, and adverbs with respective POS flags.
5. Compare stemming versus lemmatization.
6. Implement prefix/suffix rule-based decomposition for derivational morphology.
7. Print analysis tables.

## Program / Source Code
```python
"""
Experiment 03: Morphological Analysis using NLTK
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def ensure_nltk_resources():
    for res in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{res}' if 'wordnet' in res or 'omw' in res else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

def morphological_analysis():
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    
    words_data = [
        ("running", wordnet.VERB),
        ("flies", wordnet.VERB),
        ("flies", wordnet.NOUN),
        ("better", wordnet.ADJ),
        ("best", wordnet.ADJ),
        ("corpora", wordnet.NOUN),
        ("geese", wordnet.NOUN),
        ("wolves", wordnet.NOUN),
        ("studied", wordnet.VERB),
        ("happiest", wordnet.ADJ),
        ("easily", wordnet.ADV),
        ("went", wordnet.VERB)
    ]
    
    print("=" * 70)
    print("   EXPERIMENT 03: MORPHOLOGICAL ANALYSIS USING NLTK   ")
    print("=" * 70)
    
    print("\n[1] POS-AWARE LEMMATIZATION:")
    print("-" * 65)
    print(f"{'Surface Form':<15} | {'POS Tag':<10} | {'Base Lemma':<15} | {'Morphological Shift'}")
    print("-" * 65)
    
    for word, pos in words_data:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        pos_label = {wordnet.VERB: 'Verb', wordnet.NOUN: 'Noun', wordnet.ADJ: 'Adjective', wordnet.ADV: 'Adverb'}.get(pos, 'Other')
        shift = f"Inflected -> Lemma ({lemma})" if word != lemma else "Base form preserved"
        print(f"{word:<15} | {pos_label:<10} | {lemma:<15} | {shift}")
    print("-" * 65)
    
    print("\n[2] MORPHEMIC AFFIX DECOMPOSITION (Prefix / Root / Suffix):")
    print("-" * 65)
    sample_derivations = ["unhappiness", "internationalization", "disagreeable", "reconstruction", "preconditioning"]
    
    known_prefixes = ["un", "dis", "re", "pre", "inter", "anti", "mis"]
    known_suffixes = ["ness", "tion", "ation", "able", "al", "ity", "ing", "ization"]
    
    for word in sample_derivations:
        prefix, stem, suffix = "-", word, "-"
        for p in known_prefixes:
            if stem.startswith(p):
                prefix = p
                stem = stem[len(p):]
                break
        for s in known_suffixes:
            if stem.endswith(s):
                suffix = s
                stem = stem[:-len(s)]
                break
        print(f"Word: {word:<22} => Prefix: {prefix:<6} | Root: {stem:<12} | Suffix: {suffix}")
    print("=" * 70)
    print("Execution completed successfully.")

if __name__ == "__main__":
    morphological_analysis()
```

## Output Block
```text
(Refer to Lab/outputs/Exp03_Output.txt for execution log and Lab/outputs/Exp03_03_Morphological_Analysis.png for terminal screenshot)
```

## Result
Morphological analysis and POS-aware lemmatization were successfully implemented using NLTK, demonstrating accurate reduction of inflected words and morpheme decomposition.
