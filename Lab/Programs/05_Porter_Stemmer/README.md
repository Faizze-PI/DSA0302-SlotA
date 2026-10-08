# Experiment 05: Word Stemming using Porter Stemmer Algorithm

## Aim
To perform word stemming using the Porter Stemmer algorithm in Python and evaluate its performance on diverse inflected word families and natural sentences.

## Software / Tools Required
Python 3.11+, NLTK Library (`PorterStemmer`).

## Theory & Syntax
Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base, or root form. Martin Porter developed the Porter Stemmer in 1980.
It operates as a cascade of five sequential condition-action rewrite steps:
- Step 1a: Plurals and 3rd person singular forms (e.g., sses -> ss, ies -> i, s -> epsilon).
- Step 1b: Past participles and progressive verbs (e.g., eed -> ee, ed/ing stripped if preceding syllable contains vowel).
- Step 1c: Y-replacement (e.g., y -> i if preceded by non-vowel).
- Steps 2 to 4: Derivational suffixes (e.g., ational -> ate, ful -> epsilon, ness -> epsilon).
- Step 5: Clean-up (stripping redundant trailing 'e' and double consonants).

## Procedure
1. Import `PorterStemmer` from `nltk.stem`.
2. Instantiate the stemmer object.
3. Prepare a comprehensive list of words exhibiting varied grammatical inflection.
4. Apply `stem()` method on each word and record the stem.
5. Tokenize and stem a full natural language paragraph.
6. Compare stemming vs lemmatization behavior on irregulars.

## Program / Source Code
```python
"""
Experiment 05: Word Stemming using Porter Stemmer Algorithm
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

def main():
    print("=" * 75)
    print("   EXPERIMENT 05: PORTER STEMMER ALGORITHM FOR WORD STEMMING   ")
    print("=" * 75)
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    words = [
        "caresses", "ponies", "ties", "cats",
        "feed", "agreed", "disabled", "matting", "mating", "meeting",
        "happy", "sky", "crying",
        "relational", "conditional", "rational", "valence", "hesitancy",
        "digitizer", "conformabli", "radically", "different",
        "universal", "university", "universe",
        "studies", "studying", "student", "studied"
    ]
    
    print("\n[1] WORD-LEVEL STEMMING WITH STEP EXPLANATIONS:")
    print("-" * 75)
    print(f"{'Original Word':<18} | {'Stemmed Form':<15} | {'Porter Rule Phase'}")
    print("-" * 75)
    
    for word in words:
        stem = stemmer.stem(word)
        phase = "Step 1: Inflectional"
        if word.endswith(("tional", "enci", "izer", "abli", "alli")):
            phase = "Step 2/3: Derivational"
        elif word.endswith(("al", "ance", "er", "ic")):
            phase = "Step 4: Suffix Stripping"
        print(f"{word:<18} | {stem:<15} | {phase}")
        
    print("-" * 75)
    
    print("\n[2] SENTENCE-LEVEL STEMMING:")
    print("-" * 75)
    sentence = "The university students were studying computational linguistics and programming algorithms generously."
    tokens = sentence.split()
    stemmed_sentence = [stemmer.stem(tok) for tok in tokens]
    
    print(f"Original Text : {sentence}")
    print(f"Stemmed Text  : {' '.join(stemmed_sentence)}")
    print("-" * 75)
    
    print("\n[3] COMPARISON: STEMMING VS LEMMATIZATION:")
    print("-" * 75)
    print(f"{'Word':<15} | {'Porter Stemmer':<18} | {'WordNet Lemmatizer'}")
    print("-" * 75)
    comparison_words = ["running", "wolves", "better", "corpora", "generously", "flies"]
    for w in comparison_words:
        print(f"{w:<15} | {stemmer.stem(w):<18} | {lemmatizer.lemmatize(w)}")
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp05_Output.txt for execution log and Lab/outputs/Exp05_05_Porter_Stemmer.png for terminal screenshot)
```

## Result
The Porter Stemmer algorithm was successfully implemented and evaluated. The multi-step heuristic suffix stripping rules effectively reduced inflected word forms to canonical stems.
