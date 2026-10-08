# Experiment 19: Word Sense Disambiguation using Lesk Algorithm

## Aim
To create a Python program for Word Sense Disambiguation (WSD) using the Simplified Lesk algorithm and evaluate sense selection against context.

## Software / Tools Required
Python 3.11+, NLTK Library (`nltk.wsd.lesk`, `nltk.corpus.wordnet`).

## Theory & Syntax
Word Sense Disambiguation (WSD) is the task of determining which sense of a polysemous word is activated in a given context.
Michael Lesk (1986) introduced the overlap hypothesis: A word's intended sense shares vocabulary words with the context words in the sentence.
Simplified Lesk Algorithm:
1. Given target word w and sentence context C.
2. For each synset s of w in WordNet:
   - Signature(s) = Words in definition (gloss) of s union Words in example sentences of s.
   - Overlap(s) = |Signature(s) intersect C (excluding stopwords)|.
3. Return sense s* that maximizes Overlap(s).

## Procedure
1. Import `lesk` from `nltk.wsd` and `wordnet` from `nltk.corpus`.
2. Implement a custom Simplified Lesk algorithm from first principles.
3. Tokenize and preprocess the sentence (removing punctuation and stopwords).
4. Calculate the bag-of-words lexical overlap for every dictionary definition.
5. Select the synset with highest overlap.
6. Compare custom implementation against `nltk.wsd.lesk` across distinct contexts.

## Program / Source Code
```python
"""
Experiment 19: Word Sense Disambiguation using Lesk Algorithm
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

def ensure_resources():
    for res in ['wordnet', 'omw-1.4', 'stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{res}' if 'word' in res or 'stop' in res or 'omw' in res else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

def simplified_lesk(word, sentence):
    """Custom Implementation of the Simplified Lesk Algorithm"""
    stopwords = set(nltk.corpus.stopwords.words('english'))
    context = set(w.lower() for w in nltk.word_tokenize(sentence) if w.isalnum()) - stopwords
    
    best_sense = None
    max_overlap = -1
    overlap_details = []
    
    for synset in wn.synsets(word):
        # Build signature from definition and examples
        sig_words = set(w.lower() for w in nltk.word_tokenize(synset.definition()) if w.isalnum())
        for ex in synset.examples():
            sig_words.update(w.lower() for w in nltk.word_tokenize(ex) if w.isalnum())
        sig_words = sig_words - stopwords
        
        # Compute intersection overlap
        shared = context.intersection(sig_words)
        overlap = len(shared)
        overlap_details.append((synset, overlap, shared))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset
            
    return best_sense, max_overlap, overlap_details

def main():
    ensure_resources()
    print("=" * 75)
    print("   EXPERIMENT 19: WORD SENSE DISAMBIGUATION (LESK ALGORITHM)   ")
    print("=" * 75)
    
    test_cases = [
        ("bank", "The financial bank deposited money into the customer account and approved the loan.", "Financial Institution"),
        ("bank", "The fisherman sat on the muddy river bank waiting for fish in the water.", "River Bank / Slope"),
        ("bass", "The musician tuned the electric bass guitar before the rock concert.", "Musical Instrument"),
        ("bass", "The fresh sea bass fish was served with lemon and garlic.", "Fish Species")
    ]
    
    for word, sentence, expected_context in test_cases:
        best_sense, max_overlap, details = simplified_lesk(word, sentence)
        nltk_sense = lesk(nltk.word_tokenize(sentence), word)
        
        print(f"\n[*] Target Word   : '{word}' (Expected: {expected_context})")
        print(f"    Sentence      : \"{sentence}\"")
        print(f"    Disambiguated : {best_sense.name() if best_sense else 'None'}")
        print(f"    Gloss/Meaning : {best_sense.definition() if best_sense else 'N/A'}")
        print(f"    Overlap Count : {max_overlap}")
        print(f"    NLTK lesk()   : {nltk_sense.name() if nltk_sense else 'None'}")
        print("-" * 75)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp19_Output.txt for execution log and Lab/outputs/Exp19_19_Lesk_WSD.png for terminal screenshot)
```

## Result
Word Sense Disambiguation using the Lesk algorithm was successfully implemented, effectively selecting the appropriate semantic sense based on contextual vocabulary overlap.
