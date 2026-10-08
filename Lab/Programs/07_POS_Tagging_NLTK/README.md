# Experiment 07: Part-of-Speech Tagging using NLTK Library

## Aim
To write a Python program using the NLTK library to perform automated Part-of-Speech (POS) tagging on natural text and analyze tag distributions.

## Software / Tools Required
Python 3.11+, NLTK Library (`averaged_perceptron_tagger`, `punkt`).

## Theory & Syntax
Part-of-Speech (POS) Tagging is the process of marking up a word in a text corpus as corresponding to a particular part of speech (Noun, Verb, Adjective, etc.), based on both its definition and its context in the sentence.
NLTK utilizes the Penn Treebank Tagset:
- NN: Noun, singular or mass (e.g., dog, computer)
- NNS: Noun, plural (e.g., dogs, algorithms)
- NNP: Proper noun, singular (e.g., London, Google)
- VB: Verb, base form (e.g., run, eat)
- VBD: Verb, past tense (e.g., ran, ate)
- VBG: Verb, gerund or present participle (e.g., running)
- JJ: Adjective (e.g., quick, intelligent)
- RB: Adverb (e.g., quickly, silently)
- IN: Preposition or subordinating conjunction (e.g., in, on, of)
- DT: Determiner (e.g., the, a, this)

## Procedure
1. Import NLTK, `word_tokenize`, and `pos_tag`.
2. Ensure required resources (`punkt`, `averaged_perceptron_tagger`) are loaded.
3. Tokenize input sentence into discrete word units.
4. Call `nltk.pos_tag(tokens)` to generate `(word, tag)` tuples.
5. Compute summary statistics of POS category distribution.
6. Print the formatted tagged output and distribution summary.

## Program / Source Code
```python
"""
Experiment 07: Part-of-Speech Tagging using NLTK Library
Course: DSA0302 - Natural Language Processing
"""

import nltk
from collections import Counter

def ensure_resources():
    for res in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(f'taggers/{res}' if 'tagger' in res else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

def main():
    ensure_resources()
    print("=" * 75)
    print("   EXPERIMENT 07: PART-OF-SPEECH (POS) TAGGING USING NLTK   ")
    print("=" * 75)
    
    text = (
        "The quick brown fox jumps over the lazy dog while Dr. Smith "
        "carefully examines the experimental natural language models at Stanford University."
    )
    
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    tag_descriptions = {
        'DT': 'Determiner',
        'JJ': 'Adjective',
        'NN': 'Noun, singular',
        'NNS': 'Noun, plural',
        'NNP': 'Proper Noun, singular',
        'VB': 'Verb, base form',
        'VBZ': 'Verb, 3rd person sing. present',
        'VBG': 'Verb, gerund/present participle',
        'VBD': 'Verb, past tense',
        'RB': 'Adverb',
        'IN': 'Preposition/Subordinating conjunction',
        '.': 'Punctuation mark'
    }
    
    print("\n[1] INPUT SENTENCE:")
    print("-" * 75)
    print(text)
    print("-" * 75)
    
    print("\n[2] DETAILED POS TAGGING RESULTS:")
    print("-" * 75)
    print(f"{'Token':<18} | {'POS Tag':<10} | {'Grammatical Category Description'}")
    print("-" * 75)
    
    for token, tag in pos_tags:
        desc = tag_descriptions.get(tag, 'Other Part of Speech')
        print(f"{token:<18} | {tag:<10} | {desc}")
    print("-" * 75)
    
    print("\n[3] POS TAG FREQUENCY DISTRIBUTION:")
    print("-" * 45)
    tag_counts = Counter(tag for _, tag in pos_tags)
    for tag, count in tag_counts.most_common():
        print(f"    Tag: {tag:<6} ({tag_descriptions.get(tag, 'Other'):<25}) : {count}")
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp07_Output.txt for execution log and Lab/outputs/Exp07_07_POS_Tagging_NLTK.png for terminal screenshot)
```

## Result
POS tagging was successfully executed using the NLTK library, accurately classifying tokens into grammatical Penn Treebank categories.
