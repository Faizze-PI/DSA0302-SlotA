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
