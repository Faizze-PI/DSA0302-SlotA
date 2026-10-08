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
