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
