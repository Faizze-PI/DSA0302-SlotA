"""
Experiment 23: Text Coherence and Discourse Cohesion Evaluation
Course: DSA0302 - Natural Language Processing
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_coherence(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 3]
    if len(sentences) < 2:
        return 1.0, [], 0
        
    # 1. TF-IDF Vectors per sentence
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform(sentences)
        pairwise_sims = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(tfidf[i], tfidf[i+1])[0][0]
            pairwise_sims.append(sim)
    except ValueError:
        pairwise_sims = [0.0] * (len(sentences) - 1)
        
    # 2. Discourse Connectives Detection
    connectives = {"however", "therefore", "furthermore", "moreover", "consequently", "thus", "specifically", "in addition"}
    found_connectives = []
    for s in sentences:
        words = set(s.lower().split())
        shared = words.intersection(connectives)
        if shared:
            found_connectives.extend(list(shared))
            
    # 3. Overall Coherence Index
    mean_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0
    connective_bonus = min(0.2, len(found_connectives) * 0.05)
    composite_score = min(1.0, mean_sim * 0.8 + connective_bonus)
    
    return composite_score, pairwise_sims, found_connectives

def main():
    print("=" * 75)
    print("   EXPERIMENT 23: TEXT COHERENCE AND DISCOURSE COHESION EVALUATION   ")
    print("=" * 75)
    
    coherent_text = (
        "Machine learning is a subset of artificial intelligence that focuses on data. "
        "These learning algorithms build mathematical models to make predictions. "
        "Furthermore, deep learning models extend these algorithms using multi-layered neural networks. "
        "Consequently, neural networks have achieved state-of-the-art results across speech and vision tasks."
    )
    
    incoherent_text = (
        "Machine learning is a subset of artificial intelligence that focuses on data. "
        "The recipe requires two cups of flour and fresh strawberries. "
        "Penguins live predominantly in Antarctica and love swimming in freezing oceans. "
        "My bicycle tire was punctured yesterday morning."
    )
    
    passages = [
        ("Coherent Technical Text", coherent_text),
        ("Incoherent Jumbled Text", incoherent_text)
    ]
    
    for title, passage in passages:
        print(f"\n[EVALUATION]: {title}")
        print("-" * 75)
        print("Passage Content:")
        print(passage)
        print()
        
        score, sims, markers = evaluate_coherence(passage)
        print(f"Consecutive Sentence Similarities: {[round(s, 4) for s in sims]}")
        print(f"Discourse Connectives Found     : {markers}")
        print(f"Composite Coherence Index (0-1) : {score:.4f}")
        
        verdict = "HIGHLY COHERENT" if score > 0.45 else "POOR / INCOHERENT"
        print(f"Qualitative Assessment           : {verdict}")
        print("-" * 75)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
