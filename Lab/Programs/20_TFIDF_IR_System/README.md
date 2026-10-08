# Experiment 20: Information Retrieval System using TF-IDF for Document Ranking

## Aim
To implement a complete Information Retrieval (IR) system in Python using TF-IDF (Term Frequency-Inverse Document Frequency) weighting and Cosine Similarity for ranking documents.

## Software / Tools Required
Python 3.11+, Scikit-Learn (`TfidfVectorizer`), NumPy.

## Theory & Syntax
Information Retrieval (IR) retrieves relevant documents from a collection in response to a user query.
TF-IDF Formula:
- Term Frequency: TF(t, d) = count(t, d) / total_terms(d)
- Inverse Document Frequency: IDF(t, D) = log( (1 + |D|) / (1 + count(d in D : t in d)) ) + 1
- TF-IDF Weight: TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
Cosine Similarity measures the cosine of the angle between query vector Q and document vector D:
  Cosine_Sim(Q, D) = (Q . D) / (||Q|| * ||D||)
Documents are ranked in descending order of similarity score.

## Procedure
1. Prepare a corpus of text documents stored in a `documents/` directory.
2. Ingest and tokenize documents, stripping punctuation and lowercase normalizing.
3. Fit a `TfidfVectorizer` to compute the document-term matrix.
4. Transform input queries into the shared TF-IDF vector space.
5. Compute cosine similarity scores between the query vector and all document vectors.
6. Rank documents by relevance score and display the top results.

## Program / Source Code
```python
"""
Experiment 20: Information Retrieval System using TF-IDF for Document Ranking
Course: DSA0302 - Natural Language Processing
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents():
    docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    documents = {}
    
    if os.path.exists(docs_dir):
        for fname in sorted(os.listdir(docs_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                    documents[fname] = f.read().strip()
                    
    if not documents:
        # Fallback default documents
        documents = {
            "doc1_nlp.txt": "Natural language processing enables computers to understand human language and text data.",
            "doc2_cv.txt": "Computer vision models analyze images, pixels, video streams, and visual data patterns.",
            "doc3_ai.txt": "Artificial intelligence algorithms power modern natural language processing and computer vision systems.",
            "doc4_ir.txt": "Information retrieval systems rank relevant documents based on TF-IDF term weights and search queries.",
            "doc5_ml.txt": "Machine learning algorithms learn statistical predictive patterns from training datasets."
        }
    return documents

def main():
    print("=" * 75)
    print("   EXPERIMENT 20: INFORMATION RETRIEVAL USING TF-IDF & COSINE RANKING   ")
    print("=" * 75)
    
    docs_dict = load_documents()
    doc_names = list(docs_dict.keys())
    corpus = list(docs_dict.values())
    
    print(f"\n[1] LOADED DOCUMENT COLLECTION (Total: {len(corpus)} documents):")
    print("-" * 75)
    for name, content in docs_dict.items():
        print(f"[{name}]: {content[:75]}...")
    print("-" * 75)
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n[2] VOCABULARY SIZE: {len(feature_names)} unique terms.")
    
    # User Search Queries to test
    queries = [
        "natural language understanding text",
        "computer vision and image processing",
        "search query information retrieval ranking"
    ]
    
    print("\n[3] QUERY SEARCH & DOCUMENT RANKING RESULTS:")
    print("=" * 75)
    
    for q_idx, query in enumerate(queries, 1):
        print(f"\n>>> QUERY {q_idx}: \"{query}\"")
        print("-" * 65)
        
        # Transform query into TF-IDF vector
        query_vec = vectorizer.transform([query])
        
        # Compute cosine similarities
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Rank document indices in descending order
        ranked_indices = np.argsort(scores)[::-1]
        
        print(f"{'Rank':<6} | {'Relevance Score':<16} | {'Document File':<20} | {'Snippet'}")
        print("-" * 65)
        for rank, doc_idx in enumerate(ranked_indices, 1):
            score = scores[doc_idx]
            doc_name = doc_names[doc_idx]
            snippet = corpus[doc_idx][:45] + "..."
            print(f"{rank:<6} | {score:<16.4f} | {doc_name:<20} | {snippet}")
        print("-" * 65)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp20_Output.txt for execution log and Lab/outputs/Exp20_20_TFIDF_IR_System.png for terminal screenshot)
```

## Result
The TF-IDF based Information Retrieval system was successfully implemented, correctly transforming documents into term vectors and ranking documents against search queries by cosine similarity.
