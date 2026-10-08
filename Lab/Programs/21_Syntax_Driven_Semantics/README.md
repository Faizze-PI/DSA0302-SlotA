# Experiment 21: Syntax-Driven Semantic Analysis (Noun Phrase Meaning Extraction)

## Aim
To create a Python program that performs syntax-driven semantic analysis by chunking noun phrases and extracting their semantic meanings using WordNet.

## Software / Tools Required
Python 3.11+, NLTK Library (`nltk.chunk.RegexpParser`, `nltk.corpus.wordnet`).

## Theory & Syntax
Syntax-driven semantic analysis operates on the Principle of Compositionality (Frege's Principle): The meaning of a complex syntactic expression is a function of the meanings of its syntactic parts and the rules used to combine them.
Methodology:
1. Part-of-Speech Tagging: Tokenize and assign POS tags to sentence constituents.
2. Noun Phrase (NP) Chunking: Use regular expression chunk grammar NP: {<DT>?<JJ.*>*<NN.*>+} to delineate noun phrases.
3. Syntactic Head Identification: The rightmost noun in the NP is typically the syntactic head.
4. Semantic Projection: Query WordNet for the head noun's core concept, definition, and hypernym ontology.

## Procedure
1. Import `word_tokenize`, `pos_tag`, `RegexpParser` and `wordnet` from NLTK.
2. Define an NP chunking grammar.
3. Parse sentences into constituency trees.
4. Extract all NP subtrees.
5. Separate modifiers (determiners, adjectives) from the head noun.
6. Retrieve and map the semantic definition from WordNet for the extracted concept.

## Program / Source Code
```python
"""
Experiment 21: Syntax-Driven Semantic Analysis (Noun Phrase Meaning Extraction)
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.corpus import wordnet as wn

def ensure_resources():
    for res in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{res}' if 'wordnet' in res or 'omw' in res else f'tokenizers/{res}' if 'punkt' in res else f'taggers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

def extract_np_semantics(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    # Chunk grammar: Determiner (optional) + Adjectives (0 or more) + Nouns (1 or more)
    chunk_grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
    chunk_parser = RegexpParser(chunk_grammar)
    chunk_tree = chunk_parser.parse(tagged)
    
    np_results = []
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
        words_tags = subtree.leaves()
        np_text = " ".join(w for w, _ in words_tags)
        
        # Identify head noun (last noun token) and modifiers
        nouns = [w for w, t in words_tags if t.startswith('NN')]
        head_noun = nouns[-1] if nouns else words_tags[-1][0]
        modifiers = [w for w, t in words_tags if w != head_noun]
        
        # Semantic Lookup in WordNet
        synsets = wn.synsets(head_noun.lower(), pos=wn.NOUN)
        if synsets:
            primary_syn = synsets[0]
            definition = primary_syn.definition()
            hypernyms = [h.name().split('.')[0] for h in primary_syn.hypernyms()]
            hypernym_str = ", ".join(hypernyms) if hypernyms else "entity"
        else:
            definition = "Lexical concept not found in WordNet"
            hypernym_str = "unknown"
            
        np_results.append({
            "np_phrase": np_text,
            "head_noun": head_noun,
            "modifiers": modifiers,
            "semantic_gloss": definition,
            "hypernym_category": hypernym_str
        })
        
    return chunk_tree, np_results

def main():
    ensure_resources()
    print("=" * 75)
    print("   EXPERIMENT 21: SYNTAX-DRIVEN SEMANTIC ANALYSIS OF NOUN PHRASES   ")
    print("=" * 75)
    
    sentences = [
        "The intelligent robot analyzed the complex medical data.",
        "A fierce tiger hunted in the dense tropical forest."
    ]
    
    for idx, sentence in enumerate(sentences, 1):
        print(f"\n[SENTENCE {idx}]: \"{sentence}\"")
        print("-" * 75)
        
        tree, np_data = extract_np_semantics(sentence)
        print("Syntactic Chunk Tree:")
        print(tree)
        print("\nExtracted Noun Phrase Semantics:")
        print(f"{'Noun Phrase':<22} | {'Head Noun':<10} | {'Modifiers':<14} | {'Semantic Category & Meaning'}")
        print("-" * 75)
        for item in np_data:
            mods = ", ".join(item['modifiers']) if item['modifiers'] else "-"
            meaning = f"Category: [{item['hypernym_category']}] - {item['semantic_gloss'][:40]}..."
            print(f"{item['np_phrase']:<22} | {item['head_noun']:<10} | {mods:<14} | {meaning}")
        print("-" * 75)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp21_Output.txt for execution log and Lab/outputs/Exp21_21_Syntax_Driven_Semantics.png for terminal screenshot)
```

## Result
Syntax-driven semantic analysis was successfully implemented, demonstrating noun phrase chunking, syntactic head identification, and compositional semantic mapping via WordNet.
