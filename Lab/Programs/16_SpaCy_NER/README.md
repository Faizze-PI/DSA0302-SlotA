# Experiment 16: Named Entity Recognition (NER) using SpaCy

## Aim
To implement Named Entity Recognition (NER) in Python using the SpaCy library to identify and classify named entities (persons, organizations, locations, dates, monetary amounts) in unstructured text.

## Software / Tools Required
Python 3.11+, SpaCy Library (`en_core_web_sm` model).

## Theory & Syntax
Named Entity Recognition (NER) is an information extraction subtask that locates and categorizes named entities in text into predefined categories:
- PERSON: People, including fictional characters (e.g., Alan Turing, Marie Curie)
- ORG: Companies, agencies, institutions (e.g., Google, Stanford University)
- GPE: Geopolitical entities: countries, cities, states (e.g., London, Tokyo)
- LOC: Non-GPE locations, mountain ranges, bodies of water
- DATE: Absolute or relative dates or periods
- MONEY: Monetary values including unit
- CARDINAL: Numerals that do not fall under other types
SpaCy uses convolutional/transformer transition-based neural architectures to predict entity spans and IOB (Inside-Outside-Beginning) tags.

## Procedure
1. Import SpaCy and load the trained English pipeline `en_core_web_sm`.
2. Define sample sentences containing diverse named entity classes.
3. Process the text through the SpaCy language pipeline `nlp(text)`.
4. Iterate over `doc.ents` to extract entity text, label, character offsets, and explanation via `spacy.explain()`.
5. Output an entity table and IOB tag sequence.

## Program / Source Code
```python
"""
Experiment 16: Named Entity Recognition (NER) using SpaCy
Course: DSA0302 - Natural Language Processing
"""

import spacy

def main():
    print("=" * 75)
    print("   EXPERIMENT 16: NAMED ENTITY RECOGNITION (NER) USING SPACY   ")
    print("=" * 75)
    
    # Load SpaCy small English language model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[!] Model 'en_core_web_sm' not found. Installing via spacy...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        
    text = (
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California. "
        "On January 9, 2007, the company unveiled the first iPhone in San Francisco. "
        "In 2024, Tim Cook announced an investment of $10 billion into AI research centers across Germany and Japan."
    )
    
    print("\n[1] INPUT TEXT:")
    print("-" * 75)
    print(text)
    print("-" * 75)
    
    doc = nlp(text)
    
    print("\n[2] EXTRACTED NAMED ENTITIES:")
    print("-" * 75)
    print(f"{'Entity Text':<22} | {'Label':<10} | {'Char Range':<12} | {'Entity Description'}")
    print("-" * 75)
    
    for ent in doc.ents:
        desc = spacy.explain(ent.label_)
        span_str = f"[{ent.start_char}:{ent.end_char}]"
        print(f"{ent.text:<22} | {ent.label_:<10} | {span_str:<12} | {desc}")
    print("-" * 75)
    
    print("\n[3] TOKEN-LEVEL IOB (INSIDE-OUTSIDE-BEGINNING) TAGGING:")
    print("-" * 55)
    print(f"{'Token':<18} | {'POS':<8} | {'IOB Entity Tag'}")
    print("-" * 55)
    for token in list(doc)[:15]:
        iob_tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else token.ent_iob_
        print(f"{token.text:<18} | {token.pos_:<8} | {iob_tag}")
    print("-" * 55)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp16_Output.txt for execution log and Lab/outputs/Exp16_16_SpaCy_NER.png for terminal screenshot)
```

## Result
Named Entity Recognition was successfully implemented using SpaCy, correctly identifying organizations, persons, dates, geopolitical entities, and monetary figures.
