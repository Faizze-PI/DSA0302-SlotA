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
