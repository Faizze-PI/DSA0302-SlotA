# Experiment 22: Reference and Anaphora Resolution within Text

## Aim
To create a Python program that performs reference / pronominal anaphora resolution to link pronouns to their corresponding antecedent entities.

## Software / Tools Required
Python 3.11+, Standard Library `re`, `collections`.

## Theory & Syntax
Reference Resolution is the computational task of determining whether two or more expressions in a discourse refer to the same entity. Anaphora refers to the phenomenon where the interpretation of a pronoun depends on an antecedent expression.
Constraints:
- Gender Agreement: 'he' -> masculine, 'she' -> feminine, 'it' -> neuter.
- Number Agreement: 'they' -> plural, 'he/she/it' -> singular.
- Recency & Salience (Hobbs' Algorithm Heuristics): Entities mentioned more recently and in the subject position have higher salience.
Algorithm:
1. Maintain a discourse entity table tracking (Name, Gender, Number, SentenceIndex).
2. When encountering a pronoun, search backward for the most salient entity matching grammatical features.
3. Replace the anaphoric pronoun with the resolved antecedent.

## Procedure
1. Define discourse entities with annotated gender and grammatical number.
2. Parse text into individual sentences and tokens.
3. Scan for pronominal mentions ('he', 'she', 'it', 'they', 'his', 'her', 'its').
4. Perform backwards search to identify the most recent compatible antecedent.
5. Log the resolution decision and substitute referents into the text.
6. Display the original and resolved disambiguated discourse.

## Program / Source Code
```python
"""
Experiment 22: Reference and Anaphora Resolution within Text
Course: DSA0302 - Natural Language Processing
"""

import re

class ReferenceResolver:
    def __init__(self):
        # Lexical Knowledge Base of Gender and Number
        self.entity_features = {
            "dr. alan turing": ("MASC", "SG"),
            "alan turing": ("MASC", "SG"),
            "dr. turing": ("MASC", "SG"),
            "marie curie": ("FEM", "SG"),
            "marie": ("FEM", "SG"),
            "alice": ("FEM", "SG"),
            "the robot": ("NEUT", "SG"),
            "the computer": ("NEUT", "SG"),
            "the algorithm": ("NEUT", "SG"),
            "the book": ("NEUT", "SG"),
            "the students": ("ANY", "PL"),
            "the researchers": ("ANY", "PL")
        }
        
        self.pronoun_filters = {
            "he": ("MASC", "SG"),
            "him": ("MASC", "SG"),
            "his": ("MASC", "SG"),
            "she": ("FEM", "SG"),
            "her": ("FEM", "SG"),
            "hers": ("FEM", "SG"),
            "it": ("NEUT", "SG"),
            "its": ("NEUT", "SG"),
            "they": ("ANY", "PL"),
            "them": ("ANY", "PL"),
            "their": ("ANY", "PL")
        }

    def resolve(self, text):
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        discourse_history = []  # List of (entity_name, gender, number, sent_idx)
        resolutions = []
        resolved_sentences = []

        for sent_idx, sentence in enumerate(sentences):
            words = sentence.split()
            resolved_words = []
            
            # Step 1: Detect explicit entities in current sentence
            sent_lower = sentence.lower()
            for entity, (gen, num) in self.entity_features.items():
                if entity in sent_lower:
                    discourse_history.append((entity.title(), gen, num, sent_idx))
                    
            # Step 2: Scan for pronouns
            for word in words:
                clean_w = re.sub(r'[^a-zA-Z]', '', word).lower()
                if clean_w in self.pronoun_filters:
                    target_gen, target_num = self.pronoun_filters[clean_w]
                    
                    # Backward search for most recent compatible antecedent
                    matched_antecedent = None
                    for ent_name, ent_gen, ent_num, _ in reversed(discourse_history):
                        if (target_gen == "ANY" or ent_gen == "ANY" or target_gen == ent_gen) and (target_num == ent_num):
                            matched_antecedent = ent_name
                            break
                            
                    if matched_antecedent:
                        resolutions.append((clean_w, matched_antecedent, f"Sentence {sent_idx+1}"))
                        # Replace pronoun with [Antecedent]
                        replacement = f"[{matched_antecedent}]"
                        if word.endswith(','):
                            replacement += ','
                        resolved_words.append(replacement)
                    else:
                        resolved_words.append(word)
                else:
                    resolved_words.append(word)
                    
            resolved_sentences.append(" ".join(resolved_words) + ".")

        return " ".join(resolved_sentences), resolutions

def main():
    print("=" * 75)
    print("   EXPERIMENT 22: REFERENCE AND ANAPHORA RESOLUTION WITHIN TEXT   ")
    print("=" * 75)
    
    sample_text = (
        "Dr. Alan Turing was a pioneer in computing. He designed the Turing Machine. "
        "Marie Curie was a renowned physicist. She discovered radium and polonium. "
        "The computer processed the dataset. It completed the task without error. "
        "The students attended the seminar. They asked insightful questions."
    )
    
    print("\n[1] INPUT DISCOURSE (CONTAINING ANAPHORIC PRONOUNS):")
    print("-" * 75)
    print(sample_text)
    print("-" * 75)
    
    resolver = ReferenceResolver()
    resolved_text, resolutions = resolver.resolve(sample_text)
    
    print("\n[2] ANAPHORA RESOLUTION LOG:")
    print("-" * 75)
    print(f"{'Pronoun':<10} | {'Resolved Antecedent Entity':<25} | {'Location Context'}")
    print("-" * 75)
    for pron, ant, loc in resolutions:
        print(f"'{pron}'{space:<6} | {ant:<25} | {loc}", space=" "*(4-len(pron)))
    print("-" * 75)
    
    print("\n[3] DISAMBIGUATED RESOLVED TEXT:")
    print("-" * 75)
    print(resolved_text)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp22_Output.txt for execution log and Lab/outputs/Exp22_22_Reference_Resolution.png for terminal screenshot)
```

## Result
Reference and anaphora resolution was successfully implemented, linking pronominal references to their correct antecedent entities using linguistic gender, number, and recency constraints.
