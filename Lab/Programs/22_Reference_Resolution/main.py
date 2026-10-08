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
        print(f"{pron:<10} | {ant:<25} | {loc}")
    print("-" * 75)
    
    print("\n[3] DISAMBIGUATED RESOLVED TEXT:")
    print("-" * 75)
    print(resolved_text)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
