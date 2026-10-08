"""
Experiment 09: Rule-Based Part-of-Speech Tagging using Regular Expressions
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.tag import RegexpTagger

def build_rule_tagger():
    # Sequential rule definitions (ordered from most specific to default)
    patterns = [
        (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),              # Cardinal numbers
        (r'^(The|the|A|a|An|an)$', 'DT'),               # Determiners
        (r'^(In|in|On|on|At|at|By|by|With|with)$', 'IN'),# Prepositions
        (r'^(And|and|Or|or|But|but)$', 'CC'),           # Coordinating conjunctions
        (r'^(I|he|she|they|we|you|it)$', 'PRP'),        # Personal pronouns
        (r'^(can|could|may|might|will|would|must|shall|should)$', 'MD'), # Modals
        (r'.*ing$', 'VBG'),                             # Gerunds / Present participles
        (r'.*ed$', 'VBD'),                              # Simple past / Past participle
        (r'.*es$', 'VBZ'),                              # 3rd person singular present
        (r'.*ould$', 'MD'),                             # Modals (would, could, should)
        (r".*'s$", 'NN$'),                             # Possessive nouns
        (r'.*s$', 'NNS'),                               # Plural nouns
        (r'.*ly$', 'RB'),                               # Adverbs
        (r'.*(able|ful|ous|ive|less|ic)$', 'JJ'),       # Adjectives
        (r'.*ment$', 'NN'),                             # Nominalizations
        (r'.*tion$', 'NN'),                             # Abstract nouns
        (r'.*', 'NN')                                   # Default fallback: Noun
    ]
    return RegexpTagger(patterns)

def main():
    print("=" * 75)
    print("   EXPERIMENT 09: RULE-BASED POS TAGGING USING REGULAR EXPRESSIONS   ")
    print("=" * 75)
    
    tagger = build_rule_tagger()
    
    test_sentence = (
        "The quick robotic system carefully processed 42 items and happily "
        "completed the experimental verification without any complication"
    )
    
    tokens = test_sentence.split()
    tagged_output = tagger.tag(tokens)
    
    print("\n[1] INPUT SENTENCE:")
    print("-" * 75)
    print(test_sentence)
    print("-" * 75)
    
    print("\n[2] RULE-BASED TAGGING RESULTS:")
    print("-" * 75)
    print(f"{'Word':<18} | {'Assigned Tag':<14} | {'Morphological Rule Trigger'}")
    print("-" * 75)
    
    for word, tag in tagged_output:
        reason = "Default Noun Rule"
        if word.lower() in ["the", "a", "an"]:
            reason = "Determiner Regex"
        elif word.endswith("ing"):
            reason = "Suffix -ing (Gerund/Participle)"
        elif word.endswith("ed"):
            reason = "Suffix -ed (Past Tense)"
        elif word.endswith("ly"):
            reason = "Suffix -ly (Adverb)"
        elif word.endswith(("ic", "ful", "able")):
            reason = "Adjectival Suffix"
        elif word.endswith("tion"):
            reason = "Noun Suffix -tion"
        elif word.endswith("s") and not word.endswith("ss"):
            reason = "Plural Noun / 3rd Person"
        elif word.isdigit():
            reason = "Numeric Cardinal Digit"
        print(f"{word:<18} | {tag:<14} | {reason}")
    print("-" * 75)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
