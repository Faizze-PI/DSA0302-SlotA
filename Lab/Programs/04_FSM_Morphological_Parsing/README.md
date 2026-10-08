# Experiment 04: Finite-State Machine for Morphological Parsing (Pluralization)

## Aim
To implement a Finite-State Machine (FSM) / Transducer in Python to generate and parse plural forms of English nouns following standard morphological orthographic rules.

## Software / Tools Required
Python 3.11+, Standard Library.

## Theory & Syntax
Morphological parsing involves recognizing the grammatical properties of a word from its constituent morphemes. English noun pluralization follows structured rules modelled as a finite-state machine with rule-based state transitions:
1. Irregular Nouns: Exact lexical mapping (child -> children, man -> men, foot -> feet, mouse -> mice).
2. Sibilant Endings (-s, -x, -z, -ch, -sh): Append '-es' (bus -> buses, box -> boxes, match -> matches).
3. Consonant + '-y': Drop 'y' and append '-ies' (city -> cities, baby -> babies).
4. Vowel + '-y': Append '-s' (boy -> boys, day -> days).
5. Ending in '-f' or '-fe': Replace with '-ves' (leaf -> leaves, knife -> knives).
6. Ending in Consonant + '-o': Frequently append '-es' (hero -> heroes, potato -> potatoes).
7. Default Rule: Append '-s' (book -> books, cat -> cats).

## Procedure
1. Initialize the morphological parsing engine with irregular lexicon tables.
2. Design state transitions for suffix examination.
3. Implement conditional transducers testing for phonetic and spelling characteristics.
4. Apply morphological transformation corresponding to the matched transition.
5. Output input singular noun, applied morphological rule, and output plural form.
6. Test on regular, phonetic-edge, and irregular nouns.

## Program / Source Code
```python
"""
Experiment 04: Finite-State Machine for Morphological Parsing (Pluralization)
Course: DSA0302 - Natural Language Processing
"""

class MorphologicalPluralFSM:
    def __init__(self):
        self.irregulars = {
            "child": "children",
            "man": "men",
            "woman": "women",
            "foot": "feet",
            "tooth": "teeth",
            "goose": "geese",
            "mouse": "mice",
            "ox": "oxen",
            "person": "people",
            "sheep": "sheep",
            "deer": "deer",
            "fish": "fish"
        }
        self.vowels = set("aeiou")

    def parse_plural(self, noun):
        noun_clean = noun.strip().lower()
        
        # State 1: Check Irregular Lexicon
        if noun_clean in self.irregulars:
            return self.irregulars[noun_clean], "Rule 1: Irregular Lexicon Lookup"
            
        # State 2: Sibilants (-s, -sh, -ch, -x, -z)
        if (noun_clean.endswith(('s', 'x', 'z')) or 
            noun_clean.endswith('sh') or 
            noun_clean.endswith('ch')):
            return noun_clean + "es", "Rule 2: Sibilant Ending -> Append -es"
            
        # State 3: Consonant + y -> -ies
        if noun_clean.endswith('y') and len(noun_clean) > 1:
            if noun_clean[-2] not in self.vowels:
                return noun_clean[:-1] + "ies", "Rule 3: Consonant + 'y' -> Replace with -ies"
            else:
                return noun_clean + "s", "Rule 4: Vowel + 'y' -> Append -s"
                
        # State 4: Words ending in -fe or -f -> -ves
        fe_exceptions = {"roof": "roofs", "cliff": "cliffs", "belief": "beliefs", "chef": "chefs"}
        if noun_clean in fe_exceptions:
            return fe_exceptions[noun_clean], "Rule 5a: -f Exception -> Append -s"
        if noun_clean.endswith('fe'):
            return noun_clean[:-2] + "ves", "Rule 5b: Ending in -fe -> Replace with -ves"
        elif noun_clean.endswith('f') and not noun_clean.endswith('ff'):
            return noun_clean[:-1] + "ves", "Rule 5c: Ending in -f -> Replace with -ves"
            
        # State 5: Ending in Consonant + o -> -es
        if noun_clean.endswith('o') and len(noun_clean) > 1:
            if noun_clean[-2] not in self.vowels and noun_clean not in {"photo", "piano", "halo"}:
                return noun_clean + "es", "Rule 6: Consonant + 'o' -> Append -es"
                
        # State 6: Default Transduction
        return noun_clean + "s", "Rule 7: Default -> Append -s"

def main():
    print("=" * 75)
    print("   EXPERIMENT 04: FINITE-STATE MACHINE FOR MORPHOLOGICAL PARSING   ")
    print("=" * 75)
    
    fsm = MorphologicalPluralFSM()
    
    test_nouns = [
        "cat", "dog", "bus", "box", "watch", "dish", "buzz",
        "baby", "city", "boy", "day", "leaf", "knife", "wolf", "roof",
        "potato", "hero", "photo", "child", "man", "mouse", "foot", "sheep"
    ]
    
    print(f"\n{'Singular Noun':<15} | {'Plural Form':<15} | {'Morphological Rule Applied'}")
    print("-" * 75)
    for noun in test_nouns:
        plural, rule = fsm.parse_plural(noun)
        print(f"{noun:<15} | {plural:<15} | {rule}")
    print("-" * 75)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp04_Output.txt for execution log and Lab/outputs/Exp04_04_FSM_Morphological_Parsing.png for terminal screenshot)
```

## Result
The finite-state morphological parser for English noun pluralization was successfully implemented and demonstrated accurate transduction across all phonetic and grammatical cases.
