# Experiment 10: Transformation-Based Tagging (Brill's Algorithm)

## Aim
To implement transformation-based tagging (Brill's algorithm) in Python using contextual rewrite rules to iteratively correct initial baseline tagging errors.

## Software / Tools Required
Python 3.11+, Standard Library.

## Theory & Syntax
Transformation-Based Learning (TBL), introduced by Eric Brill (1995), combines the strengths of rule-based and statistical taggers.
Process:
1. Initial Baseline: Tag every word with its most likely tag in isolation (typically from unigram frequency or default NN).
2. Error Analysis: Compare current tags against gold standard tags.
3. Rule Application: Iteratively apply deterministic transformation templates:
   - Change tag from T_1 to T_2 if previous word has tag T_0.
   - Change tag from T_1 to T_2 if following word is W.
   - Change tag from T_1 to T_2 if previous word is 'to' (e.g., NN -> VB after 'to').

## Procedure
1. Initialize sentences with a baseline unigram/default tagger.
2. Define contextual transformation rules.
3. Apply transformation rules sequentially over the tagged sequence.
4. Log each transformation step, showing the initial tag, triggered rule, and updated tag.
5. Evaluate error reduction before and after rule transformations.

## Program / Source Code
```python
"""
Experiment 10: Transformation-Based Tagging (Brill's Algorithm)
Course: DSA0302 - Natural Language Processing
"""

class TransformationRule:
    def __init__(self, from_tag, to_tag, condition_type, condition_val, description):
        self.from_tag = from_tag
        self.to_tag = to_tag
        self.condition_type = condition_type
        self.condition_val = condition_val
        self.description = description

    def apply(self, words, tags, idx):
        if tags[idx] != self.from_tag:
            return False
            
        if self.condition_type == "PREV_TAG" and idx > 0:
            if tags[idx - 1] == self.condition_val:
                tags[idx] = self.to_tag
                return True
        elif self.condition_type == "PREV_WORD" and idx > 0:
            if words[idx - 1].lower() == self.condition_val.lower():
                tags[idx] = self.to_tag
                return True
        elif self.condition_type == "NEXT_TAG" and idx < len(tags) - 1:
            if tags[idx + 1] == self.condition_val:
                tags[idx] = self.to_tag
                return True
        return False

class BrillTagger:
    def __init__(self):
        # Baseline lexicon mapping (most common isolated tag)
        self.baseline_lexicon = {
            "the": "DT", "a": "DT", "an": "DT",
            "to": "TO", "in": "IN", "on": "IN",
            "he": "PRP", "she": "PRP", "they": "PRP",
            "race": "NN",    # Ambiguous: noun vs verb
            "run": "VB",
            "fast": "JJ",    # Ambiguous: adj vs adv
            "like": "IN",    # Ambiguous: prep vs verb
            "fish": "NN",
            "can": "NN"
        }
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def initial_tag(self, words):
        return [self.baseline_lexicon.get(w.lower(), "NN") for w in words]

    def transform(self, words):
        tags = self.initial_tag(words)
        initial_tags = list(tags)
        transform_log = []

        for rule in self.rules:
            for i in range(len(words)):
                prev_tag = tags[i]
                if rule.apply(words, tags, i):
                    transform_log.append((i, words[i], prev_tag, tags[i], rule.description))
                    
        return initial_tags, tags, transform_log

def main():
    print("=" * 75)
    print("   EXPERIMENT 10: TRANSFORMATION-BASED TAGGING (BRILL'S ALGORITHM)   ")
    print("=" * 75)
    
    tagger = BrillTagger()
    
    # Adding transformation rules
    tagger.add_rule(TransformationRule("NN", "VB", "PREV_TAG", "TO", "Change NN to VB if preceded by TO"))
    tagger.add_rule(TransformationRule("NN", "MD", "PREV_TAG", "PRP", "Change NN to MD if preceded by PRP"))
    tagger.add_rule(TransformationRule("JJ", "RB", "PREV_TAG", "VB", "Change JJ to RB if preceded by VB"))
    tagger.add_rule(TransformationRule("IN", "VB", "PREV_TAG", "PRP", "Change IN to VB if preceded by PRP"))
    
    test_sentences = [
        "They want to race in the park",
        "They can run fast",
        "They like to fish"
    ]
    
    for sentence in test_sentences:
        words = sentence.split()
        initial_tags, final_tags, logs = tagger.transform(words)
        
        print(f"\nInput Sentence : '{sentence}'")
        print(f"Initial Tagging: {' '.join([f'{w}/{t}' for w, t in zip(words, initial_tags)])}")
        
        if logs:
            print("Transformations Applied:")
            for idx, word, from_t, to_t, desc in logs:
                print(f"  - Word '{word}' at pos {idx}: {from_t} -> {to_t} ({desc})")
        else:
            print("Transformations Applied: None (Initial baseline was retained)")
            
        print(f"Final Tagging  : {' '.join([f'{w}/{t}' for w, t in zip(words, final_tags)])}")
        print("-" * 75)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp10_Output.txt for execution log and Lab/outputs/Exp10_10_Transformation_Based_Tagger.png for terminal screenshot)
```

## Result
Transformation-based tagging was successfully implemented, demonstrating iterative correction of baseline POS errors using learned contextual rules.
