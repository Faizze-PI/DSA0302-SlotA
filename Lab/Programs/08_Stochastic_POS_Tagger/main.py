"""
Experiment 08: Stochastic Part-of-Speech Tagging Algorithm
Course: DSA0302 - Natural Language Processing
"""

from collections import defaultdict, Counter

class StochasticPOSTagger:
    def __init__(self):
        self.emission_counts = defaultdict(Counter)  # tag -> word -> count
        self.transition_counts = defaultdict(Counter) # prev_tag -> curr_tag -> count
        self.tag_counts = Counter()
        self.word_tag_counts = defaultdict(Counter)   # word -> tag -> count

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            prev_tag = "<START>"
            self.tag_counts[prev_tag] += 1
            for word, tag in sentence:
                word_lower = word.lower()
                self.emission_counts[tag][word_lower] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.tag_counts[tag] += 1
                self.word_tag_counts[word_lower][tag] += 1
                prev_tag = tag
            self.transition_counts[prev_tag]["<END>"] += 1

    def tag_unigram(self, sentence):
        """Maximum Likelihood Unigram Assignment"""
        result = []
        for word in sentence:
            w_lower = word.lower()
            if w_lower in self.word_tag_counts:
                best_tag = self.word_tag_counts[w_lower].most_common(1)[0][0]
            else:
                # Fallback to most frequent tag in corpus
                best_tag = "NN"
            result.append((word, best_tag))
        return result

    def get_probabilities(self, word, tag, prev_tag):
        w_lower = word.lower()
        # Emission P(word | tag)
        emission = (self.emission_counts[tag][w_lower] + 1e-4) / (self.tag_counts[tag] + 1e-4 * len(self.word_tag_counts))
        # Transition P(tag | prev_tag)
        transition = (self.transition_counts[prev_tag][tag] + 1e-4) / (self.tag_counts[prev_tag] + 1e-4 * len(self.tag_counts))
        return emission, transition

def main():
    print("=" * 75)
    print("   EXPERIMENT 08: STOCHASTIC PART-OF-SPEECH TAGGING ALGORITHM   ")
    print("=" * 75)
    
    # Annotated training corpus
    training_data = [
        [("The", "DT"), ("smart", "JJ"), ("scientist", "NN"), ("can", "MD"), ("solve", "VB"), ("the", "DT"), ("problem", "NN")],
        [("A", "DT"), ("can", "NN"), ("of", "IN"), ("soda", "NN"), ("is", "VBZ"), ("on", "IN"), ("the", "DT"), ("table", "NN")],
        [("They", "PRP"), ("can", "MD"), ("fish", "VB"), ("in", "IN"), ("the", "DT"), ("river", "NN")],
        [("The", "DT"), ("fresh", "JJ"), ("fish", "NN"), ("tastes", "VBZ"), ("good", "JJ")],
        [("Time", "NN"), ("flies", "VBZ"), ("like", "IN"), ("an", "DT"), ("arrow", "NN")],
        [("Fruit", "NN"), ("flies", "NNS"), ("like", "VB"), ("a", "DT"), ("banana", "NN")]
    ]
    
    tagger = StochasticPOSTagger()
    tagger.train(training_data)
    
    print("\n[1] DISAMBIGUATION PROBABILITY ANALYSIS FOR POLYSEMOUS WORD 'can':")
    print("-" * 75)
    for tag in ["MD", "NN"]:
        em, tr = tagger.get_probabilities("can", tag, "PRP")
        print(f"    Tag = {tag:<4} | Context: Prev Tag = PRP")
        print(f"      Emission   P('can' | {tag})   = {em:.4f}")
        print(f"      Transition P({tag} | PRP)     = {tr:.4f}")
        print(f"      Combined Joint Score        = {em * tr:.6f}")
    print("-" * 75)
    
    test_sentences = [
        ["They", "can", "solve", "the", "problem"],
        ["The", "fresh", "fish", "is", "on", "the", "table"],
        ["Fruit", "flies", "like", "a", "banana"]
    ]
    
    print("\n[2] TEST SENTENCE TAGGING RESULTS:")
    print("-" * 75)
    for sent in test_sentences:
        tagged = tagger.tag_unigram(sent)
        formatted = " ".join([f"{w}/{t}" for w, t in tagged])
        print(f"Input : {' '.join(sent)}")
        print(f"Tagged: {formatted}\n")
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
