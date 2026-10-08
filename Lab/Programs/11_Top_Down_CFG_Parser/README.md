# Experiment 11: Top-Down Parser for Context-Free Grammars

## Aim
To implement a simple Top-Down Recursive Descent parser for Context-Free Grammars (CFG) in Python with derivation trace and backtracking.

## Software / Tools Required
Python 3.11+, Standard Library.

## Theory & Syntax
A Top-Down Parser starts with the start symbol S of a Context-Free Grammar (G = (V, Sigma, R, S)) and attempts to rewrite it using production rules until the string of terminal symbols matches the input sentence.
Recursive descent parsing navigates the grammar rules recursively, expanding the leftmost non-terminal. If a chosen expansion fails to match the input tokens, the parser backtracks and explores alternative production branches.
Grammar Specification:
  S -> NP VP
  NP -> Det N | Det Adj N | Pronoun
  VP -> V NP | V PP | V
  PP -> P NP

## Procedure
1. Define the context-free grammar production rules as mappings from LHS to candidate RHS lists.
2. Implement a recursive descent parser function with token index tracking and backtracking.
3. Expand non-terminals top-down starting from root S.
4. Match terminals directly against the input token sequence.
5. Log the step-by-step expansion and backtracking decisions.
6. Test on valid and invalid sentences.

## Program / Source Code
```python
"""
Experiment 11: Top-Down Parser for Context-Free Grammars
Course: DSA0302 - Natural Language Processing
"""

class TopDownCFGParser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.trace_log = []

    def parse(self, tokens):
        self.trace_log = []
        success, end_idx, tree = self._expand("S", tokens, 0, depth=0)
        if success and end_idx == len(tokens):
            return True, tree, self.trace_log
        return False, None, self.trace_log

    def _expand(self, symbol, tokens, idx, depth):
        indent = "  " * depth
        # Base case: Terminal symbol
        if symbol not in self.grammar:
            if idx < len(tokens) and tokens[idx].lower() == symbol.lower():
                self.trace_log.append(f"{indent}Match terminal: '{tokens[idx]}'")
                return True, idx + 1, symbol
            else:
                self.trace_log.append(f"{indent}Mismatch: expected '{symbol}', found '{tokens[idx] if idx < len(tokens) else '<EOF>'}'")
                return False, idx, None

        # Non-terminal expansion: try production rules
        self.trace_log.append(f"{indent}Expand non-terminal: {symbol} at token index {idx} ('{tokens[idx] if idx < len(tokens) else '<EOF>'}')")
        for production in self.grammar[symbol]:
            curr_idx = idx
            subtrees = []
            matched = True
            self.trace_log.append(f"{indent}  Trying rule: {symbol} -> {' '.join(production)}")
            
            for sub_symbol in production:
                succ, curr_idx, sub_tree = self._expand(sub_symbol, tokens, curr_idx, depth + 2)
                if not succ:
                    matched = False
                    break
                subtrees.append(sub_tree)
                
            if matched:
                return True, curr_idx, (symbol, subtrees)
            else:
                self.trace_log.append(f"{indent}  Backtrack on rule: {symbol} -> {' '.join(production)}")

        return False, idx, None

def print_tree(tree, depth=0):
    if isinstance(tree, tuple):
        symbol, children = tree
        print("  " * depth + f"({symbol}")
        for child in children:
            print_tree(child, depth + 1)
        print("  " * depth + ")")
    else:
        print("  " * depth + f"'{tree}'")

def main():
    print("=" * 75)
    print("   EXPERIMENT 11: TOP-DOWN PARSER FOR CONTEXT-FREE GRAMMARS   ")
    print("=" * 75)
    
    # Define Context-Free Grammar
    grammar = {
        "S": [["NP", "VP"]],
        "NP": [["Det", "N"], ["Det", "Adj", "N"], ["Pronoun"]],
        "VP": [["V", "NP"], ["V"]],
        "Det": [["the"], ["a"]],
        "Adj": [["quick"], ["lazy"]],
        "N": [["dog"], ["fox"], ["scientist"]],
        "Pronoun": [["she"], ["he"]],
        "V": [["sees"], ["likes"], ["chases"]]
    }
    
    parser = TopDownCFGParser(grammar)
    
    test_sentences = [
        "the fox chases the dog",
        "she likes a lazy dog",
        "the dog chases"
    ]
    
    for sentence in test_sentences:
        tokens = sentence.split()
        print(f"\nInput Sentence: '{sentence}'")
        print("-" * 60)
        success, tree, trace = parser.parse(tokens)
        
        if success:
            print("[+] STATUS: PARSED SUCCESSFULLY")
            print("\nDerived Parse Tree:")
            print_tree(tree)
        else:
            print("[-] STATUS: PARSE FAILED")
            
        print("\nSample Derivation Trace (first 6 steps):")
        for step in trace[:6]:
            print("  ", step)
        print("-" * 60)
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp11_Output.txt for execution log and Lab/outputs/Exp11_11_Top_Down_CFG_Parser.png for terminal screenshot)
```

## Result
The top-down recursive descent parser for CFGs was successfully implemented and demonstrated parsing of valid sentences along with backtracking capabilities.
