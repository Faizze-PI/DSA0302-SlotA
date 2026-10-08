# Experiment 12: Earley Parser for Context-Free Grammars

## Aim
To implement an Earley chart parser in Python for Context-Free Grammars using dynamic programming and state transitions.

## Software / Tools Required
Python 3.11+, Standard Library.

## Theory & Syntax
The Earley Parser (Jay Earley, 1970) is a chart-parsing algorithm that parses arbitrary context-free grammars in O(n^3) time (and O(n^2) for unambiguous grammars).
An Earley State is represented as:
  (A -> alpha . beta, [i, j])
Where:
- A -> alpha beta is a production rule.
- The dot (.) indicates current progress in parsing.
- i is the start position in the input where this rule began.
- j is the current chart index.

Three Core Operations:
1. PREDICTOR: If dot precedes a non-terminal B, add (B -> . gamma, [j, j]) for all productions B -> gamma.
2. SCANNER: If dot precedes a terminal matching token at j, add (A -> alpha a . beta, [i, j+1]) to chart j+1.
3. COMPLETER: If dot reaches the end of rule (A -> alpha ., [i, j]), advance dot for all states in chart i waiting for A.

## Procedure
1. Initialize chart state tables Chart[0] to Chart[N] for input length N.
2. Seed Chart[0] with initial start rule: (GAMMA -> . S, [0, 0]).
3. For each chart index j from 0 to N:
   - Execute Predictor, Scanner, and Completer operations until no new states are added.
4. Verify if accepting state (GAMMA -> S ., [0, N]) exists in Chart[N].
5. Display chart contents and step analysis.

## Program / Source Code
```python
"""
Experiment 12: Earley Parser for Context-Free Grammars
Course: DSA0302 - Natural Language Processing
"""

class EarleyState:
    def __init__(self, lhs, rhs, dot, start_pos):
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.dot = dot
        self.start_pos = start_pos

    def is_complete(self):
        return self.dot >= len(self.rhs)

    def next_symbol(self):
        return None if self.is_complete() else self.rhs[self.dot]

    def __eq__(self, other):
        return (self.lhs == other.lhs and self.rhs == other.rhs and 
                self.dot == other.dot and self.start_pos == other.start_pos)

    def __hash__(self):
        return hash((self.lhs, self.rhs, self.dot, self.start_pos))

    def __repr__(self):
        before = " ".join(self.rhs[:self.dot])
        after = " ".join(self.rhs[self.dot:])
        dot_str = f"{before} . {after}".strip()
        return f"[{self.lhs} -> {dot_str}, ({self.start_pos})]"

class EarleyParser:
    def __init__(self, grammar, start_symbol="S"):
        self.grammar = grammar
        self.start_symbol = start_symbol

    def parse(self, tokens):
        n = len(tokens)
        charts = [[] for _ in range(n + 1)]
        charts_set = [set() for _ in range(n + 1)]

        def add_state(chart_idx, state):
            if state not in charts_set[chart_idx]:
                charts_set[chart_idx].add(state)
                charts[chart_idx].append(state)
                return True
            return False

        # Seed chart 0
        add_state(0, EarleyState("GAMMA", [self.start_symbol], 0, 0))

        for i in range(n + 1):
            j = 0
            while j < len(charts[i]):
                state = charts[i][j]
                j += 1

                if not state.is_complete():
                    next_sym = state.next_symbol()
                    # 1. PREDICTOR (Non-terminal)
                    if next_sym in self.grammar:
                        for prod in self.grammar[next_sym]:
                            add_state(i, EarleyState(next_sym, prod, 0, i))
                    # 2. SCANNER (Terminal)
                    elif i < n and next_sym.lower() == tokens[i].lower():
                        add_state(i + 1, EarleyState(state.lhs, state.rhs, state.dot + 1, state.start_pos))
                else:
                    # 3. COMPLETER
                    for prev_state in charts[state.start_pos]:
                        if not prev_state.is_complete() and prev_state.next_symbol() == state.lhs:
                            add_state(i, EarleyState(prev_state.lhs, prev_state.rhs, prev_state.dot + 1, prev_state.start_pos))

        accept_state = EarleyState("GAMMA", [self.start_symbol], 1, 0)
        is_accepted = accept_state in charts_set[n]
        return is_accepted, charts

def main():
    print("=" * 75)
    print("   EXPERIMENT 12: EARLEY CHART PARSER FOR CONTEXT-FREE GRAMMARS   ")
    print("=" * 75)
    
    grammar = {
        "S": [["NP", "VP"]],
        "NP": [["Det", "N"], ["N"]],
        "VP": [["V", "NP"], ["V"]],
        "Det": [["the"], ["a"]],
        "N": [["astronomer"], ["telescope"], ["star"]],
        "V": [["saw"], ["observed"]]
    }
    
    parser = EarleyParser(grammar, start_symbol="S")
    sentence = "the astronomer saw the star"
    tokens = sentence.split()
    
    print(f"\nInput Sentence: '{sentence}'")
    print(f"Tokens: {tokens}\n")
    
    accepted, charts = parser.parse(tokens)
    print(f"[+] PARSE STATUS: {'ACCEPTED' if accepted else 'REJECTED'}\n")
    
    print("EARLEY CHARTS BREAKDOWN:")
    print("-" * 75)
    for i, chart in enumerate(charts):
        curr_tok = tokens[i - 1] if i > 0 else "<START>"
        print(f"=== Chart [{i}] (Processed: '{curr_tok}') ===")
        for state in chart:
            print(f"    {state}")
        print()
    print("-" * 75)
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp12_Output.txt for execution log and Lab/outputs/Exp12_12_Earley_Parser.png for terminal screenshot)
```

## Result
The Earley chart parser was successfully implemented with predictor, scanner, and completer operations, accurately parsing natural language sentences in polynomial time.
