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
