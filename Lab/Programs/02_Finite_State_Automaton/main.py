"""
Experiment 02: Finite State Automaton Recognizing Pattern Ending with 'ab'
Course: DSA0302 - Natural Language Processing
"""

class DFAEndingWithAB:
    def __init__(self):
        self.states = {'q0', 'q1', 'q2'}
        self.alphabet = {'a', 'b'}
        self.start_state = 'q0'
        self.accept_states = {'q2'}
        
        # Transition function delta(state, input) -> next_state
        self.transitions = {
            'q0': {'a': 'q1', 'b': 'q0'},
            'q1': {'a': 'q1', 'b': 'q2'},
            'q2': {'a': 'q1', 'b': 'q0'}
        }
        
    def process_string(self, input_str):
        current_state = self.start_state
        trace = []
        
        for idx, char in enumerate(input_str):
            if char not in self.alphabet:
                return False, f"Invalid symbol '{char}' at index {idx}. Only {self.alphabet} allowed.", trace
            
            next_state = self.transitions[current_state][char]
            trace.append((current_state, char, next_state))
            current_state = next_state
            
        is_accepted = current_state in self.accept_states
        return is_accepted, current_state, trace

def main():
    print("=" * 70)
    print("   EXPERIMENT 02: FINITE STATE AUTOMATON (DFA) FOR STRINGS ENDING IN 'ab'   ")
    print("=" * 70)
    
    dfa = DFAEndingWithAB()
    print("\n[1] DFA FORMAL SPECIFICATION:")
    print("    - States (Q): {q0, q1, q2}")
    print("    - Alphabet (Sigma): {a, b}")
    print("    - Start State (q0): q0")
    print("    - Accepting States (F): {q2}")
    print("    - Language: Strings over {a, b}* ending in 'ab'")
    
    test_cases = [
        "ab",        # Valid: exact match
        "aab",       # Valid: ends with ab
        "bab",       # Valid: ends with ab
        "abaab",     # Valid: ends with ab
        "abbab",     # Valid: ends with ab
        "bba",       # Invalid: ends with a
        "ba",        # Invalid: ends with a
        "a",         # Invalid: single char
        "b",         # Invalid: single char
        "aba",       # Invalid: ends with a
        "ababa",     # Invalid: ends with a
        ""           # Invalid: empty string
    ]
    
    print("\n[2] TEST SUITE EXECUTION:")
    print("-" * 70)
    print(f"{'Test String':<15} | {'Final State':<12} | {'Result':<10} | {'Transition Path'}")
    print("-" * 70)
    
    for test in test_cases:
        accepted, final_state, trace = dfa.process_string(test)
        path_str = " -> ".join([t[0] for t in trace] + ([final_state] if trace else [dfa.start_state]))
        status = "ACCEPTED" if accepted else "REJECTED"
        display_str = f"'{test}'" if test else "'' (empty)"
        print(f"{display_str:<15} | {str(final_state):<12} | {status:<10} | {path_str}")
        
    print("-" * 70)
    print("\n[3] DETAILED TRACE FOR STRING 'abaab':")
    _, _, detailed_trace = dfa.process_string("abaab")
    for step, (src, char, dst) in enumerate(detailed_trace, 1):
        print(f"    Step {step}: State {src} --(reads '{char}')--> State {dst}")
    print("    Final State: q2 (Belongs to F: ACCEPTED)")
    print("=" * 70)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
