# Experiment 02: Finite State Automaton Recognizing Pattern Ending with 'ab'

## Aim
To implement a basic Deterministic Finite State Automaton (DFA) in Python that recognizes a specific regular language ending with the substring 'ab' over the alphabet Sigma = {a, b}.

## Software / Tools Required
Python 3.11+, Standard Library.

## Theory & Syntax
A Deterministic Finite Automaton (DFA) is formally defined as a 5-tuple M = (Q, Sigma, delta, q0, F) where:
- Q is a finite set of states: {q0, q1, q2}
- Sigma is the input alphabet: {a, b}
- delta: Q x Sigma -> Q is the state transition function
- q0 is the initial start state
- F is the set of accepting/final states: {q2}

For the language L = {w in {a, b}* | w ends with 'ab'}:
- State q0: Start state; string does not end in 'a' or 'ab'.
- State q1: The last character read was 'a'.
- State q2: The last two characters read were 'ab' (Accepting state).

Transition Table:
+-------+---+---+
| State | a | b |
+-------+---+---+
|  q0   |q1 |q0 |
|  q1   |q1 |q2 |
|  q2   |q1 |q0 |
+-------+---+---+

## Procedure
1. Define the states Q, alphabet Sigma, start state q0, and accepting states F.
2. Implement the transition function delta as a Python dictionary or method.
3. Implement a DFA simulator that processes an input string character by character.
4. Log each step: current state, input character, and next state.
5. After reading the entire input, verify whether the final state belongs to F.
6. Test the DFA against valid and invalid strings.

## Program / Source Code
```python
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
```

## Output Block
```text
(Refer to Lab/outputs/Exp02_Output.txt for execution log and Lab/outputs/Exp02_02_Finite_State_Automaton.png for terminal screenshot)
```

## Result
The Deterministic Finite Automaton (DFA) recognizing strings ending with 'ab' was successfully constructed, simulated, and verified across all test cases.
