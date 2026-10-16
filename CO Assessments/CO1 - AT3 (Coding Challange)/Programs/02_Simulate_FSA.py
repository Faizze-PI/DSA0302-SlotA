import sys
from collections import defaultdict

def solve(input_text):
    lines = input_text.splitlines()
    if not lines:
        return
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return
    n, k = map(int, lines[idx].split())
    idx += 1
    alphabet = lines[idx].split()
    idx += 1
    start_state = int(lines[idx].strip())
    idx += 1
    accept_line = list(map(int, lines[idx].split()))
    m = accept_line[0]
    accept_states = set(accept_line[1:])
    idx += 1
    t = int(lines[idx].strip())
    idx += 1
    
    transitions = defaultdict(lambda: defaultdict(set))
    for _ in range(t):
        parts = lines[idx].split()
        idx += 1
        p, c, r = int(parts[0]), parts[1], int(parts[2])
        transitions[p][c].add(r)
        
    q = int(lines[idx].strip())
    idx += 1
    
    for _ in range(q):
        if idx >= len(lines):
            break
        query = lines[idx].strip()
        idx += 1
        
        current_states = {start_state}
        if query == "#":
            print("ACCEPT" if (current_states & accept_states) else "REJECT")
            continue
            
        for char in query:
            next_states = set()
            for st in current_states:
                if char in transitions[st]:
                    next_states.update(transitions[st][char])
            current_states = next_states
            if not current_states:
                break
        print("ACCEPT" if (current_states & accept_states) else "REJECT")

sample_input = """5 3
b a !
0
1 4
5
0 b 1
1 a 2
2 a 3
3 a 3
3 ! 4
5
baa!
baaa!
ba!
b
baa"""

solve(sample_input)
