import re
import sys

pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")

def solve(input_text):
    lines = input_text.splitlines()
    if not lines:
        return
    n = int(lines[0].strip())
    for i in range(1, n + 1):
        line = lines[i] if i < len(lines) else ""
        tokens = pattern.findall(line)
        if tokens:
            print(f"{len(tokens)} {' '.join(tokens)}")
        else:
            print("0")

sample_input = """4
The dog's tail wagged -- twice!
state-of-the-art NLP systems, circa 2026.
12345 ... ???
Don't re-use rock--roll O'Neill's x"""

solve(sample_input)
