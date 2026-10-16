import sys
from collections import defaultdict

def count_vowel_groups(s):
    vowels = set("aeiou")
    count = 0
    in_group = False
    for char in s:
        if char in vowels:
            if not in_group:
                count += 1
                in_group = True
        else:
            in_group = False
    return count

def is_consonant(c):
    return c.isalpha() and c.lower() not in "aeiou"

def is_vowel(c):
    return c.lower() in "aeiou"

def is_doublable(s):
    if len(s) < 3:
        return False
    if s[-1] in "wxy":
        return False
    if is_consonant(s[-3]) and is_vowel(s[-2]) and is_consonant(s[-1]):
        return count_vowel_groups(s) == 1
    return False

def inflect_word(stem, feature):
    if feature in ("N+PL", "V+3SG"):
        if stem.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return stem + "es"
        if len(stem) >= 2 and is_consonant(stem[-2]) and stem.endswith('o'):
            return stem + "es"
        if len(stem) >= 2 and is_consonant(stem[-2]) and stem.endswith('y'):
            return stem[:-1] + "ies"
        return stem + "s"
    elif feature == "V+PROG":
        if len(stem) >= 3 and is_consonant(stem[-2]) and stem.endswith('e'):
            return stem[:-1] + "ing"
        if is_doublable(stem):
            return stem + stem[-1] + "ing"
        return stem + "ing"
    elif feature in ("V+PAST", "ADJ+COMP", "ADJ+SUP"):
        suffix_map = {
            "V+PAST": ("d", "ed"),
            "ADJ+COMP": ("r", "er"),
            "ADJ+SUP": ("st", "est")
        }
        add_if_e, add_default = suffix_map[feature]
        if stem.endswith('e'):
            return stem + add_if_e
        if len(stem) >= 2 and is_consonant(stem[-2]) and stem.endswith('y'):
            return stem[:-1] + "i" + add_default
        if is_doublable(stem):
            return stem + stem[-1] + add_default
        return stem + add_default
    return stem

def solve(input_text):
    lines = input_text.splitlines()
    if not lines:
        return
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    s_count = int(lines[idx].strip())
    idx += 1
    
    lookup = defaultdict(set)
    for _ in range(s_count):
        parts = lines[idx].split()
        idx += 1
        stem, pos = parts[0], parts[1]
        lookup[stem].add(f"{stem}+{pos}")
        if pos == "N":
            lookup[inflect_word(stem, "N+PL")].add(f"{stem}+N+PL")
        elif pos == "V":
            lookup[inflect_word(stem, "V+3SG")].add(f"{stem}+V+3SG")
            lookup[inflect_word(stem, "V+PROG")].add(f"{stem}+V+PROG")
            lookup[inflect_word(stem, "V+PAST")].add(f"{stem}+V+PAST")
        elif pos == "ADJ":
            lookup[inflect_word(stem, "ADJ+COMP")].add(f"{stem}+ADJ+COMP")
            lookup[inflect_word(stem, "ADJ+SUP")].add(f"{stem}+ADJ+SUP")
            
    q_count = int(lines[idx].strip())
    idx += 1
    
    for _ in range(q_count):
        word = lines[idx].strip()
        idx += 1
        if word in lookup:
            sorted_ans = sorted(list(lookup[word]))
            print(", ".join(sorted_ans))
        else:
            print("UNKNOWN")

sample_input = """5
cat N
watch N
watch V
study V
big ADJ
8
cats
watches
watching
studied
bigger
cat
biggest
elephants"""

solve(sample_input)
