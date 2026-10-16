import sys

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
    l_count = int(lines[idx].strip())
    idx += 1
    
    lexicon = {}
    for _ in range(l_count):
        parts = lines[idx].split()
        idx += 1
        stem, feat, form = parts[0], parts[1], parts[2]
        lexicon[(stem, feat)] = form
        
    q_count = int(lines[idx].strip())
    idx += 1
    
    for _ in range(q_count):
        parts = lines[idx].split()
        idx += 1
        stem, feat = parts[0], parts[1]
        if (stem, feat) in lexicon:
            print(lexicon[(stem, feat)])
        else:
            print(inflect_word(stem, feat))

sample_input = """3
goose N+PL geese
go V+PAST went
good ADJ+COMP better
8
cat N+PL
church N+PL
baby N+PL
stop V+PROG
make V+PROG
carry V+PAST
goose N+PL
go V+PAST"""

solve(sample_input)
