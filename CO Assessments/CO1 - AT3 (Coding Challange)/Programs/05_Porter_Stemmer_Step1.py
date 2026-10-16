import sys

def is_cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        if i == 0:
            return True
        return not is_cons(w, i - 1)
    return True

def get_measure(w):
    if not w:
        return 0
    cv = []
    for i in range(len(w)):
        cv.append('C' if is_cons(w, i) else 'V')
    collapsed = []
    for char in cv:
        if not collapsed or collapsed[-1] != char:
            collapsed.append(char)
    c_str = "".join(collapsed)
    return c_str.count("VC")

def has_vowel(w):
    for i in range(len(w)):
        if not is_cons(w, i):
            return True
    return False

def ends_double_cons(w):
    if len(w) < 2:
        return False
    return w[-1] == w[-2] and is_cons(w, len(w) - 1)

def ends_cvc(w):
    if len(w) < 3:
        return False
    c1 = is_cons(w, len(w) - 3)
    v = not is_cons(w, len(w) - 2)
    c2 = is_cons(w, len(w) - 1)
    return c1 and v and c2 and (w[-1] not in "wxy")

def step1a(w):
    if w.endswith("sses"):
        return w[:-4] + "ss"
    if w.endswith("ies"):
        return w[:-3] + "i"
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w

def step1b(w):
    if w.endswith("eed"):
        stem = w[:-3]
        if get_measure(stem) > 0:
            return stem + "ee"
        return w
    if w.endswith("ed"):
        stem = w[:-2]
        if has_vowel(stem):
            return cleanup(stem)
        return w
    if w.endswith("ing"):
        stem = w[:-3]
        if has_vowel(stem):
            return cleanup(stem)
        return w
    return w

def cleanup(stem):
    if stem.endswith("at"):
        return stem + "e"
    if stem.endswith("bl"):
        return stem + "e"
    if stem.endswith("iz"):
        return stem + "e"
    if ends_double_cons(stem) and stem[-1] not in "lsz":
        return stem[:-1]
    if get_measure(stem) == 1 and ends_cvc(stem):
        return stem + "e"
    return stem

def step1c(w):
    if w.endswith("y"):
        stem = w[:-1]
        if has_vowel(stem):
            return stem + "i"
    return w

def solve(input_text):
    lines = input_text.splitlines()
    if not lines:
        return
    q = int(lines[0].strip())
    for i in range(1, q + 1):
        w = lines[i].strip()
        w = step1a(w)
        w = step1b(w)
        w = step1c(w)
        print(w)

sample_input = """8
caresses
ponies
cats
feed
agreed
motoring
hopping
happy"""

solve(sample_input)
