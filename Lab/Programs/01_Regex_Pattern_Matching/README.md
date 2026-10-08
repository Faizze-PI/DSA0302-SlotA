# Experiment 01: Pattern Matching and Text Searching using Regular Expressions

## Aim
To write a Python program that demonstrates how to use regular expressions to match, search, and extract specific patterns (emails, phone numbers, dates, URLs, and hashtags) from text.

## Software / Tools Required
Python 3.11+, Standard Library `re` module.

## Theory & Syntax
Regular expressions (regex) provide a concise and flexible notation for matching strings of text, such as particular characters, words, or patterns of characters. In NLP, regex is foundational for tokenization, text cleaning, entity extraction, and syntactic normalization.
Key functions in Python's `re` module:
- `re.search(pattern, string)`: Scans through a string, looking for any location where the pattern matches.
- `re.match(pattern, string)`: Determines if the regex matches at the beginning of the string.
- `re.findall(pattern, string)`: Returns all non-overlapping matches of pattern in string as a list of strings.
- `re.finditer(pattern, string)`: Returns an iterator yielding Match objects over all non-overlapping matches.
- `re.sub(pattern, repl, string)`: Replaces occurrences of the pattern that match with the replacement string.

## Procedure
1. Import the built-in `re` module.
2. Load or define raw sample text containing diverse structured patterns (emails, URLs, phone numbers, dates, and identifiers).
3. Compile regular expression patterns using standard meta-characters, quantifiers, and character classes.
4. Execute extraction routines using `re.findall()` and `re.finditer()` to obtain spans and values.
5. Implement text cleaning and anonymization routines using `re.sub()`.
6. Display the extracted entities and cleaned output in structured tabular form.

## Program / Source Code
```python
"""
Experiment 01: Pattern Matching and Text Searching using Regular Expressions
Course: DSA0302 - Natural Language Processing
"""

import re
import os

def load_sample_text():
    filepath = os.path.join(os.path.dirname(__file__), "sample_text.txt")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return """
    Contact Dr. Alan Turing at alan.turing@bletchley.org or call +1 (555) 234-5678.
    Alternate contact: support-team@nlp-labs.edu or phone: 987-654-3210.
    Conference dates: 2026-10-15 and 05/12/2026.
    Official portal: https://nlp.stanford.edu/research/ and http://huggingface.co/models.
    Follow updates on Twitter: #NLProc @OpenAI @DeepMind.
    User IDs: UID_8921, UID_4432, Order #90812.
    """

def extract_patterns(text):
    patterns = {
        "Email Addresses": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "Phone Numbers": r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ISO Dates (YYYY-MM-DD)": r"\b\d{4}-\d{2}-\d{2}\b",
        "Standard Dates (MM/DD/YYYY)": r"\b\d{2}/\d{2}/\d{4}\b",
        "Web URLs": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[\w/.-]*",
        "Hashtags": r"#\w+",
        "User Mentions": r"@\w+",
        "Custom IDs": r"\bUID_\d+\b"
    }

    results = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        results[name] = matches
    return results

def mask_sensitive_data(text):
    # Mask email usernames and phone numbers for privacy
    masked_email = re.sub(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"***@\2", text)
    masked_all = re.sub(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[REDACTED_PHONE]", masked_email)
    return masked_all

def main():
    print("=" * 70)
    print("   EXPERIMENT 01: REGULAR EXPRESSIONS FOR PATTERN MATCHING IN NLP   ")
    print("=" * 70)

    text = load_sample_text()
    print("\n[1] INPUT TEXT CORPUS:")
    print("-" * 50)
    print(text.strip())
    print("-" * 50)

    print("\n[2] PATTERN EXTRACTION RESULTS:")
    print("=" * 70)
    results = extract_patterns(text)
    for category, matches in results.items():
        print(f"\n[*] {category} (Total: {len(matches)}):")
        for idx, match in enumerate(matches, 1):
            print(f"    {idx}. {match}")

    print("\n" + "=" * 70)
    print("[3] DATA SANITIZATION / REGEX REPLACEMENT:")
    print("-" * 50)
    masked_text = mask_sensitive_data(text)
    print(masked_text.strip())
    print("=" * 70)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp01_Output.txt for execution log and Lab/outputs/Exp01_01_Regex_Pattern_Matching.png for terminal screenshot)
```

## Result
The regular expression pattern matching and text searching program was successfully implemented and tested. Various entities such as emails, phone numbers, URLs, dates, and hashtags were accurately extracted and sanitized.
