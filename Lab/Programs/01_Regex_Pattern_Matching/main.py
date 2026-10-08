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
