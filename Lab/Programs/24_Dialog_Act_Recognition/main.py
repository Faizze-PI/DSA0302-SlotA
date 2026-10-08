"""
Experiment 24: Dialogue Act Recognition in Conversational Agents
Course: DSA0302 - Natural Language Processing
"""

import re
from collections import Counter

class DialogActClassifier:
    def __init__(self):
        self.wh_words = {"what", "where", "when", "why", "who", "which", "how"}
        self.yn_auxiliaries = {"is", "are", "am", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "has", "have"}

    def classify_utterance(self, utterance):
        text = utterance.strip().lower()
        first_word = text.split()[0] if text.split() else ""
        
        # 1. Greetings
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", text):
            return "GREETING", "Greeting or salutation opening"
            
        # 2. Thanking / Closings
        if re.search(r"\b(thank you|thanks|bye|goodbye|have a nice day)\b", text):
            return "THANKING / CLOSING", "Expression of gratitude or conversation closing"
            
        # 3. Agreement / Affirmation
        if re.search(r"^\b(yes|yeah|sure|correct|absolutely|agreed|sounds good|ok|okay)\b", text):
            return "AGREEMENT", "Affirmation or consensus"
            
        # 4. Disagreement / Rejection
        if re.search(r"^\b(no|nope|not really|disagree|incorrect|wrong)\b", text):
            return "DISAGREEMENT", "Rejection or negative response"
            
        # 5. Directives / Requests
        if re.search(r"\b(please|book|reserve|show me|find|send|help me|tell me|cancel)\b", text) or first_word in ["book", "show", "find", "open", "turn"]:
            return "DIRECTIVE / REQUEST", "Command, request, or action instruction"
            
        # 6. Questions
        if text.endswith("?"):
            if first_word in self.wh_words:
                return "WH_QUESTION", "Open inquiry seeking specific informational entity"
            elif first_word in self.yn_auxiliaries:
                return "YES_NO_QUESTION", "Inquiry expecting binary confirmation"
            else:
                return "QUESTION", "General inquiry"
                
        # 7. Default: Informative Statement
        return "STATEMENT / INFORM", "Declarative statement conveying factual information"

def main():
    print("=" * 75)
    print("   EXPERIMENT 24: DIALOGUE ACT RECOGNITION IN CONVERSATIONS   ")
    print("=" * 75)
    
    dialogue = [
        ("User", "Hello there, good morning!"),
        ("Agent", "Hello! How can I assist you with your flight reservations today?"),
        ("User", "What flights are available from New York to London tomorrow?"),
        ("Agent", "There are three flights available departing at 9 AM, 2 PM, and 8 PM."),
        ("User", "Can I book a seat on the 2 PM flight?"),
        ("Agent", "Yes, there are seats available in economy class."),
        ("User", "Please reserve one window seat for me."),
        ("Agent", "Your seat is reserved. Would you like to add luggage?"),
        ("User", "No, that is not needed."),
        ("Agent", "Understood. Your booking reference is FL-9821."),
        ("User", "Thank you very much for your help! Goodbye.")
    ]
    
    classifier = DialogActClassifier()
    act_counts = Counter()
    
    print("\n[1] MULTI-TURN CONVERSATION WITH DIALOGUE ACT CLASSIFICATIONS:")
    print("-" * 75)
    print(f"{'Speaker':<8} | {'Dialogue Act':<22} | {'Utterance Text'}")
    print("-" * 75)
    
    for speaker, utterance in dialogue:
        act, desc = classifier.classify_utterance(utterance)
        act_counts[act] += 1
        print(f"{speaker:<8} | {act:<22} | \"{utterance}\"")
    print("-" * 75)
    
    print("\n[2] DIALOGUE ACT FREQUENCY BREAKDOWN:")
    print("-" * 45)
    for act, count in act_counts.most_common():
        pct = (count / len(dialogue)) * 100
        print(f"    {act:<24} : {count} ({pct:.1f}%)")
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
