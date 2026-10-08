# DSA0302: Natural Language Processing — Laboratory Coursework Manual

**Institution:** SIMATS Engineering  
**Department:** Computer Science and Engineering / Artificial Intelligence & Data Science  
**Course Code:** `DSA0302`  
**Course Title:** Natural Language Processing (NLP)  
**Target Environment:** Python 3.11+ (Windows / Linux / macOS)  
**Repository:** [Faizze-PI/DSA0302-SlotA](https://github.com/Faizze-PI/DSA0302-SlotA)

---

## 1. Laboratory Coursework Overview

This repository contains the complete, verified laboratory coursework implementation for **DSA0302: Natural Language Processing**. The curriculum covers the computational foundations of language processing from deterministic finite-state automata and morphological analyzers to context-free parsing, vector space information retrieval, discourse pragmatics, and state-of-the-art neural transformer models.

### Directory Structure
```text
Lab/
├── Programs/                          # Individual self-contained experiment folders
│   ├── 01_Regex_Pattern_Matching/
│   ├── 02_Finite_State_Automaton/
│   ├── 03_Morphological_Analysis/
│   ├── ...
│   └── 26_HuggingFace_Translation/
├── outputs/                           # Raw execution logs and terminal screenshots
│   ├── Exp01_Output.txt ... Exp26_Output.txt
│   ├── Exp01_01_Regex_Pattern_Matching.png ... Exp26_26_HuggingFace_Translation.png
│   └── README.md                      # Visual gallery of all terminal screenshots
├── test_all_experiments.py            # Master automated verification test harness
├── generate_screenshots.py            # High-resolution screenshot & gallery generator
└── README.md                          # Master documentation (this file)
```

---

## 2. Complete Experiment Index

| Exp # | Experiment Title | Module | Source Code | Output Screenshot | Raw Log |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **01** | Pattern Matching using Regular Expressions | M1: Automata & Morphology | [Code](./Programs/01_Regex_Pattern_Matching/) | [Screenshot](./outputs/Exp01_01_Regex_Pattern_Matching.png) | [Log](./outputs/Exp01_Output.txt) |
| **02** | Finite State Automaton Ending in 'ab' | M1: Automata & Morphology | [Code](./Programs/02_Finite_State_Automaton/) | [Screenshot](./outputs/Exp02_02_Finite_State_Automaton.png) | [Log](./outputs/Exp02_Output.txt) |
| **03** | Morphological Analysis using NLTK | M1: Automata & Morphology | [Code](./Programs/03_Morphological_Analysis/) | [Screenshot](./outputs/Exp03_03_Morphological_Analysis.png) | [Log](./outputs/Exp03_Output.txt) |
| **04** | FSM for Morphological Parsing (Plurals) | M1: Automata & Morphology | [Code](./Programs/04_FSM_Morphological_Parsing/) | [Screenshot](./outputs/Exp04_04_FSM_Morphological_Parsing.png) | [Log](./outputs/Exp04_Output.txt) |
| **05** | Word Stemming with Porter Stemmer | M1: Automata & Morphology | [Code](./Programs/05_Porter_Stemmer/) | [Screenshot](./outputs/Exp05_05_Porter_Stemmer.png) | [Log](./outputs/Exp05_Output.txt) |
| **06** | N-gram Model for Text Generation | M2: Language Modeling | [Code](./Programs/06_Ngram_Text_Generation/) | [Screenshot](./outputs/Exp06_06_Ngram_Text_Generation.png) | [Log](./outputs/Exp06_Output.txt) |
| **07** | POS Tagging using NLTK Library | M3: Part-of-Speech Tagging | [Code](./Programs/07_POS_Tagging_NLTK/) | [Screenshot](./outputs/Exp07_07_POS_Tagging_NLTK.png) | [Log](./outputs/Exp07_Output.txt) |
| **08** | Stochastic (Probabilistic) POS Tagging | M3: Part-of-Speech Tagging | [Code](./Programs/08_Stochastic_POS_Tagger/) | [Screenshot](./outputs/Exp08_08_Stochastic_POS_Tagger.png) | [Log](./outputs/Exp08_Output.txt) |
| **09** | Rule-Based POS Tagging using Regex | M3: Part-of-Speech Tagging | [Code](./Programs/09_Rule_Based_POS_Tagger/) | [Screenshot](./outputs/Exp09_09_Rule_Based_POS_Tagger.png) | [Log](./outputs/Exp09_Output.txt) |
| **10** | Transformation-Based Tagging (Brill) | M3: Part-of-Speech Tagging | [Code](./Programs/10_Transformation_Based_Tagger/) | [Screenshot](./outputs/Exp10_10_Transformation_Based_Tagger.png) | [Log](./outputs/Exp10_Output.txt) |
| **11** | Top-Down Parser for CFGs | M4: Syntactic Parsing | [Code](./Programs/11_Top_Down_CFG_Parser/) | [Screenshot](./outputs/Exp11_11_Top_Down_CFG_Parser.png) | [Log](./outputs/Exp11_Output.txt) |
| **12** | Earley Chart Parser for CFGs | M4: Syntactic Parsing | [Code](./Programs/12_Earley_Parser/) | [Screenshot](./outputs/Exp12_12_Earley_Parser.png) | [Log](./outputs/Exp12_Output.txt) |
| **13** | Parse Tree Generation for CFGs | M4: Syntactic Parsing | [Code](./Programs/13_Parse_Tree_Generation/) | [Screenshot](./outputs/Exp13_13_Parse_Tree_Generation.png) | [Log](./outputs/Exp13_Output.txt) |
| **14** | Subject-Verb & Number Agreement Checking | M4: Syntactic Parsing | [Code](./Programs/14_Agreement_Checking/) | [Screenshot](./outputs/Exp14_14_Agreement_Checking.png) | [Log](./outputs/Exp14_Output.txt) |
| **15** | Probabilistic CFG (PCFG) Parsing | M4: Syntactic Parsing | [Code](./Programs/15_PCFG_Parsing/) | [Screenshot](./outputs/Exp15_15_PCFG_Parsing.png) | [Log](./outputs/Exp15_Output.txt) |
| **16** | Named Entity Recognition (NER) with SpaCy | M5: Lexical & Formal Semantics | [Code](./Programs/16_SpaCy_NER/) | [Screenshot](./outputs/Exp16_16_SpaCy_NER.png) | [Log](./outputs/Exp16_Output.txt) |
| **17** | Lexical Database Exploration (WordNet) | M5: Lexical & Formal Semantics | [Code](./Programs/17_WordNet_Exploration/) | [Screenshot](./outputs/Exp17_17_WordNet_Exploration.png) | [Log](./outputs/Exp17_Output.txt) |
| **18** | First-Order Predicate Calculus (FOPC) Parser | M5: Lexical & Formal Semantics | [Code](./Programs/18_FOPC_Parser/) | [Screenshot](./outputs/Exp18_18_FOPC_Parser.png) | [Log](./outputs/Exp18_Output.txt) |
| **19** | Word Sense Disambiguation (Lesk Algorithm) | M5: Lexical & Formal Semantics | [Code](./Programs/19_Lesk_WSD/) | [Screenshot](./outputs/Exp19_19_Lesk_WSD.png) | [Log](./outputs/Exp19_Output.txt) |
| **20** | Information Retrieval using TF-IDF | M6: Information Retrieval | [Code](./Programs/20_TFIDF_IR_System/) | [Screenshot](./outputs/Exp20_20_TFIDF_IR_System.png) | [Log](./outputs/Exp20_Output.txt) |
| **21** | Syntax-Driven Semantic Analysis | M5: Lexical & Formal Semantics | [Code](./Programs/21_Syntax_Driven_Semantics/) | [Screenshot](./outputs/Exp21_21_Syntax_Driven_Semantics.png) | [Log](./outputs/Exp21_Output.txt) |
| **22** | Reference & Anaphora Resolution | M7: Discourse & Pragmatics | [Code](./Programs/22_Reference_Resolution/) | [Screenshot](./outputs/Exp22_22_Reference_Resolution.png) | [Log](./outputs/Exp22_Output.txt) |
| **23** | Text Coherence & Cohesion Evaluation | M7: Discourse & Pragmatics | [Code](./Programs/23_Text_Coherence_Evaluation/) | [Screenshot](./outputs/Exp23_23_Text_Coherence_Evaluation.png) | [Log](./outputs/Exp23_Output.txt) |
| **24** | Dialogue Act Recognition in Conversations | M7: Discourse & Pragmatics | [Code](./Programs/24_Dialog_Act_Recognition/) | [Screenshot](./outputs/Exp24_24_Dialog_Act_Recognition.png) | [Log](./outputs/Exp24_Output.txt) |
| **25** | Text Generation using OpenAI GPT-3 | M8: Deep Learning & Generative AI | [Code](./Programs/25_GPT3_Text_Generation/) | [Screenshot](./outputs/Exp25_25_GPT3_Text_Generation.png) | [Log](./outputs/Exp25_Output.txt) |
| **26** | Machine Translation using Transformers | M8: Deep Learning & Generative AI | [Code](./Programs/26_HuggingFace_Translation/) | [Screenshot](./outputs/Exp26_26_HuggingFace_Translation.png) | [Log](./outputs/Exp26_Output.txt) |

---

## 3. Automated Verification & Testing

Every experiment has been verified using the automated test suite [`Lab/test_all_experiments.py`](./test_all_experiments.py).

### How to Run the Test Suite
```powershell
# Navigate to the repository root
cd "Lab"

# Run all 26 experiments and generate logs
python test_all_experiments.py
```

### Verification Result
- **Total Experiments:** 26
- **Passed:** 26
- **Failed:** 0
- **Success Rate:** **100.0%**

---

## 4. Comprehensive Viva Voce Questions & Answers

### Q1: What is the difference between Stemming and Lemmatization?
**Answer:**  
- **Stemming** is a crude heuristic process that chops off the ends of words in the hope of achieving the goal most of the time (e.g., Porter Stemmer converting *running* $\to$ *run*, but *crying* $\to$ *cri*). It does not take part-of-speech or context into account and may generate non-dictionary words.
- **Lemmatization** uses vocabulary and morphological analysis with a lexical knowledge base (such as WordNet) to return the base dictionary form, called the *lemma* (e.g., *better* with adjective POS $\to$ *good*, *wolves* with noun POS $\to$ *wolf*).

### Q2: What is the formal definition of a Deterministic Finite Automaton (DFA)?
**Answer:**  
A DFA is a 5-tuple $M = (Q, \Sigma, \delta, q_0, F)$ where:
1. $Q$ is a finite set of states.
2. $\Sigma$ is a finite set of input symbols (alphabet).
3. $\delta: Q \times \Sigma \to Q$ is the state transition function.
4. $q_0 \in Q$ is the initial start state.
5. $F \subseteq Q$ is the set of final/accepting states.

### Q3: Explain the Markov Assumption in N-gram Language Modeling.
**Answer:**  
The Markov assumption states that the probability of the current word depends only on the previous $N-1$ words rather than the entire conversational history:
$$P(w_n \mid w_1, w_2, \dots, w_{n-1}) \approx P(w_n \mid w_{n-1}, \dots, w_{n-N+1})$$
For a Bigram model ($N=2$), $P(w_i \mid w_{1 \dots i-1}) \approx P(w_i \mid w_{i-1})$.

### Q4: How does a Hidden Markov Model (HMM) perform Part-of-Speech tagging?
**Answer:**  
An HMM treats the words as observed sequences and the POS tags as hidden states. The optimal tag sequence $\hat{T}$ is determined via the fundamental equation:
$$\hat{T} = \arg\max_T \prod_{i=1}^n P(w_i \mid t_i) P(t_i \mid t_{i-1})$$
Where $P(w_i \mid t_i)$ is the **Emission Probability** and $P(t_i \mid t_{i-1})$ is the **Transition Probability**, calculated using Maximum Likelihood Estimation and decoded efficiently using the Viterbi algorithm.

### Q5: How does Brill's Transformation-Based Tagging algorithm work?
**Answer:**  
Brill's tagger operates in two phases:
1. **Initial Tagging:** Annotate each word with its most likely tag in isolation (unigram frequency).
2. **Error-Driven Transformation:** Iteratively apply deterministic rewrite rules (e.g., *Change tag from NN to VB if preceded by TO*) to reduce tagging errors against a reference gold standard.

### Q6: What are the three fundamental operations of the Earley Parser?
**Answer:**  
1. **Predictor:** When the dot is in front of a non-terminal $B$, introduce new states for all production rules expanding $B$.
2. **Scanner:** When the dot is in front of a terminal matching the current input token, advance the dot past that terminal into the next chart.
3. **Completer:** When the dot reaches the end of a production rule, look back at the chart where this rule began and advance the dot for all waiting rules.

### Q7: What causes Prepositional Phrase (PP) attachment ambiguity in CFG parsing?
**Answer:**  
In sentences like *"I saw the man with a telescope"*, the prepositional phrase *with a telescope* can syntactically attach either to the Noun Phrase ($NP \to NP\ PP$: the man possessed the telescope) or to the Verb Phrase ($VP \to VP\ PP$: the instrument of seeing was the telescope). Without semantic or probabilistic grounding, a CFG generates multiple valid parse trees.

### Q8: What is a Probabilistic Context-Free Grammar (PCFG)?
**Answer:**  
A PCFG extends CFGs by assigning probabilities to production rules such that for each non-terminal $A$, $\sum_\beta P(A \to \beta) = 1.0$. The probability of a parse tree $T$ is the product of all production probabilities in its derivation. The Viterbi algorithm finds the single derivation with maximal joint probability.

### Q9: What is WordNet and what are Synsets?
**Answer:**  
WordNet is a large lexical database for English. Its core unit is the **Synset** (synonym set), which represents a unique cognitive concept grouped with member words sharing that concept, accompanied by a gloss (definition) and usage examples. Synsets are linked by semantic relations such as hypernyms (is-a), hyponyms, meronyms (part-of), and antonyms.

### Q10: How does the Simplified Lesk Algorithm resolve word sense ambiguity?
**Answer:**  
The Lesk algorithm relies on the **overlap hypothesis**: the intended sense of a polysemous word will share vocabulary terms between its dictionary gloss/examples and the words in the surrounding sentence context. The sense with the highest intersection word count is selected.

### Q11: How is TF-IDF calculated in Information Retrieval?
**Answer:**  
$$\text{TF}(t, d) = \frac{\text{Count}(t, d)}{\text{Total terms in } d}, \quad \text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
High TF-IDF terms are frequent in a specific document but rare across the entire collection, making them highly discriminative keywords.

### Q12: Explain the difference between Anaphora and Cataphora in reference resolution.
**Answer:**  
- **Anaphora:** The pronoun refers back to an antecedent mentioned earlier in the text (e.g., *"Alan Turing was a mathematician. **He** designed the machine"*).
- **Cataphora:** The pronoun refers forward to an entity introduced later (e.g., *"Before **he** arrived, Alan called the office"*).

### Q13: What is the difference between Discourse Cohesion and Coherence?
**Answer:**  
- **Cohesion** refers to surface linguistic links connecting sentences together (e.g., transition words like *however*, *therefore*, pronoun references).
- **Coherence** refers to the underlying conceptual and logical structure of the text that makes it meaningful and sensible as a whole.

### Q14: What is a Dialogue Act (Speech Act)?
**Answer:**  
A Dialogue Act represents the pragmatic communicative action performed by a speaker in making an utterance (e.g., *Statement*, *Question*, *Directive/Request*, *Greeting*, *Agreement*). It models the speaker's communicative goal beyond the literal words.

### Q15: How does the MarianMT Transformer architecture perform English-to-French translation?
**Answer:**  
MarianMT is a sequence-to-sequence (Seq2Seq) model with an **Encoder** and a **Decoder**:
1. The **Encoder** consumes English source tokens and creates contextual vector representations via bidirectional self-attention.
2. The **Decoder** auto-regressively predicts French tokens using masked self-attention over previously generated target tokens and cross-attention over the encoder's representations, optimized using Beam Search.

---

## 5. Course Record Verification Sign-off

- **Completed by:** Student / AI Pair Programming Engineer
- **Course Code:** `DSA0302`
- **Subject:** Natural Language Processing
- **Institution:** SIMATS Engineering
- **Status:** All 26 experiments fully coded, verified, and documented with 100% test pass rate.
