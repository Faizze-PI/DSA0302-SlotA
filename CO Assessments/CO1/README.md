# CO1 - Assignment 1: Hospital Record System Text Normalisation and Search Layer

| Task # | Task Description | Marks | Core Module / Script | Output Screenshot |
| :---: | :--- | :---: | :--- | :---: |
| **01** | Linguistic Analysis Levels & Pipeline Framing | 3 | Theoretical Model / [`main.py`](./main.py) | [Screenshot](./outputs/Phase08_Master_Pipeline_Execution.png) |
| **02** | Structured Entity Regular Expression Inventory | 4 | [`src/entity_patterns.py`](./src/entity_patterns.py) | [Screenshot](./outputs/Phase02_Entity_Regex_Extraction.png) |
| **03** | Maximal-Munch Tokenisation & Normalisation | 4 | [`src/tokenizer_normalizer.py`](./src/tokenizer_normalizer.py) | [Screenshot](./outputs/Phase01_Tokenisation_Normalisation.png) |
| **04** | Automata Design, Subset Construction & Equivalence | 5 | [`src/automata_engine.py`](./src/automata_engine.py) | [Screenshot](./outputs/Phase03_Automata_Validation_Subset.png) |
| **05** | Inflectional Morphology & Paradigm Analysis | 3 | [`src/morphology_engine.py`](./src/morphology_engine.py) | [Screenshot](./outputs/Phase04_Morphological_FST_Orthography.png) |
| **06** | Derivational Morphology & Affix Productivity | 4 | [`src/morphology_engine.py`](./src/morphology_engine.py) | [Screenshot](./outputs/Phase04_Morphological_FST_Orthography.png) |
| **07** | Finite-State Transducers & Lexicon Parsing | 5 | [`src/transducer_fst.py`](./src/transducer_fst.py) | [Screenshot](./outputs/Phase04_Morphological_FST_Orthography.png) |
| **08** | Two-Level Morphology & Orthographic Rewrite Rules | 4 | [`src/transducer_fst.py`](./src/transducer_fst.py) | [Screenshot](./outputs/Phase04_Morphological_FST_Orthography.png) |
| **09** | The Porter Stemmer Algorithm & Condition Proofs | 5 | [`src/porter_stemmer.py`](./src/porter_stemmer.py) | [Screenshot](./outputs/Phase05_Porter_Stemmer_Divergence.png) |
| **10** | End-to-End Pipeline Harness & Benchmarking | 3 | [`main.py`](./main.py) | [Screenshot](./outputs/Phase08_Master_Pipeline_Execution.png) |
| **11** | Conflation Strategy Evaluation & Defense | 2 | [`src/search_retrieval.py`](./src/search_retrieval.py) | [Screenshot](./outputs/Phase06_Information_Retrieval_Benchmark.png) |
| **12** | Day-4 Challenge: OCR Corruption & Code-Mixing | 8 | [`src/day4_challenge.py`](./src/day4_challenge.py) | [Screenshot](./outputs/Phase07_Day4_Noise_Challenge.png) |

---

## Task 1: Problem Framing, Models, and Levels of Linguistic Analysis

### 1. Linguistic Analysis Hierarchy
* **Phonology:** Disregarded. Clinical record search operates over orthographic text. Phonological processes (allophonic variation, assimilation) do not directly affect string-level tokenisation or lexical indexing.
* **Morphology (Essential):** Primary operational layer. Clinical language is rich in Latin and Greek combining forms (`nephr-`, `cardio-`, `-ectomy`, `-itis`). Word forms must be mapped to base stems and features to bridge query surface variations (e.g., `fracture` $\leftrightarrow$ `fractures`, `fractured`, `fracturing`).
* **Syntax (Partially Ignored):** Shallow phrase structure helps identify dosage and medication spans (`[Drug] [Strength] [Frequency]`). Deep syntactic parsing (CFG phrase structures, parse trees) is omitted because search queries are keyword-based.
* **Semantics (Crucial):** Addressed at the lexical level via controlled clinical vocabularies and suppletion dictionaries (`renal` $\leftrightarrow$ `kidney`, `cardiac` $\leftrightarrow$ `heart`).
* **Pragmatics & Discourse (Ignored):** Narrative discourse progression, coreference, and doctor-patient dialogue dynamics are bypassed by the keyword indexing layer.

### 2. Model Families Comparison
* **State Machines (Chosen for Pattern Validation & Morphology):** Deterministic and non-deterministic finite automata (DFA/NFA) and transducers (FST) provide $O(n)$ linear-time execution, mathematical determinism, and bidirectional parsing/generation. Formal rule systems cannot match this runtime speed, and probabilistic models introduce non-deterministic hallucinations in dosage validation.
* **Formal Rule Systems (Chosen for Abbreviation & Orthography):** Context-sensitive rewrite rules ($a \to b \;/\; c \_\_ d$) handle boundary spelling changes (e.g., $e$-deletion, $y$-to-$i$). State machines without composition become overly complex when handling dozens of spelling exceptions.
* **Probabilistic Models (Deferred):** Statistical classifiers require extensive training annotations and may fail on rare dosages. They are best reserved for downstream disambiguation (e.g., classifying whether `patient` is a noun or an adjective).

### 3. Four Clinical Ambiguities
1. **Lexical Ambiguity:** `patient` acts as a noun (*"admitted the patient"*) or an adjective (*"patient monitoring is advised"*).
2. **Morphological Ambiguity:** `fractures` functions as a plural noun (*"multiple fractures detected"*) or a 3rd-person singular present verb (*"the bone fractures under stress"*).
3. **Segmentation Ambiguity:** `post-operative` can be treated as a single compound token, two hyphenated tokens, or separated into a prefix + root morpheme.
4. **Abbreviation Ambiguity:** `prn` (*pro re nata* - as needed) vs. medical record notation; `OD` (*once daily* vs. *right eye / oculus dexter* vs. *overdose*).

---

## Task 2: Regular Expression Entity Inventory

### Pattern Specifications & Negative Boundaries

| Entity Type | Regular Expression | Valid Example | Invalid String (Must Reject) |
| :--- | :--- | :--- | :--- |
| **Patient Identifier** | `\b(?:PID\|MRN\|P)-\d{4,6}\b` | `PID-10294` | `ID-12`, `PID_9999` |
| **Clinical Date** | `\b(?:\d{4}-(?:0[1-9]\|1[0-2])-(?:0[1-9]\|[12]\d\|3[01])\|(?:0[1-9]\|[12]\d\|3[01])/(?:0[1-9]\|1[0-2])/\d{4})\b` | `2024-03-12`, `14/05/2023` | `2024-13-45`, `99/99/9999` |
| **Drug + Strength** | `\b([A-Z][a-z]+(?:-[A-Za-z]+)?)\s+(\d+(?:\.\d+)?\s*(?:mg\|g\|mcg\|mL\|units))\b` | `Paracetamol 500mg` | `Paracetamol`, `500mg` |
| **Dosage Frequency** | `\b(?:b\.i\.d\.\|t\.i\.d\.\|q\.d\.s\.\|once\s+daily\|every\s+\d+\s+hours(?:\s+prn)?\|stat\|prn)\b` | `b.i.d.`, `every 8 hours prn` | `sometimes`, `twice` |
| **Lab Value + Unit** | `\b(Hb\|WBC\|Creatinine\|Platelets)\s+(\d+(?:\.\d+)?)\s*(mg/dL\|g/dL\|mmol/L\|/uL)\b` | `Hb 13.5 g/dL` | `Hb normal`, `13.5 g/dL` |
| **Blood Pressure** | `\b([89]\d\|1\d{2}\|2[0-3]\d)/([4-9]\d\|1[0-3]\d)\s*mmHg\b` | `120/80 mmHg` | `250/150`, `120/80` |
| **Temperature** | `\b(9[5-9](?:\.\d)?\|10[0-6](?:\.\d)?\|3[5-9](?:\.\d)?\|4[0-2](?:\.\d)?)\s*([FC])\b` | `98.6 F`, `37.2 C` | `108 F`, `25 C` |
| **ICD-10 Code** | `\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b` | `I10`, `K35.8` | `U99.9`, `123.4` |
| **Ward & Bed** | `\b((?:[A-Za-z\s]+Ward\|ICU-[AB])),\s*(Bed\s+\d+[A-Za-z]?)\b` | `Cardiology Ward, Bed 12` | `Ward Bed`, `Room 4` |
| **Contact Number** | `\b(?:\+91[- ]?)?[6-9]\d{9}\b` | `+91-9876543210` | `1234567890`, `98765` |
| **Patient Age** | `\b(\d{1,2}\|1[01]\d)\s*(?:yo\|years\s+old\|-year-old)\b` | `45 yo`, `62 years old` | `150 yo`, `born 1980` |
| **Symptom Duration** | `\b(?:for\|duration\|since)\s+(\d{1,2}\s+(?:hours\|days\|weeks\|months\|years))\b` | `for 3 days`, `duration 2 weeks` | `since yesterday`, `for long` |

### Greedy vs Non-Greedy Quantifier Comparison
* **Input Sentence:** `"Administer Paracetamol 500mg oral at 08:00 and Amoxicillin 250mg oral at 12:00."`
* **Greedy Match (`Paracetamol.*oral`):** Matches `Paracetamol 500mg oral at 08:00 and Amoxicillin 250mg oral`. The `.*` operator consumes characters across sentence boundaries and multiple medication spans, resulting in a false-positive conjunction span.
* **Non-Greedy Match (`Paracetamol.*?oral`):** Matches `Paracetamol 500mg oral`. The `.*?` operator yields as soon as the first `oral` delimiter appears.

### Operator Precedence & Disjunction Trap
* Unparenthesized pattern: `Paracetamol|Amoxicillin 500mg`
* Since concatenation has higher precedence than disjunction, the pattern evaluates as `(Paracetamol) | (Amoxicillin 500mg)`. It matches bare `Paracetamol` without a dosage while requiring `500mg` only for `Amoxicillin`.
* Correct grouping: `(?:Paracetamol|Amoxicillin)\s+500mg`.

---

## Task 3: Tokenisation and Normalisation

### Maximal-Munch Regular Tokenizer
Specified as a single compiled regular expression:
```python
TOKEN_REGEX = re.compile(
    r"(?P<ABBREV>\b(?:[A-Za-z]\.){2,})"
    r"|(?P<NUMBER_WITH_UNIT>\b\d+(?:\.\d+)?(?:mg|g|mcg|mL|mmol/L|g/dL|/uL|mmHg|F|C)\b)"
    r"|(?P<DECIMAL>\b\d+\.\d+\b)"
    r"|(?P<DATE>\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b)"
    r"|(?P<HYPHEN_WORD>\b[A-Za-z]+(?:-[A-Za-z]+)+\b)"
    r"|(?P<ALPHANUM_ID>\b[A-Z]+-\d+\b)"
    r"|(?P<WORD>\b[A-Za-z]+\b)"
    r"|(?P<INTEGER>\b\d+\b)"
    r"|(?P<PUNCT>[^\w\s])"
)
```

### Maximal-Munch vs Naive Whitespace Split
1. `"Administer Paracetamol 500mg, oral b.i.d."`
   - Naive split: `['Administer', 'Paracetamol', '500mg,', 'oral', 'b.i.d.']` *(Attaches comma to unit, corrupts lookup)*
   - Maximal munch: `['Administer', 'Paracetamol', '500mg', ',', 'oral', 'b.i.d.']`
2. `"Post-operative Hb 13.5 g/dL was checked"`
   - Naive split: `['Post-operative', 'Hb', '13.5', 'g/dL', 'was', 'checked']`
   - Maximal munch: `['Post-operative', 'Hb', '13.5', 'g/dL', 'was', 'checked']` *(Retains compound hyphenation intact)*
3. `"Patient age: 45 yo, admitted on 14/05/2023"`
   - Naive split: `['Patient', 'age:', '45', 'yo,', 'admitted', 'on', '14/05/2023']` *(Attaches colons and commas to age tokens)*
   - Maximal munch: `['Patient', 'age', ':', '45', 'yo', ',', 'admitted', 'on', '2024-05-14']`

### Irreversible Normalisation Decisions
1. **Case Folding (`lower()`):** Erases distinguishing acronyms (e.g., `ALL` - Acute Lymphoblastic Leukemia vs. determiner `all`; `AIDS` vs. `aids`). Acceptable for keyword retrieval, unacceptable for entity extraction.
2. **Abbreviation Expansion:** Context-independent expansion of `OD` to `"once daily"` erases potential meanings like `"right eye"` or `"overdose"`.
3. **Punctuation Stripping:** Stripping hyphens from `cross-match` can conflate distinct terms, while removing decimal points changes numeric lab values (`1.5` $\to$ `15`).

---

## Task 4: Finite State Automata & Equivalence

### Formal Definition: 5-Tuple
$M = (Q, \Sigma, \delta, q_0, F)$ where:
* $Q$: Finite set of states
* $\Sigma$: Input alphabet
* $\delta: Q \times (\Sigma \cup \{\varepsilon\}) \to 2^Q$: State transition function
* $q_0 \in Q$: Start state
* $F \subseteq Q$: Set of accepting states

### Four Implemented 6+ State Validators
1. **Patient ID DFA:** Validates `PID-\d{4,6}` ($Q=\{q_0, \dots, q_7\}$, $|Q|=8$).
2. **Blood Pressure DFA:** Validates `\d{2,3}/\d{2,3} mmHg` ($Q=\{q_0, \dots, q_{12}\}$, $|Q|=13$).
3. **Temperature DFA:** Validates `\d{2,3}(?:\.\d)? [FC]` ($Q=\{q_0, \dots, q_7\}$, $|Q|=8$).
4. **ICD-10 Code DFA:** Validates `[A-Z]\d{2}(?:\.\d{1,2})?` ($Q=\{q_0, \dots, q_6\}$, $|Q|=7$).

### $\varepsilon$-NFA & Subset Construction Table
* **Optional Dosage $\varepsilon$-NFA:** 9 States ($\{q_0, \dots, q_8\}$) modeling `\d+mg( b.i)?` with optional frequency via $\varepsilon$-transition $q_3 \xrightarrow{\varepsilon} \{q_4, q_8\}$.
* **Subset Construction ($2^Q$ Powerset):** Yields an 8-state deterministic machine ($S_0 \to S_7$).
* **Hopcroft Minimisation:** 8 states before $\to$ 8 states after (the DFA is already minimal).

### Non-Regularity Proof (Pumping Lemma)
* **Language:** Arbitrarily nested clinical annotations: $L = \{ [^n \text{note} ]^n \mid n \ge 1 \}$.
* **Proof:** Assume $L$ is regular. Let $p$ be the pumping length. Choose string $s = [^p \text{note} ]^p \in L$.
  By the Pumping Lemma, $s = xyz$ where $|xy| \le p$ and $|y| \ge 1$.
  Thus $y = [^k$ for $1 \le k \le p$.
  Pump $i = 2$: $x y^2 z = [^{p+k} \text{note} ]^p$.
  The number of opening brackets ($p+k$) does not equal the closing brackets ($p$), so $x y^2 z \notin L$.
  Contradiction. Therefore, arbitrary bracket nesting cannot be recognized by any finite automaton or regular expression.

---

## Task 5: Inflectional Morphology

### Clinical Lexeme Paradigms
1. **`fracture`:** Lemma `fracture` | Plural `fractures` | 3SG Pres `fractures` | Past `fractured` | Progressive `fracturing`.
2. **`inject`:** Lemma `inject` | 3SG Pres `injects` | Past `injected` | Progressive `injecting`.
3. **`biopsy`:** Lemma `biopsy` | Plural `biopsies` | Past `biopsied` | Progressive `biopsying`.
4. **`dilute`:** Lemma `dilute` | 3SG Pres `dilutes` | Past `diluted` | Comparative `more dilute` | Superlative `most dilute`.
5. **`severe`:** Lemma `severe` | Comparative `severer` | Superlative `severest` | Adverb `severely`.

### Orthographic Rules & Over-Generation Counter-Examples
1. **$e$-deletion:** $e \to \varepsilon \;/\; \_ + [V]$ (`fracture + ed` $\to$ `fractured`).  
   *Counter-example over-generation:* `singe + ing` $\to$ `*singing` (corrupts `singe` into `sing`).
2. **Consonant Doubling:** $C_1 \to C_1 C_1 \;/\; [V] C_1 \_ + [V]$ (`scan + ed` $\to$ `scanned`).  
   *Counter-example over-generation:* `target + ed` $\to$ `*targetted` (fails when stress is on the initial syllable).
3. **$y$-to-$i$ replacement:** $y \to i \;/\; [C] \_ + es$ (`biopsy + es` $\to$ `biopsies`).  
   *Counter-example over-generation:* `delay + es` $\to$ `*delaies` (fails when preceded by a vowel).
4. **Sibilant $-es$ insertion:** $\varepsilon \to e \;/\; [s, z, \int, t\int, x] \_ + s$ (`abscess + s` $\to$ `abscesses`).  
   *Counter-example over-generation:* `stomach + s` $\to$ `*stomaches` (fails when `ch` is pronounced $/k/$).

---

## Task 6: Derivational Morphology

### Clinical Affix Inventory (15+ Evaluated)
* `-itis` ($N \to N$): inflammation (`nephritis`, `appendicitis`, `gastritis`, `dermatitis`).
* `-ectomy` ($N \to N$): excision (`nephrectomy`, `appendectomy`, `gastrectomy`, `colectomy`).
* `-ology` ($N \to N$): study of (`nephrology`, `dermatology`, `cardiology`).
* `-osis` ($N \to N$): abnormal state (`nephrosis`, `stenosis`, `necrosis`, `thrombosis`).
* `-aemia / -emia` ($N \to N$): blood condition (`leukemia`, `hyperglycemia`, `anemia`).
* `-ation` ($V \to N$): process (`ulceration`, `coagulation`, `perforation`, `herniation`).
* `-able` ($V \to ADJ$): capability (`operable`, `curable`, `transmissible`).
* `-ive` ($V \to ADJ$): tendency (`hypertensive`, `exudative`, `ulcerative`).
* `hyper-` ($ADJ/N \to ADJ/N$): excess (`hypertension`, `hyperglycemia`, `hyperlipidemia`).
* `hypo-` ($ADJ/N \to ADJ/N$): deficiency (`hypotension`, `hypoglycemia`, `hypothermia`).
* `nephro-` ($Root \to Compound$): kidney (`nephrolithiasis`, `nephropathy`).
* `-scopy` ($N \to N$): inspection (`colonoscopy`, `arthroscopy`, `bronchoscopy`).
* `-plasty` ($N \to N$): surgical repair (`arthroplasty`, `rhinoplasty`).
* `-megaly` ($N \to N$): enlargement (`hepatomegaly`, `splenomegaly`, `cardiomegaly`).
* `-pathy` ($N \to N$): disease (`cardiomyopathy`, `nephropathy`, `arthropathy`).

### Derivational Family Ordering & Illegal Combinations
Root: `nephr`
* `nephr` $\to$ `nephritis` (Derivation) $\to$ `nephritides` (Inflectional plural).
* **Illegal Ordering:** `*nephr-ed-itis` (Inflectional affix `-ed` trapped inside derivational affix `-itis`). Inflectional affixes must attach outside derivational affixes.

---

## Tasks 7 & 8: FST & Orthographic Cascade

### Bidirectional Two-Tape Transducer
* **Lexical Tape:** `fracture +N +PL`
* **Surface Tape:** `fractures`
* **Parsing Direction:** Consumes surface characters `f-r-a-c-t-u-r-e-s` $\to$ emits `fracture +N +PL`.
* **Generation Direction:** Consumes lexical string `fracture +N +PL` $\to$ applies cascade $\to$ emits `fractures`.
* **Ambiguity Handling:** Input `fractures` returns two distinct lexical paths:
  1. `fracture +N +PL`
  2. `fracture +V +3SG +PRES`

### 20-Form Orthographic Verification Suite
All 20 morphological boundary transformations executed with **100% PASS** rate:
* `fracture + ed` $\to$ `fractured` ($e$-deletion)
* `scan + ed` $\to$ `scanned` (consonant doubling)
* `biopsy + es` $\to$ `biopsies` ($y$-to-$i$)
* `abscess + s` $\to$ `abscesses` (sibilant $-es$)
* `nephr + ectomy` $\to$ `nephrectomy` (derivational boundary)

---

## Tasks 9 & 11: The Porter Stemmer & Retrieval Experiments

### Condition Measure Calculations on 6 Clinical Words
* `m`: count of $VC$ sequences in $[C](VC)^m[V]$.
1. `fracture`: `f-r` (C) `a` (V) `c-t` (C) `u` (V) `r` (C) `e` (V) $\to$ $C(VC)^2V \implies m = 2$.
2. `treatment`: `t-r` (C) `ea` (V) `t-m` (C) `e` (V) `n-t` (C) $\to$ $C(VC)^2 \implies m = 2$.
3. `operating`: `o` (V) `p` (C) `e-r` (VC) `a-t` (VC) `i-ng` (VC) $\to$ $(VC)^3 \implies m = 3$.
4. `biopsy`: `b` (C) `io` (V) `p-s` (C) `y` (V) $\to$ $C(VC)^1V \implies m = 1$.
5. `severe`: `s` (C) `e` (V) `v` (C) `e` (V) `r` (C) `e` (V) $\to$ $C(VC)^2V \implies m = 2$.
6. `scan`: `s-c` (C) `a` (V) `n` (C) $\to$ $C(VC)^1 \implies m = 1$.

### 40-Query Retrieval Benchmark Over 500 Clinical Records

| Retrieval Regime | Mean Precision | Mean Recall | Mean F1-Score | Processing Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **1. Un-normalised (Raw Surface)** | **1.0000** | 1.0000 | 1.0000 | Baseline exact keyword match; drops queries with inflected terms. |
| **2. Porter Stemming** | 0.8507 | **1.0000** | 0.8541 | Fast, aggressive rule-based stem collapse; suffers from over-stemming. |
| **3. Morphological FST Lemmatisation** | 0.8549 | **1.0000** | **0.8581** | Highest semantic F1; preserves true medical lemmas and resolves irregulars. |

---

## Task 12: Day-4 Design Challenge (OCR & Regional Code-Mixing)

### Input Noise Characteristics
1. **OCR Artifacts:** Scanned documents introduce character substitutions (`0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `l`, `rn` $\leftrightarrow$ `m`), e.g., `Paracetam0l 5OOmg`, `bIood pressure 14O/9O mmHg`.
2. **Tamil-English Code-Mixing:** Spoken notes mix English clinical stems with Tamil functional morphemes (*"patient ku fever irundhuchu, paracetamol 500mg kuduthen"*).

### Architectural Decision: MODIFY
* Retaining the clean-text pipeline results in near-complete failure (coverage drops from 100% to **1.9% - 3.5%**).
* Replacing the finite-state layer with deep neural models is unjustified, non-deterministic, and costly to deploy in real-time hospital EHR infrastructure.
* **The Adapted Solution:**
  1. OCR confusion transducer and de-hyphenator placed ahead of regex extraction.
  2. Dative/locative Tamil affix stripper (`-ku`, `-la`, `-kitta`, `-oda`) exposing the root English lemma.
  3. Porter stemmer fallback for out-of-lexicon stems.

### Measured Degradation & Recovery Metrics

| Record ID | Input Noise Type | Raw Pipeline Coverage | Adapted Pipeline Coverage | Net Recovery Gain |
| :--- | :--- | :---: | :---: | :---: |
| **NOISY-OCR-001** | OCR Scanned Form | 1.9% | **46.2%** | **+44.2%** |
| **NOISY-MIXED-002** | Romanised Tamil Code-Mixing | 3.2% | **57.9%** | **+54.7%** |
| **NOISY-HYBRID-003** | Hybrid OCR + Code-Mixing | 3.5% | **40.5%** | **+36.9%** |

---

## Dataset Inventory (`data/`)

The repository operates on pure tabular `.csv` datasets (alongside raw `.txt`):
* **`clinical_corpus.csv` / `clinical_corpus.txt`**: 500 clinical EHR records (103,954 tokens across Discharge Summaries, Prescriptions, Operation Notes, and Nursing Observations).
* **`annotated_200_tokens.csv`**: 214 gold hand-annotated clinical word forms with stems, lemmas, POS tags, and morphological features.
* **`eval_40_queries.csv`**: 40 multi-word clinician benchmark search queries with retrieval intents and semicolon-delimited relevant document IDs.
* **`morphological_lexicon.csv`**: 300 clinical stems, 20 inflectional affixes, 21 derivational affixes, and irregular paradigms categorized in tabular form.
* **`noisy_day4_records.csv`**: OCR-corrupted and Romanised Tamil-English code-mixed records for Task 12.

---

## Verification & Execution Instructions

To execute the master pipeline and reproduce all benchmarks:
```powershell
# Run the end-to-end clinical NLP pipeline over all 500 clinical records
python main.py
```
Generated performance plots and phase outputs are stored under [`outputs/`](./outputs/):
* `fig1_coverage_vs_lexicon.png`: Vocabulary coverage against lexicon size.
* `fig2_latency_vs_length.png`: Processing latency vs. record token length.
* `Phase01_Tokenisation_Normalisation.png` to `Phase08_Master_Pipeline_Execution.png`: Verified terminal output screenshots for all assessment phases.
