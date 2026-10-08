# Experiment 18: First-Order Predicate Calculus (FOPC) Parser

## Aim
To implement a parser and semantic evaluator for First-Order Predicate Calculus (FOPC) logical expressions in Python using NLTK's logical semantics toolkit.

## Software / Tools Required
Python 3.11+, NLTK Library (`nltk.sem.logic`).

## Theory & Syntax
First-Order Predicate Calculus (FOPC) is a formal language for representing the semantics and truth-conditional meanings of natural language sentences.
Components:
- Constants: Specific entities in the domain (e.g., john, mary, earth).
- Variables: Quantifiable place-holders (x, y, z).
- Predicates: Properties or relations over constants/variables (e.g., Student(x), Likes(john, mary)).
- Logical Connectives: Conjunction (&), Disjunction (|), Negation (-), Implication (->), Equivalence (<->).
- Quantifiers: Universal (all x.) and Existential (exists x.).
NLTK provides `Expression.fromstring()` to parse strings into typed abstract syntax trees (ASTs) of first-order logic formulas.

## Procedure
1. Import `Expression` and `Valuation` from `nltk.sem.logic` and `nltk.sem`.
2. Define natural language statements mapped to formal FOPC expressions.
3. Parse logical strings into typed AST expressions.
4. Display the internal tree representation and free/bound variables.
5. Construct a first-order model with a finite domain universe and predicate valuation.
6. Evaluate the truth value of the formulas against the model.

## Program / Source Code
```python
"""
Experiment 18: First-Order Predicate Calculus (FOPC) Parser
Course: DSA0302 - Natural Language Processing
"""

import nltk
from nltk.sem.logic import Expression
from nltk.sem import Valuation, Model, Assignment

def main():
    print("=" * 75)
    print("   EXPERIMENT 18: FIRST-ORDER PREDICATE CALCULUS (FOPC) PARSER   ")
    print("=" * 75)
    
    # 1. Parse natural language propositions into FOPC formulas
    formulas = [
        ("Every human is mortal", "all x. (human(x) -> mortal(x))"),
        ("Socrates is a human", "human(socrates)"),
        ("Someone is happy and smart", "exists x. (happy(x) & smart(x))"),
        ("John likes Mary", "likes(john, mary)"),
        ("No dog can fly", "all x. (dog(x) -> -flies(x))")
    ]
    
    read_expr = Expression.fromstring
    
    print("\n[1] FOPC EXPRESSION PARSING & SYNTAX ANALYSIS:")
    print("-" * 75)
    for nl, f_str in formulas:
        expr = read_expr(f_str)
        print(f"Natural Language: \"{nl}\"")
        print(f"FOPC Expression : {expr}")
        print(f"Expression Type : {type(expr).__name__}")
        print(f"Free Variables  : {expr.free()}")
        print("-" * 75)

    # 2. Model Valuation and Truth Evaluation
    print("\n[2] SEMANTIC MODEL VALUATION & TRUTH EVALUATION:")
    print("-" * 75)
    
    v = """
    socrates => s
    plato    => p
    human    => {s, p}
    mortal   => {s, p}
    dog      => {d}
    """
    val = Valuation.fromstring(v)
    dom = val.domain
    m = Model(dom, val)
    g = Assignment(dom)
    
    queries = [
        ("human(socrates)", "Is Socrates human?"),
        ("mortal(socrates)", "Is Socrates mortal?"),
        ("all x. (human(x) -> mortal(x))", "Are all humans mortal?"),
        ("human(dog)", "Is a dog human?")
    ]
    
    for q_str, question in queries:
        parsed_q = read_expr(q_str)
        result = m.evaluate(q_str, g)
        print(f"Query Formula : {parsed_q}")
        print(f"Meaning       : {question}")
        print(f"Truth Value   : {result}")
        print()
        
    print("=" * 75)
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
```

## Output Block
```text
(Refer to Lab/outputs/Exp18_Output.txt for execution log and Lab/outputs/Exp18_18_FOPC_Parser.png for terminal screenshot)
```

## Result
The First-Order Predicate Calculus (FOPC) parser was successfully implemented, demonstrating syntax tree decomposition, variable scoping, and model-theoretic truth evaluation.
