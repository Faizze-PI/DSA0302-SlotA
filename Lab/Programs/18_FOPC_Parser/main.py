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
    rover    => r
    human    => {s, p}
    mortal   => {s, p}
    dog      => {r}
    """
    val = Valuation.fromstring(v)
    dom = val.domain
    m = Model(dom, val)
    g = Assignment(dom)
    
    queries = [
        ("human(socrates)", "Is Socrates human?"),
        ("mortal(socrates)", "Is Socrates mortal?"),
        ("all x. (human(x) -> mortal(x))", "Are all humans mortal?"),
        ("dog(rover)", "Is Rover a dog?"),
        ("human(rover)", "Is Rover human?")
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
