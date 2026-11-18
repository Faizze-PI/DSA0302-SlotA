"""Modeling, classification, and rule expansion analysis for sentence-level syntactic constructions."""

from typing import Dict, List, Tuple


CONSTRUCTION_RULES: Dict[str, List[str]] = {
    "Declarative": [
        "S -> NP VP",
        "VP -> V_trans NP | V_intr | V_ditrans NP NP | V_comp Sbar | V_prep PP"
    ],
    "Imperative": [
        "S -> VP",
        "VP -> V_trans NP | V_ditrans NP NP | V_prep PP"
    ],
    "YesNo_Question": [
        "S -> Aux NP VP",
        "S -> Copula NP AdjP",
        "S -> Copula NP PP"
    ],
    "Wh_Subject_Question": [
        "S -> Wh VP",
        "VP -> V_trans NP | V_ditrans NP NP"
    ],
    "Wh_Object_Question": [
        "S -> Wh Aux NP VP_gap",
        "VP_gap -> V_trans",
        "VP_gap -> V_ditrans NP"
    ],
    "Relative_Clause": [
        "NP -> NP RC",
        "RC -> RelPro VP",
        "RC -> RelPro S_gap"
    ],
    "Coordination": [
        "S -> S Conj S",
        "NP -> NP Conj NP",
        "VP -> VP Conj VP"
    ]
}


def classify_construction(tokens: List[str]) -> str:
    if not tokens:
        return "Unknown"
    first = tokens[0].lower()
    wh_words = {"who", "what", "when", "where", "why", "how"}
    aux_words = {"is", "are", "was", "were", "did", "does", "do", "can", "will"}
    imperative_verbs = {"submit", "send", "pay", "download", "check", "show", "book", "apply", "update"}

    if first in wh_words:
        if len(tokens) > 1 and tokens[1].lower() in aux_words:
            return "Wh_Object_Question"
        return "Wh_Subject_Question"
    elif first in aux_words:
        return "YesNo_Question"
    elif first in imperative_verbs:
        return "Imperative"
    else:
        for t in tokens:
            if t.lower() in ("and", "or", "but"):
                return "Coordinated_Clause"
            if t.lower() in ("that", "which"):
                return "Relative_Clause"
        return "Declarative"


def get_construction_growth_metrics() -> List[Tuple[str, int, str]]:
    metrics = [
        ("Base Declarative", 8, "Baseline core phrase structure ($S \\to NP \\ VP$)"),
        ("Imperative Extension", 2, "Addition of subjectless root ($S \\to VP$)"),
        ("Yes/No Inversion", 4, "Auxiliary fronting ($S \\to Aux \\ NP \\ VP$, Copular frames)"),
        ("Wh-Questions (Subj + Obj)", 6, "Wh-movement and gap-threading ($S \\to Wh \\ Aux \\ NP \\ VP_{gap}$)"),
        ("Relative Clauses", 4, "Recursive nominal embedding ($NP \\to NP \\ RC$, $RC \\to RelPro \\ VP$)"),
        ("Coordination (S, NP, VP)", 6, "Recursive conjunction rules across major syntactic categories")
    ]
    return metrics
