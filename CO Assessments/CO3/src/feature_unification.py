"""Attribute-Value Matrices (AVMs), feature path resolution, subsumption, and unification engine."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union


FeatureValue = Union[str, int, 'FeatureStructure', None]


class FeatureStructure:
    def __init__(self, features: Optional[Dict[str, Any]] = None, variable_tag: Optional[str] = None):
        self.features: Dict[str, Any] = {}
        self.variable_tag = variable_tag
        if features:
            for k, v in features.items():
                if isinstance(v, dict):
                    self.features[k] = FeatureStructure(v)
                else:
                    self.features[k] = v

    def get_path(self, path: List[str]) -> Any:
        curr = self
        for attr in path:
            if not isinstance(curr, FeatureStructure) or attr not in curr.features:
                return None
            curr = curr.features[attr]
        return curr

    def set_path(self, path: List[str], val: Any):
        curr = self
        for attr in path[:-1]:
            if attr not in curr.features or not isinstance(curr.features[attr], FeatureStructure):
                curr.features[attr] = FeatureStructure()
            curr = curr.features[attr]
        curr.features[path[-1]] = val

    def format_avm(self, indent: int = 0) -> str:
        pad = "  " * indent
        tag_str = f"[{self.variable_tag}] " if self.variable_tag else ""
        lines = [f"{pad}{tag_str}["]
        for k, v in sorted(self.features.items()):
            if isinstance(v, FeatureStructure):
                sub_str = v.format_avm(indent + 1)
                lines.append(f"{pad}  {k}: \n{sub_str}")
            else:
                lines.append(f"{pad}  {k}: {v}")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {v}" for k, v in self.features.items())
        tag = f"({self.variable_tag})" if self.variable_tag else ""
        return f"{tag}[{inner}]"


def subsumes(fs1: FeatureStructure, fs2: FeatureStructure) -> bool:
    for k, v1 in fs1.features.items():
        if k not in fs2.features:
            return False
        v2 = fs2.features[k]
        if isinstance(v1, FeatureStructure):
            if not isinstance(v2, FeatureStructure):
                return False
            if not subsumes(v1, v2):
                return False
        else:
            if v1 != "any" and v1 != v2:
                return False
    return True


def unify(fs1: Optional[FeatureStructure], fs2: Optional[FeatureStructure]) -> Tuple[Optional[FeatureStructure], Optional[str]]:
    if fs1 is None or fs2 is None:
        return None, "Null feature structure passed"

    result = copy.deepcopy(fs1)
    for k, v2 in fs2.features.items():
        if k not in result.features:
            result.features[k] = copy.deepcopy(v2)
        else:
            v1 = result.features[k]
            if isinstance(v1, FeatureStructure) and isinstance(v2, FeatureStructure):
                unified_sub, err = unify(v1, v2)
                if unified_sub is None:
                    return None, f"Clash in nested attribute '{k}': {err}"
                result.features[k] = unified_sub
            elif isinstance(v1, FeatureStructure) or isinstance(v2, FeatureStructure):
                return None, f"Type mismatch at attribute '{k}': atomic vs complex"
            else:
                if v1 == "any":
                    result.features[k] = v2
                elif v2 == "any":
                    result.features[k] = v1
                elif v1 != v2:
                    return None, f"Value mismatch at attribute '{k}': '{v1}' vs '{v2}'"

    return result, None


def run_worked_unification_examples() -> List[Tuple[str, FeatureStructure, FeatureStructure, Optional[FeatureStructure], Optional[str]]]:
    ex1_fs1 = FeatureStructure({"CAT": "NP", "NUM": "sg"})
    ex1_fs2 = FeatureStructure({"PERS": "3", "CASE": "nom"})
    u1, err1 = unify(ex1_fs1, ex1_fs2)

    ex2_sub = FeatureStructure({"NUM": "sg"}, variable_tag="1")
    ex2_fs1 = FeatureStructure({"SUBJ": ex2_sub, "AGR": ex2_sub})
    ex2_fs2 = FeatureStructure({"AGR": FeatureStructure({"PERS": "3"})})
    u2, err2 = unify(ex2_fs1, ex2_fs2)

    ex3_fs1 = FeatureStructure({"CAT": "NP", "NUM": "sg"})
    ex3_fs2 = FeatureStructure({"CAT": "NP", "NUM": "pl"})
    u3, err3 = unify(ex3_fs1, ex3_fs2)

    return [
        ("Example 1: Compatible Disjoint Features", ex1_fs1, ex1_fs2, u1, err1),
        ("Example 2: Reentrant Coreferent Structures", ex2_fs1, ex2_fs2, u2, err2),
        ("Example 3: Conflicting Atomic Values (Failure)", ex3_fs1, ex3_fs2, u3, err3)
    ]
