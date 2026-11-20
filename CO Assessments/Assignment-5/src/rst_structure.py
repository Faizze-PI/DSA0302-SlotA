"""Rhetorical Structure Theory (RST) parse tree representation and nucleus-based summarization."""

from typing import Dict, List, Tuple


class RSTNode:
    def __init__(self, relation: str, nuclearity: str, text: str = "", children: List['RSTNode'] = None):
        self.relation = relation
        self.nuclearity = nuclearity
        self.text = text
        self.children = children or []

    def extract_nuclei(self) -> List[str]:
        if not self.children:
            return [self.text] if "N" in self.nuclearity else []
        res = []
        for c in self.children:
            res.extend(c.extract_nuclei())
        return res


def build_sample_rst_tree() -> RSTNode:
    s1 = RSTNode("SPAN", "N", "Anna Nagar main road is closed due to flooding.")
    s2 = RSTNode("CAUSE", "S", "Water level rose rapidly after monsoon rain.")
    s3 = RSTNode("ELABORATION", "S", "Emergency teams are diverting traffic.")
    sub = RSTNode("Cause", "N-S", children=[s1, s2])
    return RSTNode("Elaboration", "N-S", children=[sub, s3])
