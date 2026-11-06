"""Parse tree representation, tree metrics, and leftmost/rightmost derivation generator."""

from typing import List, Optional, Tuple


class ParseNode:
    def __init__(self, symbol: str, children: Optional[List['ParseNode']] = None, token: Optional[str] = None, prob: float = 1.0):
        self.symbol = symbol
        self.children = children if children is not None else []
        self.token = token
        self.prob = prob

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def height(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(c.height() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def rule_applications(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + sum(c.rule_applications() for c in self.children)

    def yield_tokens(self) -> List[str]:
        if self.token is not None:
            return [self.token]
        tokens = []
        for c in self.children:
            tokens.extend(c.yield_tokens())
        return tokens

    def to_bracketed(self) -> str:
        if self.token is not None:
            return f"({self.symbol} {self.token})"
        if not self.children:
            return f"({self.symbol})"
        inner = " ".join(c.to_bracketed() for c in self.children)
        return f"({self.symbol} {inner})"

    def to_tree_string(self, prefix: str = "", is_last: bool = True) -> str:
        label = f"{self.symbol} -> {self.token}" if self.token else self.symbol
        connector = "└── " if is_last else "├── "
        res = [prefix + connector + label]
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            res.append(child.to_tree_string(new_prefix, i == len(self.children) - 1))
        return "\n".join(res)


def from_bracketed(text: str) -> ParseNode:
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    stack: List[List[ParseNode]] = []
    sym_stack: List[str] = []

    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            children = stack.pop()
            sym = sym_stack.pop()
            if len(children) == 1 and children[0].symbol == children[0].token:
                node = ParseNode(sym, token=children[0].token)
            else:
                node = ParseNode(sym, children=children)
            if stack:
                stack[-1].append(node)
            else:
                return node
        else:
            if len(stack) > len(sym_stack):
                sym_stack.append(t)
            else:
                stack[-1].append(ParseNode(t, token=t))
    return ParseNode("ROOT")


def leftmost_derivation(root: ParseNode) -> List[str]:
    steps = [root.symbol]
    frontier = [root]
    while True:
        leftmost_idx = -1
        for idx in range(len(frontier)):
            n = frontier[idx]
            if n.token is None and n.children:
                leftmost_idx = idx
                break
        if leftmost_idx == -1:
            break
        target = frontier[leftmost_idx]
        frontier = frontier[:leftmost_idx] + target.children + frontier[leftmost_idx+1:]
        sent = " ".join(n.token if n.token is not None else n.symbol for n in frontier)
        steps.append(sent)
    return steps


def rightmost_derivation(root: ParseNode) -> List[str]:
    steps = [root.symbol]
    frontier = [root]
    while True:
        rightmost_idx = -1
        for idx in range(len(frontier) - 1, -1, -1):
            n = frontier[idx]
            if n.token is None and n.children:
                rightmost_idx = idx
                break
        if rightmost_idx == -1:
            break
        target = frontier[rightmost_idx]
        frontier = frontier[:rightmost_idx] + target.children + frontier[rightmost_idx+1:]
        sent = " ".join(n.token if n.token is not None else n.symbol for n in frontier)
        steps.append(sent)
    return steps


def build_ambiguous_pp_trees() -> Tuple[ParseNode, ParseNode]:
    tree_np = from_bracketed(
        "(S (VP (V_trans show) (NP (NP (Det the) (Nom (N students))) (PP (P with) (NP (Det the) (Nom (Adj late) (Nom (N fee) (Nom (N notice)))))))))"
    )
    tree_vp = from_bracketed(
        "(S (VP (VP (V_trans show) (NP (Det the) (Nom (N students)))) (PP (P with) (NP (Det the) (Nom (Adj late) (Nom (N fee) (Nom (N notice))))))))"
    )
    return tree_np, tree_vp


def build_ambiguous_coordination_trees() -> Tuple[ParseNode, ParseNode]:
    tree_left = from_bracketed(
        "(S (VP (V_trans submit) (NP (NP (NP (Det the) (Nom (N form))) (Conj and) (NP (Det the) (Nom (N fee) (Nom (N receipt))))) (Conj or) (NP (Det the) (Nom (N card))))))"
    )
    tree_right = from_bracketed(
        "(S (VP (V_trans submit) (NP (NP (Det the) (Nom (N form))) (Conj and) (NP (NP (Det the) (Nom (N fee) (Nom (N receipt)))) (Conj or) (NP (Det the) (Nom (N card)))))))"
    )
    return tree_left, tree_right
