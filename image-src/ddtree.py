# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree best-first tree construction over DFlash's parallel-draft marginals.

DFlash emits, in ONE parallel forward, K position-independent marginal
distributions q_1..q_K (one per speculative position). Per the factorized-joint
assumption (arXiv 2604.12989), the joint probability of a candidate path
Y_{1:L} is Q(Y) = prod_i q_i(Y_i). This module runs DDTree Algorithm 1
(best-first heap expansion) over the per-position top-W candidates to select a
prefix-closed tree of at most `node_budget` nodes, then emits the flattened
token batch + ancestor structure used for tree verification.

Pure CPU / no torch dependency for the core algorithm so it is unit-testable in
isolation; the caller converts the outputs to tensors.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    token_id: int
    depth: int           # 0 = root (last accepted token), 1..K = draft positions
    parent: int          # index into nodes[] (-1 for root)
    cum_logp: float      # sum of marginal log-probs along the path to this node
    children: list[int] = field(default_factory=list)


@dataclass
class DDTree:
    nodes: list[TreeNode]
    # flattened verification order (root first, parents before children)
    token_ids: list[int]
    parent_index: list[int]        # parent position in the flattened order (-1 root)
    depth: list[int]
    leaf_flags: list[bool]
    # branch (root->leaf) paths as lists of flattened indices, for GDN chains
    branch_paths: list[list[int]]

    @property
    def size(self) -> int:
        return len(self.nodes)


def build_ddtree(
    root_token: int,
    cand_token_ids: list[list[int]],   # [K][W] candidate token ids per position
    cand_logps: list[list[float]],     # [K][W] marginal log-probs (sorted desc)
    node_budget: int = 22,
    tree_width: int = 2,
) -> DDTree:
    """Best-first expansion (DDTree Algorithm 1) over the factorized joint.

    Returns a prefix-closed tree with <= node_budget nodes (incl. root). Each
    expansion step pops the highest cumulative-logp frontier node and admits it;
    its children (top `tree_width` candidates at the next depth) are pushed.
    """
    K = len(cand_token_ids)
    W = tree_width
    nodes: list[TreeNode] = [TreeNode(token_id=root_token, depth=0, parent=-1, cum_logp=0.0)]

    # frontier heap of candidate nodes not yet admitted: (-score, tie, payload)
    # payload = (token_id, depth, parent_idx, cum_logp)
    heap: list[tuple[float, int, tuple[int, int, int, float]]] = []
    counter = 0

    def push_children(parent_idx: int, parent_depth: int, parent_cum: float):
        nonlocal counter
        d = parent_depth + 1
        if d > K:
            return
        row_t = cand_token_ids[d - 1]
        row_p = cand_logps[d - 1]
        for r in range(min(W, len(row_t))):
            cum = parent_cum + row_p[r]
            heapq.heappush(heap, (-cum, counter, (row_t[r], d, parent_idx, cum)))
            counter += 1

    push_children(0, 0, 0.0)

    while heap and len(nodes) < node_budget:
        _, _, (tok, d, par, cum) = heapq.heappop(heap)
        idx = len(nodes)
        nodes.append(TreeNode(token_id=tok, depth=d, parent=par, cum_logp=cum))
        nodes[par].children.append(idx)
        push_children(idx, d, cum)

    # nodes[] is already in admission order with parents before children
    # (root at 0; a child is admitted only after its parent). Flatten directly.
    token_ids = [n.token_id for n in nodes]
    parent_index = [n.parent for n in nodes]
    depth = [n.depth for n in nodes]
    leaf_flags = [len(n.children) == 0 for n in nodes]

    # branch paths: one per leaf, root->leaf flattened indices
    branch_paths: list[list[int]] = []
    for i, n in enumerate(nodes):
        if leaf_flags[i]:
            path = []
            j = i
            while j != -1:
                path.append(j)
                j = nodes[j].parent
            branch_paths.append(list(reversed(path)))

    return DDTree(
        nodes=nodes, token_ids=token_ids, parent_index=parent_index,
        depth=depth, leaf_flags=leaf_flags, branch_paths=branch_paths,
    )


class DDTreeHeap:
    """Wrapper that turns DFlash per-position draft logits into a DDTree.

    budget: max non-root nodes (tree has at most budget+1 nodes incl. root).
    tree_width: top-W candidates considered per position.
    """

    def __init__(self, budget: int = 12, tree_width: int = 2):
        self.budget = budget
        self.tree_width = tree_width

    def build(self, logits, root_token: int) -> DDTree:
        # Local import so the pure-python core stays torch-free / host-testable.
        import torch

        K = logits.shape[0]
        topw = torch.topk(logits, self.tree_width, dim=-1)
        cand_token_ids = topw.indices.tolist()  # [K][W]
        cand_logps = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(1, topw.indices)
            .tolist()
        )  # [K][W]
        return build_ddtree(
            root_token=int(root_token),
            cand_token_ids=cand_token_ids,
            cand_logps=cand_logps,
            node_budget=self.budget + 1,  # +1 for root
            tree_width=self.tree_width,
        )


def ancestor_mask(tree: DDTree) -> list[list[bool]]:
    """[N][N] boolean: mask[i][j] = node j is an ancestor of node i (or i==j).

    Used to build the FlexAttention tree mask over the spec tokens (each spec
    node attends to its ancestors only; context attended fully separately).
    """
    n = tree.size
    m = [[False] * n for _ in range(n)]
    for i in range(n):
        j = i
        while j != -1:
            m[i][j] = True
            j = tree.parent_index[j]
    return m
