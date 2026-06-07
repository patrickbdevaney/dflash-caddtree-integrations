"""Fast unit tests for the DDTree best-first tree builder (pure CPU, no model)."""
import math, sys, os
sys.path.insert(0, "/image-src")  # ddtree.py copied here in the test container
import ddtree as D

fails = []
def check(cond, msg):
    print(("[PASS] " if cond else "[FAIL] ") + msg, flush=True)
    if not cond: fails.append(msg)

# K=3 positions, W candidates each; logps sorted desc per position.
cand_t = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
cand_p = [[math.log(0.6), math.log(0.3), math.log(0.1)],
          [math.log(0.7), math.log(0.2), math.log(0.1)],
          [math.log(0.5), math.log(0.4), math.log(0.1)]]

# W=1 must reduce to a linear chain of length K (+root) -> tree == linear path
t1 = D.build_ddtree(root_token=9, cand_token_ids=cand_t, cand_logps=cand_p,
                    node_budget=22, tree_width=1)
check(t1.size == 1 + 3, f"W=1 linear chain size {t1.size} == 4")
check(t1.token_ids == [9, 10, 20, 30], f"W=1 greedy path {t1.token_ids}")
check(len(t1.branch_paths) == 1 and t1.branch_paths[0] == [0, 1, 2, 3],
      "W=1 single root->leaf path")

# W=2, budget 22: prefix-closed, parents before children, best-first by cum logp
t2 = D.build_ddtree(root_token=9, cand_token_ids=cand_t, cand_logps=cand_p,
                    node_budget=22, tree_width=2)
check(t2.size <= 22, f"W=2 within budget ({t2.size})")
# parents always appear before children in flattened order
ok_order = all(t2.parent_index[i] < i for i in range(1, t2.size))
check(ok_order, "parents precede children in flattened order (prefix-closed)")
# root has up to W children
root_children = sum(1 for p in t2.parent_index if p == 0)
check(root_children == 2, f"root has W=2 children ({root_children})")
# every node reachable to root
def reaches_root(i):
    j = i
    while j != -1:
        j = t2.parent_index[j]
    return True
check(all(reaches_root(i) for i in range(t2.size)), "all nodes reach root")

# best-first: the first admitted (index 1) is the globally most probable child
check(t2.token_ids[1] == 20 or t2.token_ids[1] == 10,
      f"first admitted node is highest-prob candidate ({t2.token_ids[1]})")

# budget cap is respected
t3 = D.build_ddtree(root_token=9, cand_token_ids=cand_t, cand_logps=cand_p,
                    node_budget=5, tree_width=3)
check(t3.size == 5, f"budget cap honored ({t3.size} == 5)")

# ancestor mask correctness
m = D.ancestor_mask(t2)
check(all(m[i][i] for i in range(t2.size)), "ancestor mask diagonal true")
check(all(m[i][0] for i in range(t2.size)), "all nodes have root as ancestor")
# a node's parent is an ancestor; a non-ancestor sibling is not
for i in range(1, t2.size):
    p = t2.parent_index[i]
    if not m[i][p]:
        fails.append(f"node {i} parent {p} not marked ancestor")
check(not any(f.startswith("node ") for f in fails), "parent always ancestor")

print("DDTREE_RESULT", "ALL_OK" if not fails else "FAIL " + "; ".join(fails))
sys.exit(1 if fails else 0)
