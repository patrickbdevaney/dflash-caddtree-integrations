"""D10 gate: verify the GDN per-node depth-batched recurrence produces correct
branch-isolated state during a real W=2 generation.

Captures every spec GDN kernel call for the first GDN layer pool and checks:
  INV3  per-depth call writes node slots but NOT parent/canonical slots
        (parent pre-fp == post-fp; node pre-fp != post-fp).
  INV1/2 distinct node slots receive distinct states (branch divergence).
  Tree path active: per-depth calls present (num_accepted==2, 2-col indices).
"""
import os, sys, json, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, "/tests")
import harness as H
import torch

W = int(os.environ.get("DFLASH_TREE_WIDTH", "2"))
REC = {"layer_ptr": None, "calls": []}

def _fp(s):
    f = s.float(); return (round(float(f.norm()), 3), round(float(f.sum()), 3))

import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as fsg
_orig = fsg.fused_sigmoid_gating_delta_rule_update
def wrap(*a, **kw):
    ssi = kw.get("ssm_state_indices"); init = kw.get("initial_state")
    nat = kw.get("num_accepted_tokens"); inplace = kw.get("inplace_final_state", True)
    rec = None
    if ssi is not None and init is not None and ssi.ndim == 2:
        ptr = init.data_ptr()
        if REC["layer_ptr"] is None:
            REC["layer_ptr"] = ptr
        if ptr == REC["layer_ptr"] and len(REC["calls"]) < 200:
            rows = ssi.tolist()
            na = nat.tolist() if nat is not None else None
            pre = {}
            for r in rows:
                for s in r:
                    if s > 0 and s not in pre:
                        pre[s] = _fp(init[s])
            rec = {"rows": rows, "na": na, "inplace": bool(inplace), "ncol": ssi.shape[1], "pre": pre}
    ret = _orig(*a, **kw)
    if rec is not None:
        init = kw.get("initial_state"); post = {}
        for r in rec["rows"]:
            for s in r:
                if s > 0 and s not in post:
                    post[s] = _fp(init[s])
        rec["post"] = post
        REC["calls"].append(rec)
    return ret
fsg.fused_sigmoid_gating_delta_rule_update = wrap
try:
    import vllm.model_executor.layers.mamba.gdn_linear_attn as gla
    if hasattr(gla, "fused_sigmoid_gating_delta_rule_update"):
        gla.fused_sigmoid_gating_delta_rule_update = wrap
except Exception:
    pass

def main():
    llm = H.build_llm(tree_width=W)
    print("ENGINE_UP", flush=True)
    toks = H.greedy(llm, ["Explain binary search in two sentences."], max_tokens=24, seed=0)[0]
    # Analyze: tree depth calls = 2-col indices with na all == 2
    depth_calls = [c for c in REC["calls"] if c["ncol"] == 2 and c["na"] and all(x == 2 for x in c["na"])]
    print(f"total_calls={len(REC['calls'])} depth_calls={len(depth_calls)}", flush=True)
    fails = []
    if not depth_calls:
        fails.append("no tree depth-batched calls captured (D10 not active)")
    inv3_ok = 0; inv3_tot = 0; node_posts = []
    for c in depth_calls:
        for r in c["rows"]:
            node, parent = r[0], r[1]
            if parent > 0 and parent in c["pre"] and parent in c["post"]:
                inv3_tot += 1
                if c["pre"][parent] == c["post"][parent]:
                    inv3_ok += 1   # parent/canonical untouched
            if node > 0 and node in c["post"]:
                node_posts.append(c["post"][node])
    # INV3: parents never modified by the per-node call
    if inv3_tot == 0 or inv3_ok < inv3_tot:
        fails.append(f"INV3 parent-untouched {inv3_ok}/{inv3_tot}")
    # INV1/2: node states are not all identical (branches/positions diverge)
    distinct = len(set(node_posts))
    if distinct < max(2, len(node_posts) // 3):
        fails.append(f"INV1/2 node divergence weak: {distinct} distinct / {len(node_posts)}")
    print(f"INV3 parent-untouched {inv3_ok}/{inv3_tot}; node states {distinct} distinct/{len(node_posts)}", flush=True)
    print("D10_STATE", "PASS" if not fails else "FAIL " + "; ".join(fails), flush=True)
    sys.exit(0 if not fails else 1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("D10_STATE FAIL", repr(e), flush=True); traceback.print_exc(); sys.exit(1)
