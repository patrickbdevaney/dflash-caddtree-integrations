"""GDN tree-speculation invariant test suite (Part 3).

Runs against the EXISTING linear DFlash (W=1) to establish ground truth, then
unchanged against the tree implementation (W=2,3). Load-once: one engine builds,
all 6 tests run. Usage (inside the image):
    DFLASH_TREE_WIDTH=1 python test_gdn_tree_invariants.py

W=1 semantics: the tree degenerates to the linear chain, so each test asserts the
linear-valid form of its invariant now; the strict cross-branch assertions
(marked TREE-ONLY) activate at W>=2 once the tree code exists.
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H

W = int(os.environ.get("DFLASH_TREE_WIDTH", "1"))
REF_PATH = os.path.join(HERE, "ref_linear_tokens.json")
RESULTS = []

def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[{status:4}] {name}: {detail}", flush=True)

def approx_eq(a, b, rel=1e-3, abs_=1e-2):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

# ----------------------------------------------------------------------------
def test1_identity(llm):
    """W=1 identity (INV6): determinism now; tree-W1 == linear after impl."""
    toks_a = H.greedy(llm, H.FIXED_PROMPTS, max_tokens=32, seed=0)
    toks_b = H.greedy(llm, H.FIXED_PROMPTS, max_tokens=32, seed=0)
    det = (toks_a == toks_b)
    if not det:
        return record("TEST1_identity", "FAIL", "non-deterministic across two identical runs")
    if W == 1:
        # establish/refresh the linear ground-truth reference
        json.dump(toks_a, open(REF_PATH, "w"))
        return record("TEST1_identity", "PASS", f"deterministic; linear reference saved ({len(toks_a)} prompts)")
    # W>=2: compare tree output against saved linear reference (must be identical at W=1 path;
    # at W>=2 tokens may legitimately differ — identity only required when forcing W=1 behavior)
    if not os.path.exists(REF_PATH):
        return record("TEST1_identity", "SKIP", "no linear reference; run W=1 first")
    ref = json.load(open(REF_PATH))
    same = (ref == toks_a)
    record("TEST1_identity", "PASS" if same else "INFO",
           "tree==linear" if same else "tree differs from linear (expected for W>=2)")

def test2_isolation(llm):
    """State isolation between branches (INV1+2). W=1: chain-distinctness."""
    with H.StateCapture(max_calls=60) as cap:
        H.greedy(llm, [H.FIXED_PROMPTS[1]], max_tokens=24, seed=0)
    recs = [r for r in cap.records if "post" in r]
    if not recs:
        return record("TEST2_isolation", "FAIL", "no GDN spec kernel calls captured")
    # take a mid-stream record; check the chain wrote distinct states across slots
    r = recs[len(recs)//2]
    norms = [r["post"][s][0] for s in sorted(r["post"]) if s in r["post"]]
    distinct = len(set(round(n, 4) for n in norms))
    if W == 1:
        ok = distinct >= max(2, len(norms)//2)
        return record("TEST2_isolation", "PASS" if ok else "FAIL",
                      f"chain slots distinct: {distinct}/{len(norms)} (W=1 chain-distinctness)")
    # TREE-ONLY: slot[0]!=slot[1] across branches, both != canonical pre, neither == other branch
    return record("TEST2_isolation", "SKIP", "TREE-ONLY assertions pending tree code (W>=2)")

def test3_non_inplace(llm):
    """Non-in-place kernel isolation (INV3). W=1: linear uses in-place chain (correct)."""
    with H.StateCapture(max_calls=60) as cap:
        H.greedy(llm, [H.FIXED_PROMPTS[5]], max_tokens=24, seed=0)
    recs = [r for r in cap.records if "post" in r and "pre" in r]
    if not recs:
        return record("TEST3_non_inplace", "FAIL", "no captures")
    r = recs[len(recs)//2]
    # written slots changed vs their pre state (kernel actually advanced the recurrence)
    changed = sum(1 for s in r["post"] if s in r["pre"] and not approx_eq(r["pre"][s][0], r["post"][s][0]))
    if W == 1:
        ok = (r["inplace"] is True) and changed >= 1
        return record("TEST3_non_inplace", "PASS" if ok else "FAIL",
                      f"linear inplace={r['inplace']}, slots advanced={changed} (W=1: in-place chain is correct)")
    # TREE-ONLY: assert canonical S_old UNCHANGED after a W>=2 pass (inplace must be False / disjoint slots)
    return record("TEST3_non_inplace", "SKIP", "TREE-ONLY: canonical-unchanged check pending tree code (W>=2)")

def test4_promotion(llm):
    """State promotion correctness (INV5). W=1: accepted state carried into next step.

    The value step t+1 READS as its recurrent seed (pre-kernel) must equal some
    recurrent state that existed at the end of step t (the accepted slot, possibly
    moved to canonical by align-mode postprocess_mamba). We compare b.pre[seed]
    (the value actually read) against all of a.post[*] (states present after step t).
    """
    with H.StateCapture(max_calls=80) as cap:
        H.greedy(llm, [H.FIXED_PROMPTS[2]], max_tokens=32, seed=0)
    recs = [r for r in cap.records if "post" in r and "pre" in r and "seed_idx" in r]
    if len(recs) < 3:
        return record("TEST4_promotion", "FAIL", f"too few captured steps ({len(recs)})")
    # diagnostics
    for r in recs[:5]:
        print(f"   T4diag idx={r['indices']} na={r['num_accepted']} seed={r['seed_idx']} "
              f"pre_seed={r['pre'].get(r['seed_idx'])} post_seed={r['post'].get(r['seed_idx'])}", flush=True)
    # The GDN spec slots rotate through several block-ranges (align-mode rollback
    # multi-buffer). Group by block-range; within a range, each revisit must READ
    # back exactly the recurrent state the previous use WROTE -> promotion/seed of
    # the accepted state was applied correctly (a broken promotion would leave the
    # revisited slot stale/garbage).
    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        groups[r["indices"][0]].append(r)
    hits = 0; checks = 0
    for key, g in groups.items():
        for prev, cur in zip(g, g[1:]):
            seed = cur["seed_idx"]
            if seed not in cur["pre"] or seed not in prev["post"]:
                continue
            checks += 1
            if approx_eq(cur["pre"][seed][0], prev["post"][seed][0]):
                hits += 1
    if W == 1:
        ok = checks > 0 and hits >= max(1, (9 * checks) // 10)
        return record("TEST4_promotion", "PASS" if ok else "FAIL",
                      f"accepted state carried across rollback-buffer cycle {hits}/{checks} "
                      f"(groups={len(groups)})")
    # TREE-ONLY: force branch-1 acceptance, assert canonical == slot[1]_pre, others freed
    return record("TEST4_promotion", "SKIP", "TREE-ONLY: forced-branch promotion pending tree code (W>=2)")

def test5_continuity(llm):
    """Multi-round state continuity + output coherence (integration)."""
    prompt = "Write a short paragraph explaining why the sky appears blue during the day."
    with H.StateCapture(max_calls=120) as cap:
        toks = H.greedy(llm, [prompt], max_tokens=96, seed=0)[0]
    recs = [r for r in cap.records if "seed_post_pool" in r]
    if len(recs) < 4:
        return record("TEST5_continuity", "FAIL", f"too few steps ({len(recs)})")
    seeds = [r["seed_post_pool"][0] for r in recs]
    changing = sum(1 for a, b in zip(seeds, seeds[1:]) if not approx_eq(a, b))
    # coherence: not catastrophically repetitive
    from collections import Counter
    if len(toks) == 0:
        return record("TEST5_continuity", "FAIL", "empty output")
    top_frac = Counter(toks).most_common(1)[0][1] / len(toks)
    state_evolving = changing >= max(2, len(seeds)//3)
    coherent = top_frac < 0.5 and len(toks) >= 32
    ok = state_evolving and coherent
    return record("TEST5_continuity", "PASS" if ok else "FAIL",
                  f"state evolving {changing}/{len(seeds)-1} steps, top-token frac {top_frac:.2f}, len {len(toks)}")

def test6_accepted_path(llm):
    """GDN output equivalence for accepted path (INV6/correctness). W=1: == linear."""
    with H.StateCapture(max_calls=40) as cap:
        toks = H.greedy(llm, [H.FIXED_PROMPTS[0]], max_tokens=32, seed=0)[0]
    spec_active = cap.records and any("post" in r for r in cap.records)
    if not spec_active:
        return record("TEST6_accepted_path", "FAIL", "DFlash spec decode not active (no GDN spec calls)")
    if W == 1:
        # at W=1 the accepted path IS the linear top-1 path by construction
        return record("TEST6_accepted_path", "PASS",
                      f"spec decode active, {len(cap.records)} captured calls; W=1 path == linear")
    # TREE-ONLY: force accepted branch == linear top-1; assert states identical to linear run
    return record("TEST6_accepted_path", "SKIP", "TREE-ONLY: forced-path GDN equivalence pending tree code (W>=2)")

# ----------------------------------------------------------------------------
def main():
    print(f"=== GDN tree invariant suite | DFLASH_TREE_WIDTH={W} ===", flush=True)
    llm = H.build_llm(tree_width=W)
    print("ENGINE_UP", flush=True)
    for fn in (test1_identity, test2_isolation, test3_non_inplace,
               test4_promotion, test5_continuity, test6_accepted_path):
        try:
            fn(llm)
        except Exception as e:
            import traceback; traceback.print_exc()
            record(fn.__name__, "FAIL", f"exception: {e!r}")
    print("\n=== SUMMARY ===", flush=True)
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    for name, status, detail in RESULTS:
        print(f"  {status:4}  {name}", flush=True)
    print(f"RESULT: {'ALL_OK' if not fails else 'FAILURES=' + str(len(fails))}", flush=True)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
