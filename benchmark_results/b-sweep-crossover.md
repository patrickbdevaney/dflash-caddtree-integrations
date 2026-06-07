# B≫K investigation: does tree breadth beat linear? (Jetson AGX Thor SM110a)

Goal: determine whether decoupling tree node budget B from draft depth K
(allowing depth-K spine + breadth) lets DDTree beat linear DFlash. Method:
cheap probes before the large Option-X scheduler rewrite. **Result: no — depth
dominates breadth at feasible budgets for this strong-draft model.**

## Constraint (Option X, proven)

B is capped at K+1=13 by the **verify-token count**: each tree node is a query
token in the verify forward, and #scheduled-verify-tokens = `num_speculative_tokens`
(scheduler KV reservation + runner flat buffers + proposer return shape). No config
shortcut: raising `num_speculative_tokens` also deepens the draft (and is capped by
the DFlash draft `block_size=16`). B ≫ K needs the #42121-class decouple.

## Probe 1 — τ ceiling (offline, from a real W=1 run, 101 steps)

Captured per position: draft top-2 + target greedy. Simulated a depth-12 + W=2
breadth tree:

| metric | value |
|---|---:|
| τ_linear | 6.33 |
| τ_tree_lower (rigorous: branch catches the rejection) | 6.64 (+5%) |
| τ_tree_upper (optimistic: top-1│top-2 along chain) | 7.48 (+18%) |
| branch-catch rate | 31.7% |

Headroom exists but is **thin** (+5–18%), and the offline sim assumed branches
add on top of a full-depth spine (i.e. B ≫ K, no depth sacrifice).

## Probe 2 — real B>K-spine (num_spec=16, spine capped at 12, W=2 → B=17, eager)

| config (eager, 5 prompts × 128 tok) | τ | tok/s |
|---|---:|---:|
| Tree: spine-12 + ~4 branches (B=17) | **4.26** | 17.45 |
| Linear k=16 (= spine-16, depth-only B=17) | **5.82** | 40.42 |
| (ref) Linear k=12 | 6.11 | — |
| (ref) Tree spine-only B=13 | 5.48 | — |

**Multi-branch verification is correct** (diagnostics show accepted paths of
11–12 tokens; ancestor mask verified to block cross-branch attention; GDN
per-node depth routing correct). The τ=4.26 is **not a bug** — it is a *depth
sacrifice*: at fixed budget B=17, capping the spine to 12 to fit 4 branches
throws away draft depth 13–16. **Same-budget head-to-head: depth-16 spine (5.82)
beats depth-12 + 4 branches (4.26).**

## Conclusion

For DFlash's strong draft (τ≈6, ~50% acceptance), **a deeper spine accepts more
than a shallow spine + breadth**. Breadth only helps when the spine errs early
*and* a branch catches it — rare for a good draft (branch-catch 32%, thin). The
probe's +5–18% requires B ≫ K (full-depth spine + extra branch budget = the large
Option-X rewrite), and even that ceiling is thin and uncertain to net a win after
per-step overhead. **The Option-X rewrite is not worth it on this model.**

DDTree tree speculation works correctly on GDN-hybrid MoE but does not beat
strong-draft linear DFlash at feasible node budgets. It would pay off for
*weaker* drafts (lower linear τ → more room for branches to help) or *much
larger* budgets (depth saturated, breadth additive).

## Update — selective vs best-first branch placement (validated)

Selective placement (spare nodes at the highest-uncertainty spine positions,
max logp_rank2 - logp_rank1) vs best-first heap, same budget (num_spec=15,
spine-12, B=16, eager):

| branch strategy | τ | tok/s |
|---|---:|---:|
| **Selective** (uncertainty-targeted) | **4.70** | 18.7 |
| Best-first (heap, ~uniform) | 4.31 | 16.2 |

Selective beats best-first (+0.39 τ, +9%). The user's insight holds: spare
budget belongs at the most-uncertain positions. Absolute τ is still below linear
here because the num_speculative_tokens=15 shortcut DEGRADES the draft (DFlash
conditions on mask-token count, so a 15-mask draft predicts every position worse
than a 12-mask draft) AND the spine-12 cap sacrifices depth 13-15. The clean test
(num_spec=12 draft + selective branches via decoupling) is future work. The
selective strategy itself is the validated improvement over best-first/uniform.
