# No-train suite — running finding chain (newest first)

## Item 7 — typical acceptance T=0 guard FIX (2026-06-07)

Audit found typical acceptance ran softmax-threshold accept even at T=0 (no greedy guard).
Added `_is_greedy_lin` guard in `rejection_sampler.forward`: typical path taken only when
NOT greedy. GATE (CUDA graphs): T=0 eps=0.09 -> **BYTE_IDENTICAL** to baseline (20 prompts);
T>0 typical still boosts (tau 3.81->4.62 @T0.3 +21%, 3.54->4.62 @T0.5 +31%). Item 7 moves
YELLOW->GREEN (T=0 lossless gate now met). tok/s in-process is warmup-confounded; real tok/s
deferred to the serve bench.

## Stage -1 / D0 — baseline + C_verify + decision gate (2026-06-07)

**Baseline locked** (20-prompt mixed seed set, tag `linear-opt-baseline`, T=0 token IDs in
`baseline_outputs/baseline.json`). Real baseline:

| mode | T=0 | T=0.3 | T=0.5 |
|---|---|---|---|
| graphs | τ3.99 / 55.9 tok/s | τ3.81 / 53.5 | τ3.54 / 53.7 |
| eager  | τ3.85 / 29.3 | τ3.78 / 29.3 | τ3.51 / 29.5 |

**C_verify(k) measured** (`profiles/thor/c_verify_linear.json`): per-step cost is
**increasing, ≈linear +2.1 ms/token** (47.9 ms @k=1 → 70.8 ms @k=12). tok/s **plateaus
k≈4–12** (52.7–56.3, max at k=12). Verdict: adaptive-K is *applicable* but its upside is
*bounded* by the flat plateau (best static k=12 is already at the top).

**Prefix-hit-rate (D0 part i): NOT MEASURED** — no agentic transcript available. Order
defaulted to single-turn regime: [F, A+E, B, D, G]; Stage C (prefix caching) deferred until
a real agentic trace justifies its large fail-quiet cost.

**Design-first finding carried from Tier 1** (`../IMPLEMENTATION_NOTES.md`): the lossless
acceptance stages (block-verify, per-position temp, fat-chain) are MOOT for DFlash's greedy +
factorized draft. Audit (`LINEAR_OPT_AUDIT.md`): only typical acceptance landed (lossy, T>0,
in-process bench only); it had a missing T=0 greedy guard (fix in progress, item7-t0-guard).
