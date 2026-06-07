## Session outcome: STOPPED — both candidates eliminated (gate is confounded by a mode switch)

### Phase 1 — Diagnosis (config-based, no fail-quiet instrumentation needed)
Witness prompt: seed #0 (earliest first-divergence, pos 72); seed #2 (pos 90) shows the
clearest signature: APC-on `...498, 16, 11` vs APC-off `...498, 17, 11` — an adjacent-token
(`16`↔`17`) **near-tie flip**, i.e. a tiny numeric difference, not gross wrong-state.

**Candidate confirmed: NEITHER (A and B both ELIMINATED).**

Evidence:
1. **Block-size invariance.** APC-on at `mamba_block_size` ∈ {64, 128, 2048} gives the
   *identical* 14/20 diverging prompts at the *identical* positions (72/90/125). At
   block=2048 a ~180-token sequence crosses **no block boundary**, yet diverges identically.
   ⇒ the divergence is independent of the block-cache/boundary path ⇒ **Candidate A (buffer
   mis-size / wrong block gather) and Candidate B (save-boundary off-by-one) are both ruled
   out** — neither can produce a block-size-invariant divergence.
2. **The gate compares two different compute MODES, not on/off caching.** vLLM ties cache
   mode to APC: `config.py:402` "Mamba cache mode is set to **'none'** when prefix caching is
   disabled"; with APC on it forces **'align'**. Forcing `mamba_cache_mode=align` with APC
   *off* is silently reset to `none` → that run was **BYTE_IDENTICAL** to baseline,
   confirming **baseline = none mode**. So P1's "APC-on vs APC-off" = **align vs none**, two
   different GDN recurrence implementations (block-aligned vs contiguous), which differ in
   floating-point accumulation order → greedy near-tie flips that cascade over 128 tokens.

Interpretation: the wrong-state is **not in the GDN recurrent cache** (Candidates A/B). The
P1 bitwise gate as defined is **confounded**: toggling APC simultaneously switches the GDN
compute mode (none↔align), and the *mode switch alone* breaks bitwise T=0. dtype is already
bf16 ("auto"), already ruled out.

### Phase 2 — Fix applied
**NONE.** Per hard-rule 1 + 6: no candidate was confirmed (both eliminated), so no fix is
written. Porting the #39809 Bug-1/Bug-2 cache fixes would NOT help — they address all-mode
crash/wrong-state, not the align-vs-none numeric difference, and the cache path is not the
cause here. Recovery point clean: all tests were config-only via gated harness flags
(`DFLASH_APC`, `DFLASH_MAMBA_MODE`, `DFLASH_MAMBA_BLOCK`); no image-src/kernel change shipped.

### Phase 3 — Bitwise gate
Not re-run (no fix to test). Status unchanged: 14/20 diverge — but now *explained* as a
mode-switch numeric difference, not a cache-state bug.

### Why the gate is the WRONG gate here (key finding)
APC requires align mode; align mode legitimately computes the GDN recurrence in a different
(block-aligned) FP order than none mode. So **bitwise(align-APC-on vs none-APC-off) is
unreachable in principle** on this base — not because caching is wrong, but because the two
modes don't bit-match. The correct correctness check is **cold==warm WITHIN align mode**
(same mode, caching off vs on), which isolates the cache. That requires **decoupling cache
mode from the APC flag** (a small code change to make a user-set `mamba_cache_mode=align`
authoritative even with APC off) so a true cold-vs-warm align run can be diffed. Whether
align-mode is benign-numeric or subtly wrong is then answerable; today it cannot be isolated.

### Next session (exact remaining gap)
1. Patch the `mamba_cache_mode` reset (config.py:402 path / platforms/interface.py) so
   `align` can run with APC OFF → enables the **cold==warm align** bitwise test.
2. If cold==warm align is bitwise-identical → APC caching is correct; the only divergence is
   align-vs-none numerics → adopt align as the baseline and the gate passes by construction
   (compare align-on vs align-off). Then measure hit-rate/e2e (Tier-A reached).
3. If cold!=warm align → THEN it is a real cache-state bug; re-run the A/B instrumentation.
