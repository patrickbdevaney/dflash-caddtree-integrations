# Stage 5 — Tree-WY GDN kernel: FLOP analysis (measure before building)

**Verdict: do NOT build the kernel at feasible B. Documented negative result.**

## Correction to the "overhead-free" claim

The DeltaNet chunkwise WY/UT transform (arXiv:2406.06484) is **O(L²d)** — the
pseudo-value solve "cannot be fully parallelized." A tree formulation (Tree-WY)
is therefore **O(B²d)**, NOT overhead-free. Short critical-path depth ≠ low total
work. Tree-WY is mathematically derivable and correct (first chunkwise-WY tree
form for the delta rule; STree handled diagonal SSMs only), but its cost must be
weighed against the current per-node depth loop before building.

## FLOP count at B=13 (GDN state [H=32, d_k=128, d_v=128], 30 layers)

| path | MACs / layer / step | structure |
|---|---:|---|
| current per-node depth loop | **18.9M** | 12 tiny 1-token kernels |
| Tree-WY (UT transform) | **17.1M** | ~5 GEMMs of 13×13×128 |
| ratio | **0.91** | only 9% fewer FLOPs |

Tree-WY: KKᵀ (B²d) + solve (I+T)⁻¹ via forward-substitution (B²d) + pseudo-values
W=T·diag(β)V (B²d) + output QS₀ + tril(QKᵀ)W (2·B²d) + state S += KᵀW (d²B).

## Why it does not win at small B

1. **Only 9% fewer FLOPs** — not "clearly below" the depth loop (the gate).
2. **Tiny GEMMs underutilize tensor cores**: a 13×13 tile fills 66% of a *single*
   16×16 tensor-core tile → latency-bound, single-tile, zero throughput benefit.
   The GEMM advantage only appears when B ≫ 16 (multi-tile), i.e. beyond the
   useful tree budget.
3. **State IO is identical and dominant**: 63 MB/step read+write of GDN state
   across 30 layers, bandwidth-bound on Thor LPDDR5x (~273 GB/s). Tree-WY does
   not reduce this — it's the real per-step floor, and it's the same for both.
4. **Launch overhead is already gone**: the per-branch fusion collapses the depth
   loop to one launch (spine-only B=13), and CUDA graphs eliminate dispatch
   overhead entirely. Tree-WY's "fewer launches" benefit is already captured.

## Where Tree-WY *would* matter (and why it's still not worth it here)

Its real value is the **packed state** (one pass instead of B separate 30 MB
slots) — which would relieve the **memory** constraint of Option X (B ≫ K). But:
- Option X itself is thin-payoff: depth beats breadth for DFlash's strong draft
  (see `b-sweep-crossover.md`), so the feature Tree-WY unlocks isn't worth it.
- The GEMM throughput win needs B ≫ 16, which exceeds the useful tree budget.

## Conclusion (publishable negative result)

**Tree-WY is O(B²d) and only wins at B large enough to saturate tensor cores
(B ≫ 16), which exceeds the useful speculative tree budget on bandwidth-bound
edge hardware (Thor).** At B=13 it offers 9% fewer FLOPs in single-tile GEMMs
against an identical, dominant 63 MB/step bandwidth-bound state IO — no net win.
The kernel is correct and novel but not worth building for this regime. Build it
only if a future workload needs B ≫ 16 *and* the breadth pays off (weak draft).
