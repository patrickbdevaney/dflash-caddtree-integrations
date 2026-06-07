# No-train suite — cumulative summary (real measurements, Jetson Thor SM110a)
Baseline: `linear-opt-baseline` (20-prompt mixed seed set). Model Qwen3.6-35B-A3B-NVFP4 + DFlash.

## What actually moved (and what didn't)

| Stage | Outcome | Status | byte-id @T=0 |
|-------|---------|--------|--------------|
| Item 7 typical acceptance | **+21% τ @T0.3, +31% τ @T0.5** (T=0 guard added) | **GREEN, landed** | YES |
| -1/D0 baseline + C_verify | baseline locked; C_verify increasing (+2.1ms/tok), tok/s plateaus k≈4–12 | done | — |
| F NVFP4 audit | **CLEAN — no dequant fallback**; MoE FP4 GEMM dominates (~34%) | GREEN | YES (kernel sel. unchanged) |
| A2 verify fusion | **already fused** (single kernel/layer on linear path) | pre-existing | YES |
| E expert prefetch | **moot on Thor** (weights resident in unified mem; GEMM bandwidth-bound) | documented moot | — |
| A1 GDN rollback | correctness **assertion designed**; implicit mechanism believed correct | designed, impl deferred | — |
| 3 block-verify / 4 pos-temp / 6 fat-chain / D top-p | **moot for greedy+factorized draft** (proofs in IMPLEMENTATION_NOTES) | declined by design | — |
| 5 adaptive-K | applicable (C_verify increasing) but **bounded** by tok/s plateau | not implemented | — |
| B CPU suffix drafter | genuine lossless win for repetitive/agentic text; **not implemented** (large async) | actionable TODO | — |
| C GDN prefix caching | **deferred** (agentic-only; no trace to justify fail-quiet cost) | deferred | — |
| G graph capture tuning | minor; harness captures [1,K+1]; no gap found | minor | — |

## Honest bottom line

For **this specific setup** (DFlash *greedy* + *factorized* draft, Thor *unified memory*,
*already-fused* linear GDN verify), most of the suite's stages are **moot, already-done, or
deferred** — a faithful negative-heavy characterization, not a failure:
- The acceptance-side lossless stages (block-verify, pos-temp, fat-chain, top-p) cannot help
  a deterministic factorized draft (exact-match is already lossless-optimal).
- The verify-cost stages are largely satisfied or device-moot (A2 fused; E moot on unified
  memory; F already FP4).
- **The one landed win is typical acceptance** (lossy, opt-in, T>0): **+21–31% τ**, now
  byte-identical at T=0. Validated in-process; production serve bench (35B/122B/27B, T=0 and
  T>0) is the final step.
- **Genuine remaining opportunities:** B (CPU suffix drafter, repetitive text) and C (prefix
  caching, agentic) — both substantial follow-ups, design-docs in /designs/.

state-mismatch assertion count: 0 (no mismatches observed; A1 formal proof designed not run).
FP4 dequant fallback: none found. RED items: 0.
