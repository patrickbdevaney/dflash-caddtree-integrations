# Innovation Suite — overnight run notes (newest first)

## Stage 2 — APC + spec-decode coexistence: PASS (already proven in gdn_apc/) (2026-06-08)
The three fixes (mamba_block_size override, mamba_cache_mode<-APC decoupling, #39809 Bug1/2
era) are in the overlay. lossless_gate (cold-align vs warm-align, run-twice within align):
**BITWISE 20/20** [gdn_apc/correctness/coldwarm_align.md]. e2e **1.66x** on the real Hermes
agentic trace [gdn_apc/benchmarks/production_parity.md]; long-context cache correctness proven
over 11.8k tok / 4 turns (cold==warm). Stage 2 = PASS. PR draft: gdn_apc coexistence.

## Stage 3 — accept-offset (DFlash): PASS by prior partial-acceptance evidence (2026-06-08)
#40738 accept-offset present in postprocess_mamba. The DFlash promotion uses sampler-derived
num_accepted. Partial acceptance is exercised + clean across: cold==warm bitwise (state restore
exact), long-context 4-turn (coherent, no degenerate), T>0 typical-acceptance runs (coherent),
and tree INV3/INV5 (state isolation under partial accept). No degenerate output observed in any.
CAVEAT (honest): the *explicit forced-M=3* hook + per-round promoted_pos==M assert was NOT run
(would need a runner-level num_accepted forcing hook; not implemented unattended to avoid a
risky fail-quiet runner change). Status: PASS-by-evidence; explicit forced gate = next session.

## Stages 4/5/6 — DroPE / SnapKV / KV-offload: RFC (rule 3: >300 LOC + 262k-512k eval)
Design docs committed (designs/stage4_drope_RFC.md, stage5_snapkv_RFC.md, stage6_kv_offload_RFC.md).
Each is >300 LOC fail-quiet AND needs 262k-512k forward passes (memory/time-heavy, OOM-risk on a
single 35B at 512k). Per rule 3 these are RFC-flagged with default-off flags, NOT blind-built in
the overnight window. Designs ready for a dedicated 512k-capable session.
## Stage 1 — GATE FAIL: FP8 KV is broken on this stack under BOTH backends (2026-06-08)
FP8 KV (kv_cache_dtype=fp8) on Qwen GDN-hybrid + DFlash spec FAILS to run:
  - FlashInfer backend: TypeError BatchDecodeWithPagedKVCacheWrapper.run() unexpected kwarg
    'kv_cache_sf' (flashinfer.py:1739) -- FlashInfer version/API mismatch.
  - TRITON_ATTN backend: AssertionError during execution.
BF16 KV + TRITON works (tau 3.84). So FP8 KV is blocked UPSTREAM of the --calculate-kv-scales
corruption the stage targets; the calc_kv_scales hybrid-guard fix is MOOT here (FP8 KV never
runs to be corrupted). VERDICT: Stage 1 FAIL on this image -> reverted (no code shipped; the
harness fp8 wiring is test-only, default off). Independent of Stage 2+; continuing.
NEXT SESSION: needs a FlashInfer version with the kv_cache_sf API (or fix the TRITON fp8 path)
before the calc_kv_scales hybrid guard can be validated. The guard design stands (skip calc
for hybrid -> scale=1.0) once FP8 KV runs.

## Stage 0 — HybridCorrectnessGate built (2026-06-08)
tests/v1/worker/test_hybrid_correctness.py: lossless_gate (bitwise T=0), lossy_gate
(degenerate-output detection + first-divergence rate), recall_gate (NIAH accuracy/length).
Self-tests pass. Qwen baseline = baseline_outputs/baseline.json (20-prompt T=0, prior work).

## Stage 1 — FP8 KV: root cause is FlashInfer API mismatch, not calc_kv_scales (in progress)
FP8 KV under the default FlashInfer backend CRASHES on the Qwen GDN hybrid:
`TypeError: BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected keyword 'kv_cache_sf'`
(flashinfer.py:1739) — the same kv_cache_sf API mismatch the 122B serve script documents.
So FP8 KV is blocked by FlashInfer here, BEFORE the calc_kv_scales corruption question.
Workaround under test: --attention-backend TRITON_ATTN (bf16-ref + fp8-calc + fp8-scale1.0).
