# Innovation Suite — overnight run notes (newest first)

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
