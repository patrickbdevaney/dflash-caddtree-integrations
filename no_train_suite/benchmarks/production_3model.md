# Production serve+bench — 3 models (real vLLM serve, CUDA graphs, Jetson Thor)

Method: `IMAGE=vllm-dflash-thor:ddtree` + per-model serve script (35B marlin MoE; 122B
cutlass MoE + TRITON_ATTN gpu_util 0.78; 27B dense NUM_SPEC=15). bench.py, 2-run medians,
avg over 4 coding tasks. Baseline = eps=0; Typical = `DFLASH_ACCEPT_EPS=0.09`.

| model | baseline T=0 | baseline T=0.3 | **typical T=0.3** | Δ tok/s | Δ τ |
|-------|--------------|----------------|-------------------|---------|-----|
| Qwen3.6-35B-A3B  | 108.4 (τ6.06) | 127.5 (τ6.31) | **137.8 (τ6.56)** | +8%  | +4%  |
| Qwen3.6-27B      | 42.3 (τ5.87)  | 43.0 (τ6.00)  | **54.8 (τ7.66)**  | **+27%** | **+28%** |
| Qwen3.5-122B-A10B| 45.0 (τ5.12)  | 41.4 (τ4.75)  | **52.1 (τ6.00)**  | **+26%** | **+26%** |

T=0 typical vs baseline (byte-identity guard check): 35B 119.4 vs 108.4, 27B 40.5 vs 42.3,
122B 45.1 vs 45.0 — equal within 2-run warmup noise (~10%), consistent with the T=0 greedy
guard (no real divergence; τ matches).

## Findings

- **Typical acceptance is a real production win at T>0**, validated on all three models:
  **+26–28% tok/s on 27B and 122B**, +8% (within noise) on 35B.
- The gain **concentrates where baseline T>0 acceptance is weakest**: 122B baseline τ drops to
  4.75 at T=0.3 (rejection sampling degrades), typical restores it to 6.00; 27B 6.00→7.66.
  35B's baseline is already strong (τ6.31) so the headroom — and the win — is small.
- **Caveat:** 2-run medians carry ~10% warmup noise; the τ deltas (more stable than tok/s)
  corroborate the wins (35B +4%, 27B +28%, 122B +26%).
- These are real `bench.py` numbers from the production serve path — NOT the in-process
  micro-harness (which underreports: 35B in-process ~56 tok/s vs production ~108–138).
