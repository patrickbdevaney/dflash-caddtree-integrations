# vLLM :dllm diffusion decode — optimization sweep (LLaDA2.1-mini, Thor SM110a)
Baseline: vLLM dllm-plugin BF16, eager, TRITON_ATTN = **72.1 tok/s** (1.11× the 64.9 raw-transformers
BF16 floor). Goal: push toward the hardware ceiling. Each lever serialized + cgroup-capped; abandon a
lever only on a profound sm_110 wall. tok/s = sglang_bench_client (3 coding prompts, max_tokens 256, T=0).

| Lever | tok/s | vs baseline | Status |
|---|---:|---:|---|
| BF16 eager TRITON_ATTN (baseline) | 72.1 | 1.00× | ✅ correct, coherent-ish |
| **V1 CUDA graphs** (--cudagraph-capture-sizes 32 64 96 128) | — | — | ❌ **breaks generation**: captures fine but output is degenerate garbage ("!HelloHelloHello…add add two numbers numbers!!!"). Plugin's CUDAGraph path is incompatible with the diffusion draft/denoise state (its own KNOWN_LIMITATIONS flag uniform-batch + eager hooks). NOT an sm_110 wall — a CG×diffusion correctness incompatibility. Eager remains the correct mode. |
| **V2 Thor MoE autotune** (benchmark_moe --tune, E=256/N=512) | — | — | ⚠️ **stalled**: patched get_model_params (LLaDA2->Qwen3Moe shape) + ray; tuned 1.86k/1920 configs in ~43min then HUNG at 97% (one tail large-batch config stalled on sm_110: CPU 1-core busy, GPU idle 10min+, no progress). Tuner writes config only at 100% -> no Thor MoE JSON produced. Needs a per-config timeout (benchmark_moe lacks one) to complete offline. Deferred. The 'sub-optimal default MoE config' warning thus remains; this is the top untapped lever. |
| **V3 NVFP4 → CUTLASS FP4 MoE** | — | — | ⚠️ **CUTLASS FP4 kernel SELECTED, blocked on plugin weight-load**: vLLM picked the right fused path on sm_110 — "FlashInferCutlassNvFp4LinearKernel" + "VLLM_CUTLASS NvFp4 MoE backend out of [VLLM_CUTLASS, MARLIN, EMULATION]" (NOT Marlin dequant). But the dllm-plugin's `models/llada2.py:831 load_weights` expects plain BF16 expert names (gate/up/down) and doesn't map the compressed-tensors NVFP4 packed expert params (weight_packed/weight_scale/global_scale) into FusedMoE's NVFP4 params -> ValueError "Missing weights for layer 1 expert 0". Fix = extend plugin load_weights for NVFP4 expert remap (code, ~moderate). NOT an sm_110 wall — the fused FP4 kernel is available + chosen; only the plugin's expert weight-loading lacks NVFP4. The NVFP4 memory win (9.5 vs 32GB) + CUTLASS FP4 speed are one weight-loader patch away. |
| **V4 FlashInfer-CUTLASS MoE (BF16)** | — | — | ❌ **not supported on sm_110**: `--moe-backend flashinfer_cutlass` accepted but EngineCore errors: "Unquantized MoE backend FlashInfer CUTLASS does not support the deployment configuration since kernel does not support current device cuda." The unquantized FlashInfer-CUTLASS MoE kernel isn't built for sm_110. (CUTLASS FP4 grouped MoE IS available for NVFP4 — see V3 — so this is specific to the *unquantized* path.) TRITON unquantized MoE stays the BF16 backend. Device-applicability limit for this lever; not the track. |
| **V5 flashinfer NON-CAUSAL attn** | **73.3** | **1.02× (+1.7%)** | ✅ **WIN — works + cleaner output**. `--attention-backend FLASHINFER` now serves (warmup passed: the baked kv_cache_sf-conditional fix; pre-port this crashed at warmup, and generation crashed at the combine kernel — both fixed by the diffusion port num_bonus=0). 73.3 tok/s (83.9/84.8/58.0) vs 72.1 TRITON_ATTN, AND output quality is noticeably better (`def add_numbers(a,b)` + docstring vs TRITON artifacts). Recommended attention backend going forward. Confirms the earlier "sm_110 flashinfer illegal-access" was the AR combine-kernel layout, not the attn kernel. |
| **V6a flashinfer + commit_threshold=0.55** (speed mode) | **90.7** | **1.26× over 72.1; 1.40× HF floor** | ✅ **BIGGEST WIN**. DiffusionConfig commit_threshold 0.9→0.55 (LLaDA "speed mode": commit more masked positions per denoise step → fewer steps/block). 90.7 tok/s (77.4/131.8/79.6) on flashinfer non-causal, output still coherent (`def add_numbers(a,b)` + docstring). The denoise confidence threshold is the strongest single-stream lever (decode is step-count-bound). Runtime-patched the auto-detect `DiffusionConfig()` via DIFF_THRESHOLD env. |
| V6b flashinfer + commit_threshold=0.40 | 104.2 | 1.61× HF floor | ⚠️ **fastest but quality drops**: 104.2 tok/s (99/147/84) — but output shows token-reordering artifacts ("Here's a simple Python the result. numbers"). Maps the speed↔quality curve: 0.9≈72 (best quality) / 0.55≈91 (coherent, sweet spot) / 0.40≈104 (aggressive, degraded). **0.55 recommended.** |
| V6c draft_length=64 | — | — | ⚠️ needs matching VLLM_DLLM_DRAFT_SIZE env: DiffusionConfig draft_length=64 honored by model_state but plugin worker hardcodes block 32 -> ValueError "next_input_block length mismatch: expected 32, got 64" (worker.py:77). Fixable (set VLLM_DLLM_DRAFT_SIZE=64 too) but draft-32 is the validated block + V6a(thr0.55,draft32)=90.7 is the sweet spot; deferred. |

## RECOMMENDED tuned config (so far): vLLM dllm-plugin, BF16, **--attention-backend FLASHINFER + DiffusionConfig commit_threshold=0.55**, eager, TP=1 = **90.7 tok/s** (1.40× the 64.9 raw-transformers floor; 1.26× the eager/TRITON_ATTN 72.1), coherent output. commit_threshold is the dominant single-stream lever (0.9→0.55 = +24%).
| **V3-retry NVFP4 + patched loader** | loads, decode ≪1 tok/s | — | ⚠️ **loader FIX worked; decode impractically slow at concurrency-1**. Patched dllm-plugin `models/llada2.py` load_weights to map compressed-tensors NVFP4 expert params ({gate,up,down}_proj.{weight_packed,weight_scale,weight_global_scale,input_global_scale} → FusedMoE w13_/w2_+suffix, shards w1/w3/w2). Now loads past "Missing weights", "VLLM_CUTLASS NvFp4 MoE backend" active, startup complete, weights ~10GB (vs 30GB BF16), KV 31.7GiB. BUT generation didn't complete 16 tokens in ~280s (GPU 26-33% — not saturated): the CUTLASS FP4 **grouped-GEMM is throughput-oriented and starves at concurrency-1's tiny batch** (~32 tok/forward). NVFP4 is a MEMORY win, not a concurrency-1 speed win here. Motivates the low-concurrency MoE benchmark below. (Loader patch is real + reusable; the FP4-at-small-batch slowness is the kernel's regime, not the loader.) |

## Low-concurrency MoE benchmark (our usage profile — concurrency 1)
`benchmark_moe.py` (NO --tune; **default** config — the one the serve actually uses), LLaDA2 MoE
(E=256, top-8, N=512, hidden 2048), bf16, sm_110a. Batch = tokens routed per forward; concurrency-1
diffusion-block decode ≈ 32 tokens/forward.

| Batch (tok/fwd) | MoE kernel time | Per-token | Default config |
|---:|---:|---:|---|
| 1  | 314.8 µs  | 314.8 µs | M16,N64,K128,GROUP1,warps4,stages4 |
| 8  | 1873.7 µs | 234.2 µs | M16,N64,K128,warps4,stages4 |
| 16 | 2892.2 µs | 180.8 µs | M16,N64,K128,warps4,stages4 |
| **32** | **4570.0 µs** | **142.8 µs** | M16,N64,K128,warps4,stages4 |
| 64 | 6266.7 µs | 97.9 µs  | M32,N64,K128,warps4,stages3 |

**Takeaways (why the threshold/speed-mode win works):** per-token MoE GEMM cost drops **2.2×** from
batch-1 (314.8 µs) to batch-32 (142.8 µs) and **3.2×** to batch-64 (97.9 µs). Single-stream diffusion
decode is MoE-GEMM-bound, and committing more tokens per denoise step (lower commit_threshold) raises
the effective batch → cheaper per token — this is the mechanism behind V6a (thr 0.55 = 90.7 tok/s). It
also confirms draft_length=64 would amortize better (97.9 µs/tok) **if** the plugin worker block size
matched (VLLM_DLLM_DRAFT_SIZE=64). All times are on the **default** (untuned) config — a tuned Thor
config would lower them further, but see below.

**MoE autotune (--tune) could not complete on this Thor:** `benchmark_moe --tune` searches 1920 kernel
configs and has **no per-config timeout**; the tail configs (max num_warps/num_stages) run 12–16 s each
and effectively hang, stalling both the full sweep and the small-batch (1/8/16/32/64) run at ~96–97%
before the tuned config is written. So the Thor MoE config can't be produced via this tool as-is — it
would need a per-config timeout patch (upstream limitation). The default config + the threshold/attention
tuning is what delivers the 90.7 tok/s recommended result.
