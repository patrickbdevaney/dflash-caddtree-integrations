# LLaDA2.1-mini — Raw Transformers Benchmark (Stages 3 + 4)
Hardware: Jetson AGX Thor SM110a / CUDA 13 / aarch64
Image: `vllm-dflash-thor:latest` (torch 2.10.0, transformers 4.57.3)
Model: LLaDA2.1-mini, **BF16** (16B MoE, ~1.4B active, 20 layers, 256 experts ×8)
Harness: S2D2 `LLaDA2/` unified path (@66ef72d), one model load, 3 coding prompts,
gen_length=256, block_length=32, temperature=0, eos_early_stop. tok/s & NFE measured
in-process (torch.cuda.synchronize around each generate). Raw JSON:
`baseline_and_stack_bf16.json`.

`nocache`=static block-diffusion baseline (full-seq recompute/step) ·
`cached`=KV-cache decode (== Fast-dLLM-style prefix+block cache) ·
`ssd_policy`=S2D2 training-free self-speculation (cache + AR verify, mask_span_length policy).

| Config | mode | threshold / editing | tok/s | tok/NFE | vs static-baseline |
|---|---|---|---:|---:|---:|
| nocache_quality | static | 0.7 / 0.5 | 38.9 | 10.38 | 1.00× |
| nocache_speed | static | 0.5 / 0.0 | 30.5 | 7.60 | 0.78× |
| **cached_quality** | KV cache | 0.7 / 0.5 | **61.2** | 8.17 | **1.57×** |
| **cached_speed** | KV cache | 0.5 / 0.0 | **64.9** | 8.08 | **1.67×** |
| ssd_quality | S2D2 | 0.7 / 0.5 | 39.7 | 5.82 | 1.02× |
| ssd_speed | S2D2 | 0.5 / 0.0 | 35.9 | 5.41 | 0.92× |
| ssd_conservative | S2D2 | 0.9 / 0.5 | 34.3 | 5.02 | 0.88× |

**RAW CEILING (BF16): 64.9 tok/s — `cached_speed` (KV-cache decode).**
Reference point: Qwen3.6-35B-A3B DFlash on the same Thor = 137 tok/s (AR spec-decode,
different model/decoding paradigm; ratio shown for scale only, not a like-for-like claim).

## Findings (honest)
1. **KV caching is the win on Thor: ~1.6× over the static baseline** (38.9→61.2 quality,
   30.5→64.9 speed). All outputs remained coherent (correct code). This is the headline.
2. **S2D2 (`ssd_policy`) did NOT improve speed at these operating points** — 34–40 tok/s,
   *slower than plain `cached`*. Cause is visible in tok/NFE: S2D2 drops to **5.0–5.8
   tok/NFE vs cached's ~8.1**, i.e. the AR-verification passes add ~40–60% more forward
   evals, and on Thor each forward eval is the bottleneck (MoE-GEMM / memory-bandwidth
   bound), so verification cost exceeds the denoising-step savings.
   - This does NOT contradict the S2D2 paper's "4.4× vs static baseline": that figure is
     **accuracy-matched on full benchmarks (GSM8K/HumanEval/MBPP) at long generations**,
     comparing against the *static `nocache`* baseline — not against an already-KV-cached
     decode, and not at 256-token coding-prompt scale. Our sweep is a coarse 3-prompt
     latency probe; it measures *raw decode speed at fixed settings*, not accuracy-matched
     throughput. Properly reproducing the 4.4× needs the eval harnesses
     (`eval_gsm8k_llada.py` / `eval_mbpp_llada.py`) with accuracy as the matched axis —
     flagged as future work, NOT claimed here.
3. **`nocache_speed` < `nocache_quality`** (30.5 < 38.9): with no KV cache and no editing,
   threshold=0.5 needed *more* denoising iterations on the two harder prompts (BST, coin
   change) — total NFE 101 vs 74. Confidence-threshold decode step count is not monotonic
   in the threshold; high per-prompt variance (85→22→25 tok/s).

## Why BF16 (not NVFP4) is the correct base for THIS stack
In raw HuggingFace Transformers there are **no fused NVFP4 MoE kernels for SM110a** — a
compressed-tensors NVFP4 model would dequantize weights on the fly each forward pass,
which is *slower*, not faster, and the forward pass is already the bottleneck (see tok/NFE).
NVFP4 only pays off under a serving engine with fused FP4 MoE GEMMs (vLLM + dllm-fork, or
dInfer) — and that serving path is blocked on this stack (Stage 5). On Thor's 128 GB
unified memory, BF16 (32.5 GB resident) fits comfortably, so quantization buys no memory
headroom we need either. NVFP4 is still attempted as an artifact in Stage 2 and measured
empirically (Stage 3b) to confirm this prediction.
