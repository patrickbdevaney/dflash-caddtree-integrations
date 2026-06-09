# LLaDA2.1-mini Maximum-Speed Stack — Full Results
Jetson AGX Thor SM110a · CUDA 13 · aarch64 · 2026-06-09
Model: inclusionAI/LLaDA2.1-mini — 16B MoE, ~1.4B active/forward, `llada2_moe`
(20 layers, hidden 2048, 16 q / 4 kv heads, 256 experts ×8 + 1 shared, ctx 32768).
Image: `vllm-dflash-thor:latest` (torch 2.10.0, transformers 4.57.3). All GPU work via the
`gpu_run.sh` single-container guard (flock + graceful stop + Created→running watchdog).

## Decode benchmark — BF16, raw Transformers (S2D2 harness, gen 256, block 32, 3 coding prompts)
| Method | mode | thr / edit | tok/s | tok/NFE | vs static baseline |
|---|---|---|---:|---:|---:|
| nocache_quality | static | 0.7 / 0.5 | 38.9 | 10.38 | 1.00× |
| nocache_speed | static | 0.5 / 0.0 | 30.5 | 7.60 | 0.78× |
| **cached_quality** | KV cache | 0.7 / 0.5 | **61.2** | 8.17 | 1.57× |
| **cached_speed** | KV cache | 0.5 / 0.0 | **64.9** | 8.08 | **1.67×** |
| ssd_quality | S2D2 | 0.7 / 0.5 | 39.7 | 5.82 | 1.02× |
| ssd_speed | S2D2 | 0.5 / 0.0 | 35.9 | 5.41 | 0.92× |
| ssd_conservative | S2D2 | 0.9 / 0.5 | 34.3 | 5.02 | 0.88× |

**RAW CEILING (BF16): 64.9 tok/s — `cached_speed` (KV-cache decode).**
(Qwen3.6-35B-A3B DFlash on the same Thor = 137 tok/s, AR spec-decode — different paradigm;
shown for scale only, not like-for-like.)

## Quantization — NVFP4 (Stage 2): artifact SUCCESS, raw-transformers decode impractical
| | BF16 | NVFP4 |
|---|---|---|
| disk | 32.5 GB | **9.5 GB (3.4×)** |
| runtime GPU mem | 32.5 GB | **10.1 GB** |
| load time | 33 s | 320 s (CPU-bound decompress) |
| decode | 64.9 tok/s (cached_speed) | **impractical** (GPU 28% / CPU 92%; 256-tok nocache >4 min, aborted) |
NVFP4 raw-transformers decode is CPU-bound dequant (no fused FP4 MoE kernel on SM110a) → BF16
is the speed choice. NVFP4 is the right artifact for the fused-kernel serving path. Calibration
caveat: activated-expert only (cold-expert quality risk). Details: `benchmarks/nvfp4_quantization_result.md`.

## Serving (Stage 5 vLLM / Stage 6 SGLang) — BLOCKED / deferred, documented
- **vLLM:** LLaDA2 diffusion needs the `AlonKellner-RedHat/vllm:dllm-fork-coherent` branch
  (non-causal attn + draft-token buffers + slot remap); base/our `0.20.0.dev0+dflash` fork yields
  causal (wrong) attention. `dllm-plugin` ships a MOCK model (production Phase 7 in progress).
  Plus the official modeling won't import on transformers 4.57.3 (`create_bidirectional_mask`).
  → `pr_drafts/vllm_diffusion_lm_rfc.md` (infrastructure finding, no quality claim).
- **SGLang:** most mature dLLM path; supports Thor (Apr-2026 release, `TORCH_CUDA_ARCH_LIST=11.0a`)
  and SGLang-Diffusion exists, but native dLLM serving is "not production-stable." Full serve NOT
  run (poor ROI / Triton-kernel risk for a fallback). Highest-probability next step for serving.
  → `sglang/sglang_findings.md`.

## FP8 KV cache (Stage 7) — valid, gated on serving
Standard GQA attention (4 KV heads) → FP8 KV architecturally valid (~halves KV: 1.31→0.66 GB
@32k). NOT a raw-transformers concept; only realizable under a serving engine — same blocker as
Stage 5. Matters at the 100B flash scale, not 16B-mini on 128 GB unified. → `benchmarks/fp8_kv_result.md`.

## Draft head (Stage 8) — feasibility + architecture, training deferred
Prior art correction: DFlash/DFlare/DART use block-diffusion as the **drafter** for an AR
**target** (DFlare 5.46× on Qwen3-8B). A trained draft head *for* a diffusion target is the
inverted, under-explored direction; its speed case is **contraindicated** by our Stage-4 result
(adding speculative verify to LLaDA2.1-mini was slower than plain KV-cache — Thor is forward-
bound and the diffusion target already amortizes ~8 tok/forward). DFlare-style 7-layer skeleton
written (`draft_head/draft_head_architecture.py`); data-gen + training is a multi-day job,
deferred. → `draft_head/draft_head_feasibility.md`.

## Key findings (honest)
1. **KV-cache decode (`cached`) is the win on Thor: ~1.6× over static baseline → 64.9 tok/s (BF16).**
2. **S2D2 self-speculation did NOT help at these settings** (verification NFEs dominate on a
   forward-bound device); the paper's 4.4× is accuracy-matched on full evals at long gens — a
   different axis, flagged future, not claimed.
3. **NVFP4 quantizes successfully (3.4× smaller) but is impractical in raw transformers** (CPU
   dequant); it belongs to the fused-kernel serving path.
4. **Diffusion serving (vLLM/SGLang) for LLaDA2 is not yet available on this Thor stack**;
   dInfer (standalone) + SGLang-Diffusion are the routes — documented, not yet built.
