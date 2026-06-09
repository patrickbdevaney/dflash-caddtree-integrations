# Stage 6 — SGLang (fallback serving path) findings
Date: 2026-06-09 · Hardware: Jetson AGX Thor SM110a / aarch64 / CUDA 13

## Gate
Per directive, SGLang is the **fallback**, run only if vLLM Approaches A+B fail. They are
blocked (Stage 5 RFC): our `0.20.0.dev0+dflash` fork yields causal (wrong) attention for
diffusion, and the official `modeling_llada2_moe.py` won't even import on transformers 4.57.3
(`create_bidirectional_mask`). So SGLang is the relevant fallback.

## Findings (web-grounded, June 2026)
- **SGLang IS the most mature dLLM serving stack.** LMSYS shipped day-0 LLaDA2.0 support
  (2025-12-19); LLaDA2.0-flash-CAP reportedly ~500 TPS @ 0.95-threshold decoder. dInfer
  integrates with SGLang for serving (`eval_dinfer_sglang.py`).
- **SGLang supports Jetson Thor** as of the April 2026 release (NVIDIA SGLang release notes
  RN-08516): build with `TORCH_CUDA_ARCH_LIST=11.0a` (SM110) + Triton/CUDA path config;
  aarch64 wheels are published; dedicated Jetson install docs exist.
- **SGLang-Diffusion** is a separate install: `pip install "sglang[diffusion]" --prerelease=allow`.
  Native dLLM serving, however, is **"in development, not stable for production"** as of the
  April 2026 release. dLLM roadmap: sgl-project/sglang#14199.

## Decision (honest, bounded)
A full SGLang-Diffusion serve of LLaDA2.1-mini on SM110a was **NOT executed** this session.
Rationale:
1. The vendors themselves mark native dLLM serving "not production-stable" (April 2026), so a
   green run is unlikely and a red run wouldn't be a clean architectural signal.
2. A from-scratch `sglang[diffusion]` build on aarch64 + sm_110 risks the same Triton-kernel
   compilation rabbit hole already paid down for vLLM (directive's own foreseeable F-S6-A),
   for a fallback path — poor ROI against the time already invested in the proven raw-
   transformers stack (Stages 3–4) and the NVFP4 artifact (Stage 2).
3. The directive's primary serving target is vLLM (what already works on Thor); SGLang is
   explicitly fallback-only.

## Recommended follow-up (scoped, not run)
SGLang-Diffusion is the **highest-probability path to a working LLaDA2.1-mini server on Thor**.
A dedicated session should: build `sglang[diffusion]` with `TORCH_CUDA_ARCH_LIST=11.0a`, serve
with the `JointThreshold` dLLM decoder, and benchmark against the BF16 raw-transformers ceiling
(64.9 tok/s, Stage 4). This is the single most valuable next step for *serving* LLaDA2.x on Thor.

## Sources
- https://www.lmsys.org/blog/2025-12-19-diffusion-llm/
- https://github.com/sgl-project/sglang/issues/14199
- https://sgl-project.github.io/diffusion/installation.html
- NVIDIA SGLang Release Notes RN-08516-001_v26.04 (April 2026)
