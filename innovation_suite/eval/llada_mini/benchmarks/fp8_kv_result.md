# Stage 7 — FP8 KV Cache for LLaDA2.1-mini
Date: 2026-06-09 · Hardware: Jetson AGX Thor SM110a

## Architectural validity (YES)
LLaDA2.1-mini uses **standard softmax attention with GQA** (NOT GDN/Mamba): 20 layers,
16 query heads / **4 KV heads**, head_dim 128, rotary_dim 64. So there is a real KV cache and
FP8 KV is architecturally valid (unlike GDN-hybrid recurrent layers, which have no KV to quant).

KV memory math (per token, both K and V, 4 KV heads × 128 = 512 dims/layer × 20 layers):
- BF16: 4 KV heads × 128 × 2 (K+V) × 20 layers × 2 B = **40 KB/token** → at 32k ctx ≈ **1.31 GB**.
- FP8 : half that → ≈ **0.66 GB** at 32k.
At 16B-mini scale on Thor's 128 GB unified memory this is **not** a meaningful lever; it matters
for the 100B LLaDA2.0-flash (32 layers, far larger KV) under a serving engine.

## Can it be tested here? NO — gated on the serving path
FP8 KV cache is a **serving-engine feature** (`--kv-cache-dtype fp8` in vLLM/SGLang). It is not
a raw-Transformers concept: the diffusion decode in S2D2's `cached` path uses an in-house block
KV cache in BF16, with no FP8 quantization hook. And the serving engines that *do* expose FP8 KV
(vLLM, SGLang) cannot correctly serve LLaDA2 diffusion on this stack (Stage 5: needs the
dllm-fork-coherent non-causal attention surgery; SGLang dLLM not production-stable). Therefore:
- **FP8 KV for LLaDA2.1-mini was NOT benchmarked** — there is no engine on this stack that both
  (a) serves LLaDA2 diffusion correctly and (b) exposes FP8 KV. Same root blocker as Stage 5.
- Separately, FP8 KV on SM110a has historically hit the same kernel-enablement gap seen for
  Qwen3.x on this Thor build (prior project finding) — a second, independent gate.

## Conclusion
FP8 KV is valid for this architecture and would roughly halve KV memory, but it is only
realizable once a fused-kernel diffusion serving engine is available on Thor (vLLM dllm-fork or
dInfer/SGLang) — at which point it should be measured on the 100B flash variant where KV size
actually matters. Documented as gated, not run.
