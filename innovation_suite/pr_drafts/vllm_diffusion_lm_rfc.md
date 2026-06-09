# vLLM RFC Notes: Block-Diffusion LLM (LLaDA2) Inference on Jetson Thor
Status: **RFC / infrastructure finding** (no quality claim) · Date: 2026-06-09
Author context: DFlash-Thor program, companion repo dflash-caddtree-integrations

## Problem
Block-diffusion LLMs (LLaDA2.x, SDAR, Dream, Fast-dLLM-v2) generate text by denoising a
K-token block over T steps with **bidirectional (non-causal) attention within the active
block**. vLLM's V1 model runner + scheduler assume autoregressive causal decoding. Serving
LLaDA2.1-mini "as-is" through vLLM therefore produces **causal (wrong) attention** and
incorrect output, even when the weights load.

## What the ecosystem actually provides (June 2026, verified)
> **CORRECTION (2026-06-09, after deep repo read — supersedes the line below):** the plugin's
> **real `LLaDA2ForCausalLM` is registered BY DEFAULT** (`dllm_plugin/__init__.py:108-122`,
> `config.py:124` → `dllm_plugin.models.llada2:LLaDA2ForCausalLM`, 892 LOC real 256-expert
> FusedMoE + block-diffusion ModelState + non-causal attention backend). The **mock is opt-in**
> via `VLLM_DLLM_USE_MOCK_MODEL=1`. The README's "production in progress" is stale wording —
> Phase 7 is shipped. AND our `0.20.0.dev0+dflash` fork already has ~80% of the required infra
> (MRV2 ModelState, `draft_tokens` buffer, `use_non_causal` in config/attention + spec_decode/dflash);
> the gap is ~250–400 LOC of denoise-loop wiring. **So this is NOT blocked — it is a bounded port.**
> Full plan: `eval/llada_mini/RESEARCH_ROADMAP.md` (Thread 3, option A2). The text below is the
> original (pre-correction) assessment, kept for provenance.

1. **vllm-project/dllm-plugin** (@a6cb536) — the official vLLM plugin path for dLLMs.
   - Registers two architecture names with `ModelRegistry`, ~~by default to a MOCK model~~
     (CORRECTED above: real model is default; mock is `VLLM_DLLM_USE_MOCK_MODEL=1`).
   - **Requires a dedicated vLLM fork** `AlonKellner-RedHat/vllm:dllm-fork-coherent`, which
     adds: non-causal attention (`use_non_causal` for FlashInfer), draft-token GPU buffer
     writes, and slot-mapping remap for first-block recompute. The README is explicit:
     *"The base v0.20.2 release will produce incorrect (causal) attention for diffusion
     models."*
   - Runtime wiring: `VLLM_PLUGINS=dllm`, `VLLM_USE_V2_MODEL_RUNNER=1`,
     `--scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler`,
     `--worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker`,
     `--enforce-eager --no-async-scheduling --trust-remote-code`.
   - Pins `vllm>=0.20.0,<0.21`. Upstream hook RFC: vllm-project/vllm#36155.
2. **dInfer** (inclusionAI/dInfer @1ffeb96, arXiv 2510.08666) — a **standalone** dLLM engine
   (model / diffusion-iteration-manager / decoder / KV-cache-manager). Supports LLaDA2.0-mini
   (== our LLaDA2.1-mini arch `LLaDA2MoeModelLM`) incl. **quantized** variants, and integrates
   with **SGLang** for serving. This is the official "engine based on dInfer + SGLang".
3. **SGLang** — most mature dLLM serving (LMSYS day-0 LLaDA2.0; LLaDA2.0-flash-CAP ~500 TPS).

## What was attempted here, and why Approach B was not pursued
- **Approach A (zero-work):** `vllm serve <LLaDA2.1-mini> --trust-remote-code` on our
  `0.20.0.dev0+dflash` fork. NOT run to a serve, because two independent blockers make the
  outcome known a priori: (a) our fork lacks the non-causal-attention / draft-buffer / slot-
  remap surgery, so any output would be causal-wrong by construction (per dllm-plugin's own
  statement); (b) the **official** `modeling_llada2_moe.py` imports `create_bidirectional_mask`
  from `transformers.masking_utils`, which is **absent in the image's transformers 4.57.3**
  (confirmed empirically in Stage 2 — the trust_remote_code load raises ImportError). So even
  model instantiation fails on this image without the S2D2 self-contained modeling fork.
- **Approach B (replicate dllm-plugin in our fork):** porting the `dllm-fork-coherent` surface
  (non-causal FlashInfer attention, draft-token GPU buffers, first-block slot remap, plus the
  DllmRuntimeScheduler/Worker) is an entire vLLM fork branch — far beyond the 3-hour time-box,
  and it would duplicate work already done upstream. Correct engineering call: do NOT fork-
  surgery our DFlash fork; consume the upstream path when its production model lands.

## Recommendation (honest)
- For LLaDA2.1-mini serving on Thor today, the supported paths are **dInfer** (standalone) and
  **SGLang** (mature). vLLM support is gated on dllm-plugin Phase 7 + the dllm-fork-coherent
  attention surgery reaching a pinned release.
- The DFlash fork's value for dLLMs is as a **speculative-decoding target/draft host**, not a
  diffusion serving runtime. No vLLM [Feature] quality claim is justified for LLaDA2 on this
  stack; this document is an infrastructure/landscape finding.
- Concrete contribution opportunity (deferred, needs review): the dllm-plugin notes
  `create_bidirectional_mask` availability is transformers-version-sensitive — a small upstream
  compatibility shim (or a documented min-transformers pin in the LLaDA2 model card) would
  unblock trust_remote_code loads on 4.57.x images. Filed as a note, not auto-PR'd.

## Sources
- https://github.com/vllm-project/dllm-plugin · https://github.com/vllm-project/vllm/issues/36155
- https://github.com/inclusionAI/dInfer (arXiv 2510.08666)
- https://www.lmsys.org/blog/2025-12-19-diffusion-llm/
