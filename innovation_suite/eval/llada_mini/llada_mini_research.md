# LLaDA2.1-mini Maximum Speed Stack — Research Log
Jetson AGX Thor SM110a / CUDA 13 / aarch64
Started: 2026-06-09

## Environment
- User: patrickd (uid 2002, non-root) — verified
- HOME: /home/patrickd ; models dir writable; 143 GB free on /
- Host has **no torch/transformers** — the working SM110a CUDA-13 stack lives inside
  the docker image `vllm-dflash-thor:latest` (+ variants ddtree/fa-native/fastsafe).
  Therefore: downloads run host-side (pure-python hf_hub 1.18.0); quantization +
  all GPU benchmarks run **inside the docker image** via `gpu_run.sh`
  (`gpu_preflight` / `gpu_run` / `gpu_wait`). `assert_clean_gpu`/`launch_tracked`
  in the directive map to those functions.
- GPU clean at start: 0 dflash-gpu containers.

## Stage 0 — vLLM diffusion-LM support status (June 2026)
Web search findings:
- **vLLM `dllm-plugin`** exists: https://github.com/vllm-project/dllm-plugin — a
  vLLM *plugin* (general_plugins entry point `dllm`) for block-based diffusion LMs.
  Provides config/remasking contracts, a mock registered model, scheduler/worker
  adapters, stack validation, CPU CI smoke + a GPU-gated mock-stack generate test.
  CAVEAT: "Production LLaDA2.0 model logic remains in progress" — i.e. the plugin
  is the integration scaffolding; the real model runner may still be partial.
  This is the correct Stage-5 path to evaluate BEFORE hand-patching the vLLM fork.
- **SGLang has the most mature dLLM support** as of 2026: LMSYS shipped day-0 support
  for LLaDA2.0 (https://www.lmsys.org/blog/2025-12-19-diffusion-llm/); LLaDA2.0-flash-CAP
  reportedly 500 TPS with a 0.95-threshold decoder in SGLang. SGLang roadmap issue
  sgl-project/sglang#14199. NOTE: this contradicts the directive's "vLLM primary,
  SGLang fallback" ordering — recorded honestly. Directive order still followed
  (vLLM first) since vLLM is what already works on Thor for NVFP4/DFlash/APC.
- vLLM core dLLM support tracked in vllm-project/vllm#18532.

## Stage 1 — LLaDA2.1-mini architecture (from config.json, no weights)
- model_type: `llada2_moe`; architectures: `LLaDA2MoeModelLM`
  auto_map: configuration_llada2_moe.LLaDA2MoeConfig / modeling_llada2_moe.*
- num_hidden_layers: 20  (first_k_dense_replace=1 → layer 0 is dense, 1..19 MoE)
- hidden_size: 2048 ; intermediate_size: 5120 (dense layer)
- num_attention_heads: 16 ; num_key_value_heads: 4 (GQA) ; head_dim: 128
- rotary_dim: 64 ; partial_rotary_factor: 0.5 ; rope_theta: 600000 ; rope_scaling: null
- MoE: num_experts: 256 ; num_experts_per_tok: 8 ; num_shared_experts: 1 ;
  moe_intermediate_size: 512 ; n_group 8 / topk_group 4 ; sigmoid router ;
  routed_scaling_factor 2.5 ; router_dtype fp32 ; moe_router_enable_expert_bias true
- vocab_size: 157184 ; max_position_embeddings: 32768 ; pad_token_id 156892
- sliding_window 4096 but use_sliding_window=false ; dtype bfloat16 ; tie_word_embeddings false
- Standard softmax attention (NOT GDN/Mamba) → FP8 KV cache architecturally valid (Stage 7).
- Total ≈16B params, ~1.4B active/forward. Expected BF16 weights ≈ 32 GB.

### Generation API (LLaDA2.1, from model card / search — to verify against modeling_*.py)
- Diffusion decode: block_length=32, denoising `steps`, `threshold`, `editing_threshold`.
- Quality Mode: threshold=0.7, editing_threshold=0.5
- Speed Mode:   threshold=0.5, editing_threshold=0.0
- temperature=0.0, top_p=None, top_k=None ; recommended output length 16384.
- LLaDA2.1 adds **token editing** (editing_threshold) vs 2.0 → "Speeding Up Text
  Diffusion via Token Editing" (Feb 2026).

### Draft-head dims (Stage 8) derived from config
- d_expert = moe_intermediate_size = 512 ; num_active = 8 → DRAFT_FFN_DIM = 4096
- conditioning layer candidates (of 20): [2, 10, 17]

## Stage 4 — S2D2 / Fast-dLLM (read before code)
- S2D2 (arXiv 2603.25702, github phymhan/S2D2 @66ef72d) ships a full `LLaDA2/`
  integration with its OWN `modeling_llada2_moe_cache.py` + `configuration_llada2_moe.py`
  (KV-cache-enabled fork of the official modeling). It imports these locally (NOT
  trust_remote_code for the model class; tokenizer uses trust_remote_code).
- One script `example_llada.py` exposes three decode modes via `--generate_fn`:
  - `nocache`    = static block-diffusion baseline (full-seq recompute each step)
  - `cached`     = KV-cache decode (== Fast-dLLM-style prefix+block cache)
  - `ssd_policy` = S2D2 training-free self-speculation (cache + AR verify, routing policy)
  Returns stats incl. `nfe` (number of forward evals) → tok/NFE is the key efficiency metric.
- Param names confirmed: block_length, gen_length, threshold, editing_threshold,
  max_post_steps, num_to_transfer, eos_id=156892, mask_id=156895, temperature.
- README headline: "On LLaDA2.1-Mini, S2D2 ... 4.4x faster than the static baseline
  with slightly higher accuracy" (conservative setting). Complementary to KV caching.
- Fast-dLLM repo (NVlabs) current layout has `v2/` + `fast_ddrive/` + `fast_dvlm/`,
  NO standalone `LLaDA2/` dir — its LLaDA KV approach == S2D2 `cached`. So the 3 S2D2
  modes cover baseline + Fast-dLLM-KV + S2D2 in one model load. Decision: use S2D2's
  unified harness (`bench_s2d2.py`, loads model once, sweeps all modes).
- Plan: run inside vllm-dflash-thor:latest (torch 2.10 / transformers 4.57.3), model
  mounted at /models, S2D2 at /work/S2D2; gpu_run guard; gen_length 256, block 32.

## Stage 5 — vLLM diffusion serving (read before code) — KEY BLOCKER
- **dInfer** (inclusionAI/dInfer @1ffeb96, arXiv 2510.08666) is a STANDALONE dLLM
  inference framework (4 modules: model / diffusion-iteration-manager / decoder /
  KV-cache-manager), NOT a vLLM backend. Supports LLaDA2.0-mini (same LLaDA2MoeModelLM
  arch as 2.1-mini) incl. quantized versions, and integrates with SGLang for serving
  (eval_dinfer_sglang.py). This is the official "engine based on dInfer and SGLang".
- **vllm-project/dllm-plugin** (@a6cb536) is the actual vLLM integration path, BUT:
  - **Requires a dedicated vLLM fork** `AlonKellner-RedHat/vllm:dllm-fork-coherent`
    that adds non-causal attention (`use_non_causal` for FlashInfer), draft-token GPU
    buffer writes, and slot-mapping remap for first-block recompute.
  - **"The base v0.20.2 release will produce incorrect (causal) attention for diffusion
    models."** → our `0.20.0.dev0+dflash` fork CANNOT serve LLaDA2 diffusion correctly
    without that scheduler/worker/attention surgery.
  - Registers a **MOCK** model by default; production LLaDA2.0 logic is "Phase 7 / in
    progress" (docs/ROADMAP.md). Real model only with the fork + extra.
  - Runtime: `--scheduler-cls DllmRuntimeScheduler --worker-cls DllmRuntimeWorker`,
    VLLM_PLUGINS=dllm, VLLM_USE_V2_MODEL_RUNNER=1, --enforce-eager --no-async-scheduling.
  - RFC: vllm#36155 (dLLM draft-token hook alignment).
- **Honest implication for Stage 5:** Approach A (plain `vllm serve --trust-remote-code`)
  will at best run the model as a CAUSAL LM → wrong attention / garbage for diffusion.
  Approach B (replicate in our fork) = porting an entire fork branch (non-causal attn +
  draft buffers + slot remap) >> 3h time-box. Correct outcome = test A briefly, then
  RFC documenting that LLaDA2 vLLM serving needs the dllm-fork-coherent surgery; SGLang
  (Stage 6) and dInfer are the mature serving paths. This is an infrastructure finding,
  not a failure.

## Sources
- https://huggingface.co/inclusionAI/LLaDA2.1-mini
- https://github.com/inclusionAI/LLaDA2.X
- https://github.com/vllm-project/dllm-plugin
- https://www.lmsys.org/blog/2025-12-19-diffusion-llm/
- https://github.com/sgl-project/sglang/issues/14199
- https://github.com/vllm-project/vllm/issues/18532
- https://arxiv.org/pdf/2512.15745 (LLaDA2.0 100B)
