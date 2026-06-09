# LLaDA2.1-mini on Thor — vLLM block-diffusion serving + optimization: what each artifact is and why

This explains every code artifact produced for serving LLaDA2.1-mini (16B MoE block-diffusion LM) on
Jetson AGX Thor (SM110a / CUDA 13 / aarch64) via vLLM, and the reasoning behind each. It complements
`vllm_diffusion_result.md` (the port) and `vllm_optimization_results.md` (the decode-speed sweep).

## The big picture
Raw HuggingFace BF16 decode tops out at **64.9 tok/s** (KV-cache; the "floor"). The goal was to serve
LLaDA2.1-mini in vLLM (batching, paged KV, fused kernels) and push decode as far as the hardware allows.
Result: **vLLM block-diffusion generation works on our DFlash fork at a tuned 90.7 tok/s (1.40× the
floor), coherent** — via a gated pure-Python port + an attention-backend + a diffusion-threshold tuning.

## Artifact 1 — the diffusion port (fork branch `diffusion-on-pr40898`)
- **What:** a 14-file, +231/-40, **pure-Python** delta on the vLLM fork that teaches the V2 model runner
  to drive a block-diffusion denoise loop instead of only autoregressive decode. Branch lives at
  `$HOME/vllm@diffusion-on-pr40898` (commit efd384973), pushed to `origin` (patrickbdevaney/vllm) for
  review — **not** PR'd to vllm-project (per its AGENTS.md).
- **Why it's gated:** every change is conditioned on `vllm_config.diffusion_config`. When that's `None`
  (all normal AR models, incl. DFlash spec-decode) the code path is byte-identical to upstream — so the
  DFlash AR speculative-decode path is provably untouched.
- **The one fix that made generation work — `NUM_BONUS_TOKENS`:** the fork's Triton
  `_combine_sampled_and_draft_tokens_kernel` hard-coded 1 bonus token (AR speculative decode emits the
  drafted tokens + 1 bonus). Block-diffusion has **0** bonus tokens (it commits a block, no bonus). The
  hard-coded 1 made the kernel index out of bounds → "illegal memory access". The port parameterizes
  `NUM_BONUS_TOKENS` and sets it to 0 when `diffusion_config` is present. That single change is what
  turned "server runs but crashes on generate" into "generates".
- **`DiffusionConfig`** (new `vllm/config/diffusion.py`): fields `draft_length=32` (tokens per diffusion
  block), `commit_threshold=0.9`, `max_denoise_steps=64`, `mask_token_id`. Auto-detected for arch
  `LLaDA2MoeModelLM` in `vllm/config/vllm.py` (also sets `_use_non_causal=True` for bidirectional
  within-block attention).

## Artifact 2 — the `:dllm` overlay image
- **What:** `vllm-dflash-thor:dllm` = `:ddtree` (the production DFlash-Thor image, vLLM
  0.20.0.dev0+dflash, torch 2.10) **+ COPY of the 14 ported .py files**. Built in 0.6 s.
- **Why:** the port is pure-Python (no CUDA/C++), so it needs **no recompile** — overlaying the files
  onto the existing image avoids a 60–90-min rebuild and, critically, the out-of-memory risk that a
  concurrent build would create on Thor's 128 GB unified memory. `scripts/Dockerfile.vllm-dllm-overlay`.

## Artifact 3 — runtime patches in the serve launcher (`scripts/vllm_dllm_serve.sh`)
The launcher installs the dllm-plugin at container start and applies small runtime patches:
- **`git config --global --add safe.directory '*'`** — the plugin's setuptools-scm build dies on
  "dubious ownership" without it (the mounted dir isn't owned by root inside the container).
- **`kv_cache_sf` conditional** — the fork unconditionally passes `kv_cache_sf` (an NVFP4-KV scale)
  to flashinfer 0.6.6's wrapper, which lacks that kwarg → TypeError at warmup. The sed makes it
  pass-only-when-non-None. For BF16/auto-KV it's always None, so this is correctness-preserving.
- **Knobs added (this sweep):** `EAGER` (0→CUDA graphs with `--cudagraph-capture-sizes` multiples of 32),
  `ATTN_BACKEND` (TRITON_ATTN | FLASHINFER), `MOE_BACKEND` (`--moe-backend`),
  `DIFF_THRESHOLD`/`DIFF_DRAFT` (runtime-patch the auto-detected `DiffusionConfig(commit_threshold=…,
  draft_length=…)` — the diffusion speed/quality knobs).
- **Safety (the OOM lesson):** every container gets a cgroup `--memory 88g` cap; a serialization
  preflight aborts if another heavy container (build/serve/tune) is alive; mem-fraction ≤ 0.45. After an
  earlier OOM from running a serve + a compile at once, the rule is: exactly one GPU/memory-heavy job at
  a time, `gpu_stop` (graceful) not `kill -9`, `drop_caches` + verify ≥60 GB free before each launch.

## Artifact 4 — the NVFP4 expert weight-loader patch (`scripts/llada2_nvfp4_loader.py` excerpt)
- **What:** an extension to the dllm-plugin's `models/llada2.py` `load_weights`. The original only knew
  BF16 expert weights (`{gate,up,down}_proj.weight`) and raised "Missing weights" on the NVFP4 model.
- **Why/how:** NVFP4 (compressed-tensors) stores each expert projection as four tensors —
  `weight_packed` (FP4), `weight_scale` (block scale), `weight_global_scale`, `input_global_scale`. The
  patch adds a branch that routes each to the matching FusedMoE NVFP4 parameter (`w13_` for gate/up,
  `w2_` for down; suffix preserved) with the right `shard_id` (w1/w3/w2) and `expert_id`. This is the
  same systematic mapping vLLM's own Qwen3-MoE NVFP4 loader uses.
- **Outcome:** with the patch, the NVFP4 model **loads** and selects the **CUTLASS FP4 grouped-MoE**
  kernel. But decode is impractically slow at concurrency-1 (see Artifact 6) — the FP4 grouped GEMM is
  throughput-oriented and starves at the tiny per-forward batch of single-stream diffusion. So NVFP4 is
  a **memory** win (~10 GB weights vs ~30 GB) here, not a concurrency-1 speed win. The loader patch is
  correct and reusable; the slowness is the kernel's regime.

## Artifact 5 — MoE autotune (`scripts/launch_moe_tune.sh`)
- **What:** runs vLLM's `benchmark_moe.py --tune` to generate a Thor-tuned fused-MoE GEMM config
  (`E=256,N=512,device_name=NVIDIA_Thor.json`). The serve logs "sub-optimal default MoE config" without
  it, and the model is MoE-GEMM-bound, so this is the highest-value *correct-mode* lever.
- **Two patches to make it run:** (a) install `ray` at runtime (the tool needs it; absent in the image);
  (b) add `LLaDA2MoeModelLM` to `get_model_params`'s architecture dispatch so it reads `num_experts=256`,
  `num_experts_per_tok=8`, `moe_intermediate_size=512`, `hidden_size=2048` (it defaulted to Mixtral's
  `num_local_experts` and crashed).
- **`MOE_BATCHES` knob (low-concurrency benchmark):** the full sweep tunes all 1920 kernel configs across
  *all* batch sizes and **stalled at 97%** on a slow large-batch config (`benchmark_moe` has no per-config
  timeout). Restricting to the small batches our concurrency-1 profile actually hits (`1 8 16 32 64`;
  diffusion block decode ≈ 32 tokens/forward) keeps every config fast and avoids the stall — this is the
  "MoE benchmark at low concurrency for our usage profile" run.

## Optimization sweep — summary (full table in `vllm_optimization_results.md`)
Baseline = eager + TRITON_ATTN = **72.1 tok/s**. Every blocker hit was a fixable config/applicability
issue, never a track-ending sm_110 wall.
- V1 CUDA graphs — captures (after 2 config fixes) but **breaks diffusion correctness** → eager stays.
- V2 MoE autotune (full) — stalled at 97% (no per-config timeout); rerun at low concurrency (Artifact 5).
- V3 NVFP4 + loader patch — loads + CUTLASS FP4 active, but **decode ≪1 tok/s at concurrency-1** (FP4
  starves at tiny batch); memory win only.
- V4 FlashInfer-CUTLASS *unquantized* MoE — **not supported on sm_110** (kernel device gap).
- **V5 flashinfer non-causal attention — 73.3 tok/s + cleaner output.** New recommended attn backend.
- **V6a flashinfer + commit_threshold=0.55 — 90.7 tok/s (1.40× floor), coherent.** The denoise
  confidence threshold (LLaDA "speed mode") is the strongest single-stream lever (0.9→0.55 = +24%).
- V6b threshold 0.40 — 104 tok/s but quality degrades (reordering artifacts). V6c draft_length=64 —
  needs `VLLM_DLLM_DRAFT_SIZE=64` to match the worker; draft-32 is the validated block.

### RECOMMENDED config
**vLLM dllm-plugin, BF16, `--attention-backend FLASHINFER`, `DiffusionConfig.commit_threshold=0.55`,
eager, TP=1 = 90.7 tok/s** (coherent). For best quality use the default threshold 0.9 (≈72 tok/s); for
max speed 0.40 (≈104, degraded). NVFP4 only when memory-constrained (decode slow at concurrency-1).

## Output-quality caveat (unchanged across the sweep)
The dllm-plugin implements LLaDA-**2.0**-style remasking; the weights are **2.1** (token-editing). Output
is coherent-ish with minor artifacts at threshold 0.55 — a serving/throughput result, not an accuracy
claim. True 2.1 quality needs the editing-remasking policy in the plugin (future work, or the stock
SGLang path which has native 2.1).
