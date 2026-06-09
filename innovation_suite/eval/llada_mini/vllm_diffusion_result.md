# vLLM Block-Diffusion Serving of LLaDA2.1-mini on Thor — RUNS end-to-end; one precise kernel blocker
Date: 2026-06-09 · Hardware: Jetson AGX Thor SM110a / CUDA 13 / aarch64
Fork: vLLM `0.20.0.dev0+dflash` (`:ddtree`) · Plugin: vllm-project/dllm-plugin @a6cb536

## Headline (overturns the earlier Stage-5 "blocked / RFC-only" conclusion)
**The vLLM block-diffusion serving stack initializes and serves end-to-end on our DFlash fork.**
With three concrete fixes, `vllm serve` for LLaDA2.1-mini reaches **"Application startup complete"**,
loads weights, allocates the KV cache (36.5 GiB / 950,478 tokens / 464× concurrency @2048), registers
the OpenAI routes, and answers `/v1/models`. The dllm-plugin's **real** `LLaDA2ForCausalLM` (not the
mock), `DllmRuntimeScheduler`, `DllmRuntimeWorker`, and the V2 Model Runner all load on our fork.
The **only** remaining blocker is a single sm_110a Triton-kernel illegal-memory-access at token
generation — precisely localized below.

## Reproducible recipe (what made it work on our fork)
Image `:ddtree` (vllm 0.20.0.dev0+dflash, torch 2.10.0). At container runtime:
1. `git config --global --add safe.directory '*'` then `pip install /work/dllm-plugin`
   (setuptools-scm needs git; without the safe.directory the editable/sdist build dies on
   "dubious ownership"). Plugin `dependencies=[]` → does NOT perturb torch/vllm.
2. Env: `VLLM_PLUGINS=dllm`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
   `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` (vllm._C symbol).
3. **Patch the fork's flashinfer backend** so `kv_cache_sf` is only passed when non-None:
   `flashinfer.py` `kv_cache_sf=kv_cache_sf,` → `**({"kv_cache_sf": kv_cache_sf} if kv_cache_sf is not None else {}),`.
   The fork unconditionally passes `kv_cache_sf` (NVFP4-KV scale) to flashinfer 0.6.6's
   `BatchPrefillWithPagedKVCacheWrapper.run()`, which has no such kwarg → TypeError during the
   FlashInfer kernel-warmup. For BF16/auto-KV `kv_cache_sf` is always None, so dropping the kwarg
   is correctness-preserving. (Same `kv_cache_sf` skew that blocked FP8-KV on the 122B serve.)
4. `vllm serve /models/LLaDA2.1-mini --trust-remote-code --max-model-len 2048 --max-num-seqs 4
   --gpu-memory-utilization 0.55 --enforce-eager --no-async-scheduling
   --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler
   --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker` (TP=1; mem-util must fit the
   unified-memory free pool — drop_caches first, and don't run other big jobs concurrently).
Scripts: `scripts/vllm_dllm_serve.sh`. The plugin registers arch `LLaDA2MoeModelLM` → its own
`LLaDA2ForCausalLM`, so the official HF `modeling_llada2_moe.py` (which needs
`transformers.masking_utils.create_bidirectional_mask`, absent in 4.57.3) is **never imported** —
bypassing the blocker that defeats plain `--trust-remote-code` serving.

## The precise remaining blocker (pinpointed with CUDA_LAUNCH_BLOCKING=1)
Token generation crashes here (synchronous trace under CLB=1):
```
model_runner.py:1042 execute_model -> :832 prepare_inputs
  -> input_batch.py:356 combine_sampled_and_draft_tokens
     -> input_batch.py:280 _combine_sampled_and_draft_tokens_kernel  (Triton)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```
(Without CLB=1 the async error surfaces misleadingly at `flashinfer.py:984 build -> seq_lens_cpu`;
CLB=1 shows the true culprit is the draft-token combine kernel, not attention.)

**Root cause:** our DFlash fork's `_combine_sampled_and_draft_tokens_kernel` is the **AR
speculative-decode** draft-token combine kernel (`model_runner.py:74,184 DraftTokensHandler`,
`:744 draft_tokens = scheduler_output.scheduled_spec_decode_tokens`). The `DllmRuntimeScheduler`
**reuses the spec-decode-shaped `scheduled_spec_decode_tokens` field** to carry the diffusion
**block-draft** tokens (block=32, `num_bonus_tokens=0`). The AR kernel indexes that buffer with
AR-spec-decode assumptions (per-request bonus-token layout) that don't hold for the diffusion
block-draft layout → out-of-bounds → illegal memory access. This is exactly the runner draft-handoff
wiring identified as the HIGH-risk part of the dllm-fork-coherent delta (RESEARCH_ROADMAP Thread-3 A2).

## The fix (scoped)
Reconcile the runner's draft path with the dllm ModelState contract — the `dllm-fork-coherent`
delta: route diffusion drafts through `model_state.take_draft_token_ids()` with `num_bonus_tokens=0`
semantics instead of the AR `_combine_sampled_and_draft_tokens_kernel`, plus add the
`CommonAttentionMetadata.dllm_prefix_lengths` field for multi-block prefix attention. Estimated
~250-400 LOC (Agent-C analysis), the bulk being this runner draft-handoff wiring. Extract via
`git diff dllm-fork-coherent..v0.20.2` and replay onto our fork. NOT auto-applied (needs review;
must not break the DFlash AR spec-decode path that shares this kernel).

## Status vs the two serving paths
Both LLaDA2 serving paths on Thor now reduce to **localized low-level issues**, not architecture:
- **vLLM (this doc):** server runs end-to-end on our fork; one Triton draft-combine kernel
  (AR-spec vs diffusion-draft layout) from working. Closest path; fix is the scoped fork delta.
- **SGLang:** blocked by the aarch64 sgl-kernel↔torch ABI corridor (sgl-kernel 0.3.21 wheel built
  for torch 2.9.1; image has 2.10.0; downgrade blocked by reverse-deps + missing aarch64
  vision/audio wheels). Source-building sgl-kernel against torch 2.10.0 is the in-progress path.

No fabricated numbers: no tok/s is reported because generation does not yet complete. The milestone
is that the stack **stands up and serves** on our fork — a large advance over "blocked".
