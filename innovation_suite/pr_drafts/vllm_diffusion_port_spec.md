# vLLM Diffusion Generation Port Spec — get LLaDA2.1-mini generating on our DFlash fork
Date: 2026-06-09 · Status: server RUNS (vllm_diffusion_result.md); this spec = the exact fix to make
it GENERATE. Source: `AlonKellner-RedHat/vllm:dllm-fork-coherent` (HEAD `4019579`).

## Root cause (pinpointed via CUDA_LAUNCH_BLOCKING=1)
Generation crashes in the fork's AR-spec-decode Triton kernel `_combine_sampled_and_draft_tokens_kernel`
(`v1/worker/gpu/input_batch.py:280`, called from `model_runner.py prepare_inputs:832`): it hardcodes
`num_draft_tokens = num_logits - 1` and **unconditionally** writes a "bonus" last-sampled token at
`query_end - num_logits`. The dllm-plugin's `DllmRuntimeScheduler` carries diffusion **block-drafts**
(block=32, **no bonus token**) in `scheduled_spec_decode_tokens`; with no bonus token the `-1` and the
unconditional write index **out of bounds** → illegal memory access. Even if crash-fixed, AR sampling
on a diffusion model is wrong — correct output needs the diffusion **denoise loop** (`custom_sampler`).

## The fix is GATED on `diffusion_config` → DFlash AR path is preserved by construction
fork-coherent computes `self._num_bonus_tokens = 0 if vllm_config.diffusion_config is not None else 1`
(`model_runner.py:182`). When `diffusion_config is None` (DFlash/AR), `num_bonus_tokens=1`, the kernel
behaves exactly as today, and all new ModelState hooks return their AR defaults. **So merging the delta
does not change the DFlash AR spec-decode path.** This is the key safety property.

## Exact delta to merge (6 files, ~428 diff-lines; from dllm-fork-coherent vs its base)
1. **`v1/worker/gpu/input_batch.py`** (~22 lines): add `NUM_BONUS_TOKENS: tl.constexpr = 1` to
   `_combine_sampled_and_draft_tokens_kernel`; `num_draft_tokens = num_logits - NUM_BONUS_TOKENS`;
   guard the bonus write `if NUM_BONUS_TOKENS > 0`. Add `num_bonus_tokens: int = 1` to
   `combine_sampled_and_draft_tokens(...)` and pass `NUM_BONUS_TOKENS=num_bonus_tokens` to the kernel.
   (Default 1 ⇒ AR unchanged.)
2. **`v1/worker/gpu/model_states/interface.py`** (~62 lines): add 4 ModelState methods —
   `custom_sampler(sampler, config)` (wrap/replace sampler → **the block-diffusion denoise loop**),
   `before_step(scheduler_output, dummy_run)` (extract per-step diffusion metadata),
   `take_draft_token_ids()` (produce next-step block drafts), and `@property num_bonus_tokens` (→1 AR, 0 diffusion).
3. **`v1/worker/gpu/model_runner.py`** (~134 lines): add `diffusion_config` plumbing;
   `self._num_bonus_tokens = 0 if diffusion_config is not None else 1`; thread `bonus` into
   `total_num_logits`/`num_logits`/`combine_sampled_and_draft_tokens(..., num_bonus_tokens=bonus)`
   (lines ~731,751,753,824); call `model_state.before_step()` / `custom_sampler()` / `take_draft_token_ids()`.
4. **`v1/worker/gpu/attn_utils.py`** (~204 lines): `build_attn_metadata(..., causal=, dllm_prefix_lengths=)`
   passthrough for non-causal + multi-block prefix concat.
5. **`v1/attention/backend.py`** (~5 lines): add `causal: bool=True` (present) + `dllm_prefix_lengths`
   field to `CommonAttentionMetadata`.
6. **`v1/worker/gpu/model_states/default.py`** (1 line) + add `diffusion_config` to `VllmConfig`/config.
Also: `model_states/__init__.py` `init_model_state` must dispatch to `model.get_model_state_cls()` so the
dllm-plugin's `LLaDA2ModelState` (which supplies these hooks) is used.

## Already-applied prerequisite fixes (in vllm_diffusion_result.md, runtime-patched today)
- flashinfer `kv_cache_sf` made conditional (flashinfer 0.6.6 lacks the kwarg). Fold into the branch.
- dllm-plugin install: `git config --global --add safe.directory '*'` + `pip install` (setuptools-scm).

## How to apply (clean, reviewable, DFlash-safe) — recommended
Our fork source is a git repo at `$HOME/vllm` (branch gdn-apc-gate-fix). Plan:
1. `git -C $HOME/vllm checkout -b feat/llada2-diffusion-runner <main-base>` (branch off main, NOT the
   dflash-thor branch, per project rule; do not auto-PR).
2. Add `AlonKellner-RedHat/vllm` as a remote, fetch `dllm-fork-coherent`, find the merge-base, and
   `git cherry-pick`/3-way-merge the diffusion commits for the 6 files above (they're gated on
   diffusion_config, so conflicts with DFlash should be minimal/additive). Resolve any DFlash overlaps
   keeping AR behavior at num_bonus_tokens=1.
3. Rebuild the Thor image from this branch (~90 min source build, as the original DFlash image build was
   ~92 min) — run ALONE, memory-capped, serialized (the OOM lesson).
4. **Verify BOTH paths**: (a) DFlash AR spec-decode serve still works bitwise (regression gate);
   (b) LLaDA2.1-mini diffusion serve now generates coherent text via the dllm-plugin. Then benchmark.

## Honest status / recommendation
The vLLM diffusion **server runs end-to-end on our fork today**; the remaining work to make it
**generate** is this well-specified, gated, ~428-line merge + a ~90-min source rebuild + dual-path
verification. It is the correct, lowest-risk way (gated design preserves DFlash). It is NOT a 5-minute
patch and warrants review (per the standing rule to review autonomous fork contributions), so it is
written up here ready to implement rather than applied blind. No tok/s is claimed until generation works.
