# LLaDA2.1-mini on Thor — Viability Research & Roadmap
Date: 2026-06-09 · 3 threads: (1) speculation that actually speeds diffusion decode,
(2) NVFP4 with fused kernels, (3) vLLM diffusion serving (use dllm-fork-coherent vs hand-roll).
Grounded in repo reads (S2D2, dInfer, dllm-plugin, our vLLM fork `vllm-src`) + 2026 web sources.

## The unifying finding (changes the Stage-5 conclusion)
All three threads converge on **one critical path**: stand up **vLLM block-diffusion serving via
the `dllm-plugin`**, and the other two unlock for free. Two facts I verified directly that
overturn the earlier "blocked / mock" read:
1. **The dllm-plugin's real `LLaDA2ForCausalLM` is registered BY DEFAULT** (36 KB, 892 LOC, real
   256-expert FusedMoE + block-diffusion `ModelState` + non-causal attention backend + runtime
   scheduler/worker). The mock is **opt-in** via `VLLM_DLLM_USE_MOCK_MODEL=1`
   (`dllm_plugin/__init__.py:108-122`, `config.py:124`). The README's "production in progress"
   is stale wording — Phase 7 is shipped.
2. **Our `0.20.0.dev0+dflash` fork already has ~80% of the diffusion infra** the plugin needs:
   MRV2 `ModelState` (`v1/worker/gpu/model_states/interface.py`), `draft_tokens` GPU buffer
   (`v1/worker/gpu/states.py`), and `use_non_causal` plumbing in `config/attention.py`,
   `v1/attention/{selector,backend}.py`, and even `v1/spec_decode/dflash.py` (DFlash's drafter is
   already non-causal). Verified present on this box.

So vLLM diffusion serving is **~250–400 LOC of porting**, not a from-scratch build — and once it
runs, the plugin threads vLLM's `quant_config` into `FusedMoE`, which selects
`CompressedTensorsW4A4Nvfp4MoEMethod` → the **CUTLASS FP4 grouped GEMM already proven on this
Thor** (our 122B NVFP4 run). That single path makes NVFP4 fused AND gives a serving surface where
spec-decode infra coexists.

---

## Thread 1 — Make speculation actually speed up diffusion decode
**Why S2D2 lost here (code+device):** with the eval default `min_ssd_span_length=1`, S2D2 fires a
~full-cost **2L-length verify forward almost every denoising step** (`S2D2/LLaDA2/generate_utils.py`
draft `nfe+=1` ~858-865, verify `nfe+=1` ~970-978, gate never trips ~915-924). On bandwidth-bound
Thor a 64-tok verify forward ≈ a 32-tok draft forward, so tok/NFE fell 8.1→~5.5 and tok/s tracks
tok/NFE → −30%. The paper's 4.4× is vs the **static no-cache** baseline, not vs KV-cached decode.

**Ranked, concrete next steps:**
1. **Gate verification hard (highest leverage, training-free).** Run S2D2 with
   `--do_verify_policy score_threshold --do_verify_score_type difference_static
   --token_acceptance_estimator soft_entropy_negexp --score_penalty_coef 1.0
   --do_verify_score_threshold <tune 2–6>` so verify fires only when expected accepted span
   E[K]≳4–6 (break-even when a verify forward costs ~1 draft forward). Expected: recover toward
   8.1 tok/NFE + upside on confident spans. Realistic ceiling: **break-even to +15%**, not multiples.
2. **Longer gen_length (512, 1024).** S2D2/dLLM wins appear at long gens (more low-entropy spans →
   higher E[K] per verify). Our 256-tok probe was in the worst regime.
3. **Raise the BASELINE first (pure win, no spec):** fold **Fast-dLLM v1 dynamic confidence
   threshold** (arXiv 2505.22618) into `generate_cached` (commit all tokens above an adaptive
   confidence vs fixed 0.7), and **remove the redundant per-block KV-commit forward**
   (`generate_utils.py:255-265, 1081-1090`) by reusing the last denoise forward's KV with
   `store_kv=True` — ~5–10% free.
4. **Best higher-ceiling bet = d3LLM** (hao-ai-lab, ICML 2026, arXiv 2601.07568): pseudo-trajectory
   distillation → **~9.91 tokens/forward at batch 1** (vs our cached ~8.1) — it directly maximizes
   the metric that dominates a forward-bound device. Cost: a distillation pass; released models are
   dense LLaDA/Dream 7-8B, so it'd need running their recipe on LLaDA2.1-mini MoE. This is the path
   most likely to beat 64.9 tok/s by a real margin.
**Honest verdict:** training-free self-speculation will at best match KV-cache here; the durable win
is tokens-per-forward via distillation (d3LLM) or fused kernels (Thread 2) — not an extra verify pass.
Not recommended: SimSD (needs separate draft, vs TP=2 baseline), DART (AR target, wrong direction).

## Thread 2 — NVFP4 with fused kernels
**State:** the fused NVFP4 grouped-MoE GEMM (CUTLASS `cutlass_scaled_fp4_mm`/`cutlass_moe_fp4`,
flashinfer trtllm-gen fp4) exists and **works on this Thor** (our 122B run), but only inside
vLLM/SGLang — NOT in HF Transformers (hence our CPU-bound dequant) and **NOT in dInfer's own code**.
Key correction: **dInfer's LLaDA2 quant path is FP8 (ModelOpt), not NVFP4** — it builds SGLang
`FusedMoE` with `ModelOptFp8Config` and a per-tensor FP8 scale layout
(`dInfer/.../modeling_llada2_moe_sglang.py:113-117,492-503`; `benchmarks/benchmark_dataset_sglang.py:14,78-89,116-118`).
Our compressed-tensors `nvfp4-pack-quantized` artifact will **not** load there unmodified.
SM caveat: CUDA 13 renamed Thor SM101→**SM110**; CUTLASS FP4 grouped-GEMM schedules are SM100-family
hardcoded — Thor is treated as SM100-family and works for the MoE path (122B proof), but it's the
fragile kernel (cf. CUTLASS #3096, vLLM #31085/#43906).

**Ranked routes:**
1. **(Primary) vLLM FusedMoE NVFP4 via dllm-plugin** — once Thread 3 lands, the plugin already
   threads `quant_config` into real `FusedMoE` (`dllm_plugin/models/llada2.py:33,175-189,360`) →
   `CompressedTensorsW4A4Nvfp4MoEMethod` → CUTLASS FP4 grouped GEMM. **Keeps our 9.5 GB artifact,
   reuses the proven kernel.** FP4 cost here ≈ free; gated only on Thread 3. (Quantize the shared
   expert too — it's manual SwiGLU, outside FusedMoE.)
2. **Patch dInfer's SGLang loader for compressed-tensors NVFP4** (~3–6 days): swap
   `ModelOptFp8Config`→compressed-tensors NVFP4, rewrite `_update_state_dict_for_fusemoe_quant`
   (`...sglang.py:1383`) to assemble packed-uint8 + group-16 block scales + global scale. Keeps the
   artifact; rides CUTLASS FP4. Risk in scale layout + Thor arch dispatch.
3. **Re-quantize to FP8 + dInfer/SGLang** (~0.5–1 day, highest success, **abandons NVFP4**): produces
   a ModelOpt FP8 checkpoint and uses dInfer's existing `--use_quant` path. Fastest route to *any*
   fused low-precision decode if NVFP4 isn't a hard requirement.
4. Hand-write an FP4 grouped GEMM into HF/S2D2 modeling — weeks, re-solves solved problems. No.

## Thread 3 — vLLM diffusion serving: dllm-fork-coherent vs hand-roll
**Recommendation A2 (do this): surgically rebase the `dllm-fork-coherent` diffusion delta onto our
`0.20.0.dev0+dflash` fork.** Because ~80% is already present, the net delta is ~**250–400 LOC**:
- 4 `ModelState` hooks (`before_step`, `custom_sampler`, `take_draft_token_ids`, `num_bonus_tokens`)
  in `v1/worker/gpu/model_states/interface.py` (default no-ops; `LLaDA2ModelState` already implements).
- `init_model_state` dispatch to `model.get_model_state_cls()` (~10 LOC).
- **The real work (~150–300 LOC, HIGH risk):** wire the V2 runner sampling path to call
  `before_step()` / `custom_sampler()` (the T-step denoise loop) / `take_draft_token_ids()` without
  breaking the shared DFlash spec-decode path in `v1/worker/gpu/model_runner.py`.
- `build_attn_metadata`: pass `causal` + add a `dllm_prefix_lengths` field to
  `CommonAttentionMetadata` (multi-block prefix attention is silently wrong without it).
Extract via `git diff dllm-fork-coherent..v0.20.2` and replay; plugin Python is reused **unchanged**
via `--scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler --worker-cls ...DllmRuntimeWorker`,
`VLLM_PLUGINS=dllm VLLM_USE_V2_MODEL_RUNNER=1`. Keeps DFlash spec-decode + our working SM110a build.
Run TP=1 (TP=2 hurts mini: −24% tok/s), FlashInfer non-causal, CUDAGraph UNIFORM_BATCH.
- **A1 (run their fork directly via PYTHONPATH):** zero wiring but a fresh 16k-commit v0.20.2 fork
  must be rebuilt for sm_110a from scratch (FA3 crashes on Thor → FlashInfer non-causal, SM110
  support unverified) and you lose DFlash. Fallback only.
- **C (dInfer + SGLang standalone):** the model authors' first-class path (merged SGLang PR,
  `--dllm-algorithm JointThreshold`, native 2.1 Speed/Quality **+ editing**). Avoids the vLLM fork
  but needs an SGLang SM110a build and gives a queue/lm-eval API, not OpenAI HTTP. Best if both vLLM
  paths stall, and the only route with true **2.1 editing** (the plugin implements 2.0-style
  remasking only — 2.1 weights would run with 2.0 decode semantics).

## Recommended critical path (single sequence)
1. **Thread 3 / A2:** port the ~250–400 LOC diffusion delta onto our fork; serve LLaDA2.1-mini
   (BF16) via dllm-plugin on Thor, TP=1, FlashInfer non-causal. Validate coherence vs the S2D2
   raw-transformers outputs. ← unlocks everything.
2. **Thread 2 / Route 1:** flip the same server to our existing 9.5 GB NVFP4 artifact (plugin →
   FusedMoE → CUTLASS FP4). Measure tok/s vs BF16 — this is where NVFP4 should finally pay off.
3. **Thread 1:** with a real server + fused kernels lowering per-forward cost, re-test gated S2D2
   (step 1 config) at gen 512; in parallel scope d3LLM distillation on LLaDA2.1-mini for the
   tokens-per-forward ceiling.
4. Fallback if Thor kernels block the vLLM fork: **dInfer+SGLang** (Thread 3-C) with FP8 (Thread
   2-Route-3), accepting native 2.1 editing in exchange for the FP4 artifact.

## Sources
S2D2 arXiv 2603.25702 · Fast-dLLM 2505.22618 · d3LLM 2601.07568 (github hao-ai-lab/d3LLM) ·
DFlare 2606.02091 · dllm-plugin github.com/vllm-project/dllm-plugin · dllm-fork-coherent
github.com/AlonKellner-RedHat/vllm · vLLM#36155, #31085, #43906 · CUTLASS#3096 ·
compressed-tensors W4A4Nvfp4MoEMethod (vLLM docs) · LLaDA2.X github / HF model card.
