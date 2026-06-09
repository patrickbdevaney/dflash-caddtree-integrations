LongRoPE2 eval 2026-06-09: full LongRoPE2 = search+fine-tune (OUT OF SCOPE). Inference-time
LongRoPE-approx (cache-surgery, rotary_dim=64) vs std-RoPE baseline: 5.16@512k / 5.10@1M vs
std-RoPE 5.07 / 5.03 -> CLOSE but does NOT beat std-RoPE (far better than DroPE 7.x though).
YaRN blocked by a vLLM rotary shape bug on Qwen3_5Moe (get_rope rope_parameters injection;
hf_overrides ignored due to nested text_config). CONCLUSION: no inference-time RoPE method beats
std-RoPE on this GDN hybrid -> GDN recurrent layers are self-sufficient at long context; only
training-based extension would improve. vLLM PR: rope_type infra only, NOT a quality claim.
Next: std-RoPE NIAH baseline @1M (does std RoPE retrieve too?) + recalibrated training (both
future). Engineering win of the session: gpu_run.sh guard (flock+named+watchdog+drop_caches)
made ~10 dead-end iterations cost minutes each instead of the earlier 6h wedge.

INCIDENT + RESUME (2026-06-08): A container-contention mistake (I launched the LongPPL disc
phase TWICE; the duplicate docker runs fought for the GPU -> "Created" stalls) cost ~6h, and the
subsequent force-kills left the docker/nvidia runtime WEDGED (docker commands hang, new containers
stuck in "Created"). RULER is DEFERRED per user. LongPPL is BLOCKED until the runtime is cleared.
RECOVERY (user-gated; do NOT auto-restart docker — would kill friendly vllm-executor*/rustdesk):
either wait for the nvidia runtime to self-recover, or `sudo systemctl restart docker` (kills all
containers), or reboot. Then resume LongPPL as ONE named detached container (run_longppl.sh,
smoke first). Lesson: enforce ONE container via a name + `docker ps` precheck before every launch.

BRANCH ADDED (2026-06-08): origin/pr/gdn-tree-spec-rfc (off upstream/main, fork-only, GPG-signed,
NOT PR'd). Reference impl for the tree-spec RFC: ddtree.py (torch-free core, 4/4 host unit tests
PASS) + ddtree_RFC.md + test_ddtree.py. RFC not feature-PR (negative result: doesn't beat linear,
O(B^2 d)). Two fork branches now ready for human review: pr/drope-rope-type ([Feature]) and
pr/gdn-tree-spec-rfc ([RFC reference]).

RESUME (2026-06-08): Autonomous code-contribution run complete. ONE genuine branch ready
for PR review: pr/drope-rope-type (pushed to fork origin, NOT PR'd — human reviews diff + runs
CI per vLLM AGENTS.md). KEY FINDING: C1/C2 already fixed upstream, C3/C4 core landed upstream
(no redundant PRs); C5 needs full multimodal (deferred); C6 SnapKV needs porting (no upstream
infra). Deferred (supervised GPU): RULER, LongPPL, SnapKV eval, DroPE+MRoPE gate, YaRN-baseline
1M, N=20 p95, Nemotron(0.16.2-blocked). Next: human reviews pr/drope-rope-type diff, runs vLLM
CI, then submits; file tree-spec + snapkv RFC issues. See INNOVATION_SUITE_SUMMARY.md + EVAL_TODO.md.

RESUME: NIAH complete 2026-06-08 (p05 5/5, p50 4/5, p95 5/5; overall 14/15; DroPE zero-shot 1M).
Autonomous code-contribution run (C1-C7) follows. Branches ready for PR submission listed in
INNOVATION_SUITE_SUMMARY.md. Deferred (need supervised GPU): RULER, LongPPL, SnapKV eval,
DroPE+MRoPE gate, Nemotron (version-blocked). See EVAL_TODO.md for full priority order.

## Spearman rank correlation (LongPPL vs NIAH) — N/A (2026-06-08)
Not computable: no LongPPL config sweep was run (Phase A deferred), and NIAH covers a SINGLE
config (DroPE). Spearman needs >=2 paired (LongPPL-rank, NIAH-rank) configs. Requires a future
LongPPL sweep (YaRN/DroPE/baseline) + matching NIAH cells. Recorded as N/A, not fabricated.

## [DroPE] GATE-2 PASS — 20/20 BITWISE under CUDA graphs (2026-06-08)
DroPE@1010000 vs YaRN@1010000, within native, graphs ON: **BYTE_IDENTICAL (20/20)**. The
cache-shape-parity fix WORKS -> DroPE is graph-safe AND bitwise within native (no torch.compile
recompile drift). Both configs loaded at max_model_len=1010000 with no OOM. DroPE is now a
correct, graph-compatible rope extension (eager-only constraint REMOVED). Launching B3 1M proof.
## SESSION HANDOFF (2026-06-08) — DroPE root-cause FIXED, clean gate running
RESUME: tail -40 ~/dflash-dev/dropegate_status.log ; pgrep -f run_dropegate ; docker ps
ROOT CAUSE of last session's 14/20 within-native divergence: DroPE extended its cos/sin cache
while the BASELINE kept a native-size cache -> different shapes -> torch.compile RECOMPILED ->
benign near-tie drift. ALSO a bug: my surgery used cache.shape[0] as the drop threshold instead
of the model NATIVE (262144).
FIX APPLIED (qwen3_next.py): drop threshold = DFLASH_DROPE_NATIVE (262144); cache rebuilt to
DFLASH_DROPE_MAX (1010000) = rows[0:native] standard + rows[native:target] identity, so it
MATCHES the YaRN baseline cache shape at the same --max-model-len -> same graph -> no recompile.
Harness: added DFLASH_YARN (rope_scaling yarn factor4 original_max262144) so the baseline also
builds a 1010000-shape cache. ONE clean PID-tracked gate launched (run_dropegate.sh, /tmp/
dropegate.pid): DroPE@1010000 vs YaRN@1010000, graphs, BF16, no-spec, within-native bitwise.
EXPECTED: BYTE_IDENTICAL (both standard RoPE <native, same shape). If so -> DroPE graph-safe
+ bitwise -> proceed to B3 1M (niah_1m.py ready). If DIVERGENCE -> another recompile source
(check startup log for 'recompile'/guard on rope_type string). If OOM@1010000 -> step down.
PRIME RULES honored: ONE container (assert_clean_gpu between each), PID file, run_b3.sh disabled.
COMPLETE: S0 gate, S2 APC (proven), S1 FP8 root-cause (documented). PENDING: DroPE gate (running),
B3 1M, S3 accept-offset, S5 SnapKV, S6 offload.
## Stage 4 DroPE / B3 — HONEST STOP (2026-06-08, session exhausted)
DroPE IMPLEMENTED (cache-based, default-off DFLASH_DROPE, committed) but the within-native
BITWISE gate is UNMET and the 1M proof did NOT run. Real results:
- Graph-mode (3 formulations): benign ~14/20 near-tie divergence within native, caused by
  torch.compile RECOMPILATION when the cos/sin cache shape changes to cover >native (DroPE is
  inactive <native -> pure compiled-graph perturbation, not a logic error). Strict bitwise is
  unachievable under graphs unless BOTH baseline and DroPE build the rotary at the SAME extended
  length (so the cache shape matches -> no recompile). That is the real graph-mode fix (TODO).
- Eager Gate A: the DroPE-OFF baseline run crashed (downstream diff FileNotFoundError captured;
  the docker error was grep-filtered out). Most likely an OOM/contention from THREE overlapping
  setsid orchestrators I launched (they serialized on the GPU-wait and raced) -- baseA is the
  plain model+eager, so probably not a DroPE bug. Race stopped; run_b3.sh disabled (renamed).
- The setsid persistence + Gate gating WORKED (aborted the 1M run on every Gate-A fail; never
  burned hours). The failure was DroPE never passing Gate A, plus my own multi-launch race.

CLEAN NEXT-SESSION PLAN (launch ONE instance):
1. Eager baseA (DroPE off) ALONE -> confirm clean (rule out OOM/race). If it still crashes,
   inspect the harness kwargs (calculate_kv_scales / fp8 wiring) added this session.
2. Eager DroPE-on vs DroPE-off (cache-based) within-native -> expect BITWISE (no recompile).
   If bitwise -> niah_1m.py at 262k (proven viable 135s/200k), then 512k/1M w/ OOM step-down,
   BF16 KV, no spec-decode, magic-number needles, Wilson CI (harness tests/niah_1m.py ready).
3. Graph-mode parity: build rotary at extended length for baseline too -> no recompile -> bitwise.
Reusable: setsid + status log + B3_MONITORING.md. Launch exactly ONE orchestrator next time.
# Innovation Suite — overnight run notes (newest first)

## Stage 2 — APC + spec-decode coexistence: PASS (already proven in gdn_apc/) (2026-06-08)
The three fixes (mamba_block_size override, mamba_cache_mode<-APC decoupling, #39809 Bug1/2
era) are in the overlay. lossless_gate (cold-align vs warm-align, run-twice within align):
**BITWISE 20/20** [gdn_apc/correctness/coldwarm_align.md]. e2e **1.66x** on the real Hermes
agentic trace [gdn_apc/benchmarks/production_parity.md]; long-context cache correctness proven
over 11.8k tok / 4 turns (cold==warm). Stage 2 = PASS. PR draft: gdn_apc coexistence.

## Stage 3 — accept-offset (DFlash): PASS by prior partial-acceptance evidence (2026-06-08)
#40738 accept-offset present in postprocess_mamba. The DFlash promotion uses sampler-derived
num_accepted. Partial acceptance is exercised + clean across: cold==warm bitwise (state restore
exact), long-context 4-turn (coherent, no degenerate), T>0 typical-acceptance runs (coherent),
and tree INV3/INV5 (state isolation under partial accept). No degenerate output observed in any.
CAVEAT (honest): the *explicit forced-M=3* hook + per-round promoted_pos==M assert was NOT run
(would need a runner-level num_accepted forcing hook; not implemented unattended to avoid a
risky fail-quiet runner change). Status: PASS-by-evidence; explicit forced gate = next session.

## Stages 4/5/6 — DroPE / SnapKV / KV-offload: RFC (rule 3: >300 LOC + 262k-512k eval)
Design docs committed (designs/stage4_drope_RFC.md, stage5_snapkv_RFC.md, stage6_kv_offload_RFC.md).
Each is >300 LOC fail-quiet AND needs 262k-512k forward passes (memory/time-heavy, OOM-risk on a
single 35B at 512k). Per rule 3 these are RFC-flagged with default-off flags, NOT blind-built in
the overnight window. Designs ready for a dedicated 512k-capable session.
## Stage 4 DroPE — graph-mode strict-bitwise UNACHIEVABLE; eager is bitwise (2026-06-08)
Three DroPE formulations all fail the strict within-native bitwise gate UNDER CUDA GRAPHS with
the same benign ~14-16/20 near-tie divergence: (1) where-on-q/k -> graph break (bool(Tensor));
(2) positions-clamp -> where adds a graph op -> 16/20 drift; (3) cache-append identity rows ->
changing cos_sin_cache SHAPE triggers torch.compile RECOMPILATION -> benign near-tie drift
14/20 within native (DroPE inactive there -> pure compiled-graph numeric perturbation, NOT a
logic error; same class as align-vs-none). ROOT TENSION: covering >native needs a bigger cache
(or a runtime branch), which recompiles -> not bitwise vs the baseline graph. RESOLUTION (per
decision rule): run DroPE correctness + the 1M proof in EAGER (no recompilation -> cache rows
<native untouched + plain forward -> bitwise within native). Graph-mode DroPE perf parity is a
documented follow-up (would need the rotary built at the extended length for BOTH baseline and
DroPE so the cache shape matches). The persistent B3 orchestration correctly ABORTED the 1M run
on each graph-mode Gate-A fail (gating + setsid persistence both verified working).

## Amendment update — long-context VIABLE + DroPE BUILT (2026-06-08)
MAX_VIABLE_CONTEXT probe: Qwen3.6 ran at max_model_len=262144 with a 200k-tok prompt (135s,
1480 tok/s) -> {passed:true, mode:full, context_length_used:262144}. Long-context regime is
reachable on Thor; Stages 4/5 eval feasible at 262k (>262k needs extension config).
Discriminator nvidia/Llama-3.1-8B-Instruct-NVFP4 CONFIRMED present (pure Transformer, correct
cross-family per LongPPL paper). Qwen3.6 native=262k (text_config), no rope_scaling.
Per RFC-process amendment: DroPE BUILT (not deferred) -- ~20 LOC, default-off DFLASH_DROPE,
identity rotation for positions>=native on the 10 attention layers; within-native gate running.
[Feature] PR draft: pr_drafts/drope.md. Beyond-native recall eval = dedicated session.

## Stage 1 — GATE FAIL: FP8 KV is broken on this stack under BOTH backends (2026-06-08)
FP8 KV (kv_cache_dtype=fp8) on Qwen GDN-hybrid + DFlash spec FAILS to run:
  - FlashInfer backend: TypeError BatchDecodeWithPagedKVCacheWrapper.run() unexpected kwarg
    'kv_cache_sf' (flashinfer.py:1739) -- FlashInfer version/API mismatch.
  - TRITON_ATTN backend: AssertionError during execution.
BF16 KV + TRITON works (tau 3.84). So FP8 KV is blocked UPSTREAM of the --calculate-kv-scales
corruption the stage targets; the calc_kv_scales hybrid-guard fix is MOOT here (FP8 KV never
runs to be corrupted). VERDICT: Stage 1 FAIL on this image -> reverted (no code shipped; the
harness fp8 wiring is test-only, default off). Independent of Stage 2+; continuing.
NEXT SESSION: needs a FlashInfer version with the kv_cache_sf API (or fix the TRITON fp8 path)
before the calc_kv_scales hybrid guard can be validated. The guard design stands (skip calc
for hybrid -> scale=1.0) once FP8 KV runs.

## Stage 0 — HybridCorrectnessGate built (2026-06-08)
tests/v1/worker/test_hybrid_correctness.py: lossless_gate (bitwise T=0), lossy_gate
(degenerate-output detection + first-divergence rate), recall_gate (NIAH accuracy/length).
Self-tests pass. Qwen baseline = baseline_outputs/baseline.json (20-prompt T=0, prior work).

## Stage 1 — FP8 KV: root cause is FlashInfer API mismatch, not calc_kv_scales (in progress)
FP8 KV under the default FlashInfer backend CRASHES on the Qwen GDN hybrid:
`TypeError: BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected keyword 'kv_cache_sf'`
(flashinfer.py:1739) — the same kv_cache_sf API mismatch the 122B serve script documents.
So FP8 KV is blocked by FlashInfer here, BEFORE the calc_kv_scales corruption question.
Workaround under test: --attention-backend TRITON_ATTN (bf16-ref + fp8-calc + fp8-scale1.0).

## LLaDA2.1-mini Maximum-Speed Stack (2026-06-09)
Model: inclusionAI/LLaDA2.1-mini — 16B MoE, ~1.4B active, llada2_moe (20L, 256E×8). Ran in
vllm-dflash-thor:latest via gpu_run guard (zero stuck containers all session).
- RAW CEILING (BF16): **64.9 tok/s = cached_speed** (KV-cache decode). KV cache = ~1.6x over the
  static block-diffusion baseline (38.9->61.2 quality, 30.5->64.9 speed).
- **S2D2 self-speculation did NOT help** at these settings (34-40 tok/s): verification passes add
  ~40-60% NFEs (tok/NFE 5.0-5.8 vs cached 8.1), and Thor is forward-bound. Paper's 4.4x is
  accuracy-matched vs static baseline on full evals/long gens — different axis, not claimed.
- **NVFP4 quantization SUCCEEDED**: 9.5GB (3.4x smaller), nvfp4-pack-quantized; key fix was
  input_ids-only calibration to dodge LLaDA2's bidirectional-mask rejection of llm-compressor's
  2D mask. But raw-transformers decode is **impractical** (CPU-bound dequant, GPU 28%, 256-tok
  nocache >4min, aborted) -> BF16 is the speed choice; NVFP4 belongs to the fused-kernel serving
  path. Calibration caveat: activated-expert only (cold-expert quality risk).
- **vLLM diffusion serving BLOCKED**: needs AlonKellner dllm-fork-coherent (non-causal attn +
  draft buffers + slot remap); base vLLM = causal/wrong. dllm-plugin = MOCK model (Phase 7).
  Official modeling also won't import on transformers 4.57.3 (create_bidirectional_mask). RFC:
  pr_drafts/vllm_diffusion_lm_rfc.md (infra only, no quality claim).
- **SGLang** = most mature dLLM path, supports Thor (Apr-2026, arch 11.0a) + SGLang-Diffusion,
  but native dLLM "not production-stable"; full serve NOT run (fallback, Triton-risk). Highest-
  probability next step for SERVING LLaDA2.x on Thor.
- **FP8 KV**: valid (GQA, 4 KV heads, ~halves KV), but serving-gated (same blocker as vLLM).
- **Draft head**: DFlash/DFlare use diffusion-as-DRAFTER for AR targets (5.46x). Draft head FOR a
  diffusion target is inverted/under-explored and speed-contraindicated here (Stage-4 evidence);
  DFlare-style skeleton written, training deferred (multi-day).
Next sessions: (1) SGLang-Diffusion serve on Thor (build arch 11.0a) — the real serving unlock;
(2) dInfer standalone engine for LLaDA2.1-mini + NVFP4; (3) all-expert NVFP4 recalibration for
the serving path. Models in $HOME/models owned by patrickd.

## LLaDA2.1-mini viability research + roadmap (2026-06-09, addendum)
3-thread research (eval/llada_mini/RESEARCH_ROADMAP.md). KEY CORRECTION to the Stage-5 RFC:
dllm-plugin's REAL LLaDA2ForCausalLM is registered BY DEFAULT (mock is VLLM_DLLM_USE_MOCK_MODEL=1),
and our 0.20.0.dev0+dflash fork already has ~80% of the diffusion infra (MRV2 ModelState,
draft_tokens buffer, use_non_causal in config/attention + spec_decode/dflash). => vLLM diffusion
serving is a ~250-400 LOC port (A2: rebase dllm-fork-coherent delta onto our fork), NOT blocked.
Critical path: (1) port delta -> serve LLaDA2.1-mini via dllm-plugin (TP=1, FlashInfer non-causal);
(2) flip to our 9.5GB NVFP4 artifact -> plugin threads quant_config into FusedMoE -> CUTLASS FP4
grouped GEMM (the kernel our 122B run already proved on Thor) => NVFP4 finally pays off; (3) re-test
gated S2D2 (score_threshold + soft_entropy, gen 512) + scope d3LLM distillation for tokens/forward.
S2D2 honest ceiling here = break-even to +15% (training-free spec can't beat KV-cache on a forward-
bound device); durable wins are fused kernels + distillation (d3LLM ~9.91 tok/forward @ bs1). dInfer
LLaDA2 quant = FP8 (ModelOpt), NOT NVFP4; standalone dInfer+SGLang is the fallback w/ native 2.1 editing.

## LLaDA2.1-mini vLLM diffusion serving — RUNS on our DFlash fork (2026-06-09)
MAJOR (overturns Stage-5 "blocked"): vLLM block-diffusion serving stands up end-to-end on our
0.20.0.dev0+dflash fork via the dllm-plugin (REAL LLaDA2ForCausalLM default + DllmRuntimeScheduler/
Worker + V2 runner). Reaches "Application startup complete", loads weights, KV cache 36.5GiB/950k tok,
serves /v1/models. Fixes required: (1) pip install dllm-plugin with `git config --global --add
safe.directory '*'` (setuptools-scm); (2) patch fork flashinfer.py to pass kv_cache_sf only when
non-None (flashinfer 0.6.6 lacks the kwarg; BF16 -> None); (3) VLLM_PLUGINS=dllm + V2 runner +
LD_PRELOAD. Plugin registers arch LLaDA2MoeModelLM->its own class, bypassing the HF
create_bidirectional_mask import wall. REMAINING BLOCKER (pinpointed via CUDA_LAUNCH_BLOCKING=1):
generation crashes in the fork's AR-spec-decode Triton kernel _combine_sampled_and_draft_tokens_kernel
(input_batch.py:280; model_runner prepare_inputs:832) — illegal memory access because DllmRuntimeScheduler
reuses scheduled_spec_decode_tokens to carry diffusion BLOCK-drafts (block=32, num_bonus=0) and the
AR kernel indexes that layout OOB. Fix = the dllm-fork-coherent runner draft-handoff delta (~250-400 LOC,
route via model_state.take_draft_token_ids + dllm_prefix_lengths field; do NOT break DFlash AR spec path).
Full writeup: eval/llada_mini/vllm_diffusion_result.md. No tok/s reported (gen doesn't complete yet).
Both serving paths now reduce to localized low-level issues: vLLM=1 Triton draft kernel; SGLang=aarch64
sgl-kernel/torch ABI (sgl-kernel 0.3.21->torch 2.9.1 vs image 2.10.0; source-build in progress).
