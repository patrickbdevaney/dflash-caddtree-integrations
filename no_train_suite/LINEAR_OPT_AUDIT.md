# Linear Speculation Optimization Audit
Generated: 2026-06-07
Branch (vLLM fork): `~/vllm` is on **`dflash-thor`** — the directive's `no-train-suite`
branch does NOT exist yet (no per-stage branches created; the no-train suite has not
begun implementation).
Authoring surface: `~/dflash-dev/image-src/` is **FLAT** (e.g. `dflash.py`,
`rejection_sampler.py`), NOT the `vllm/spec_decode/...` tree the directive lists.
All paths below are the real flat files.
Baseline tag: **`linear-opt-baseline` EXISTS** (in the companion repo, not `~/vllm`).

## Summary

| # | Optimization              | Status                    | Correct?              | Flag   |
|---|---------------------------|---------------------------|-----------------------|--------|
| 0 | Baseline lock             | PARTIAL                   | N/A                   | YELLOW |
| 1 | Stage 0 instrumentation   | PARTIAL                   | partial logging only  | YELLOW |
| 2 | NVFP4 audit               | NOT STARTED               | N/A                   | GREY   |
| 3 | Block verification        | NOT STARTED (design: moot)| N/A                   | GREY   |
| 4 | Per-position calibration  | NOT STARTED (design: moot)| N/A                   | GREY   |
| 5 | Adaptive-K / K-push       | NOT STARTED (profiling WIP)| N/A                  | GREY   |
| 6 | Fat-chain verification    | NOT STARTED (design: moot)| N/A                   | GREY   |
| 7 | Typical acceptance        | IMPLEMENTED — in-process bench only | NO — no T=0 greedy guard | YELLOW |
| 8 | GDN state rollback        | PARTIAL (implicit)        | NEEDS RUNTIME VERIF.  | YELLOW |
| 9 | Verify-path kernel fusion | NOT STARTED (linear)      | N/A                   | GREY   |
|10 | MoE expert prefetch       | NOT STARTED               | N/A                   | GREY   |
|11 | CPU suffix drafter        | NOT STARTED               | N/A                   | GREY   |
|12 | Draft top-p tuning        | NOT STARTED               | N/A                   | GREY   |
|13 | GDN prefix caching        | NOT STARTED               | N/A                   | GREY   |
|14 | CUDA graph tuning         | PARTIAL (harness only)    | NEEDS RUNTIME VERIF.  | YELLOW |
|15 | Companion repo docs       | PARTIAL                   | N/A                   | YELLOW |

**Counts: GREEN 0 · YELLOW 5 (items 0,1,7,8,14; item 15 informational) · RED 0 · GREY 9 (items 2,3,4,5,6,9,10,11,12,13).**

> **Critical context:** A committed design-first finding (`IMPLEMENTATION_NOTES.md`,
> commit `d653d1d`) establishes that items **3, 4, 6 are MOOT for DFlash's greedy +
> factorized draft** — they provide no lossless gain (proof in that doc). They are
> GREY not because they're pending implementation but because implementation was
> *correctly declined* after design analysis. The only landed win is item 7 (typical
> acceptance, lossy, T>0), and it is validated **only in the in-process micro-harness**
> (+11% tok/s / +17% τ at T=0.3), **not yet via the production serve path**.

---

## Detail — Item by Item

### Item 0 — Baseline lock
**Status:** PARTIAL  **Correct:** N/A  **Flag:** YELLOW
**Files inspected:**
  - `~/dflash-caddtree-integrations` tags: `linear-opt-baseline` PRESENT.
  - `~/vllm` tags: no `linear-opt*` tag (NOT FOUND).
  - `~/dflash-dev/backups/`: empty (no per-stage snapshots); `~/dflash-dev/_baseline_linear_opt/` snapshot of image-src PRESENT.
  - `~/dflash-dev/cost_models/thor/`: directory exists but **EMPTY** (C_verify curve not saved — k-sweep is mid-run, not yet persisted).
  - `~/dflash-dev/tests/seeds.py`: PRESENT, **20 prompts** (5 code / 5 JSON-tool / 5 reasoning / 5 free-form).
  - `~/dflash-dev/baseline_outputs/`: `baseline.json` (T=0 graphs token IDs) + `baseline_eager.json` PRESENT.
**Implementation summary:** Baseline tag, 20-prompt seed set, and T=0 token-ID
references exist; a full baseline table (graphs+eager, T=0/0.3/0.5) was measured.
**Correctness notes:** Token-ID byte-identity reference exists and is usable.
**Missing pieces:** C_verify(k) curve not yet saved to `cost_models/thor/` (k-sweep
running); no per-stage backup snapshots; baseline tag is on the companion repo, not
`~/vllm`.

### Item 1 — Stage 0 instrumentation
**Status:** PARTIAL  **Correct:** partial  **Flag:** YELLOW
**Files inspected:**
  - `rejection_sampler.py`: `DFLASH_PROBE` (lines 38-41, 218-225) records per round
    `{draft top-1, target greedy, bonus, draft top-2}` into `_PROBE_RECORDS`.
    `DFLASH_DBG` (335-341) prints `DBG_ACCEPT` (path_len, tok_ids, draft_ids, tgt).
  - `llm_base_proposer.py`: `DFLASH_PROBE` (line ~520) stores draft top-2.
  - No `DFLASH_LINEAR_DEBUG`; **NOT FOUND**: the specified per-position cumulative
    *survival product* `Π q_j` and explicit per-position accept/reject decision log.
**Implementation summary:** A τ-ceiling probe + an accept-debug print exist, capturing
top-1/top-2/target/bonus/accept-length — enough for the offline analyses already done.
**Correctness notes:** Logging only; no behavior change. Looks correct.
**Missing pieces:** The exact survival-product / per-position decision log the directive
specifies is absent (the probe captures related but not identical data).

### Item 2 — NVFP4 verify-path kernel audit (Stage F)
**Status:** NOT STARTED  **Correct:** N/A  **Flag:** GREY
**Files inspected:** `no_train_suite/profiles/thor/` empty; no per-layer kernel-dispatch
log anywhere. No FP4 fallback profiling performed.
**Implementation summary:** Not implemented.
**Missing pieces:** Profile per-layer-type kernel dispatch on the verify pass; flag any
GDN/MoE/attention layer that dequant-then-computes. (This is the no-train suite's Stage F.)

### Item 3 — Block verification (Stage 1)
**Status:** NOT STARTED — **design-determined MOOT**  **Correct:** N/A  **Flag:** GREY
**Files inspected:** `rejection_sampler.py` — no `block_verify` / `backward_induction`
flag or logic (NOT FOUND). `DFLASH_BLOCK_VERIFY` absent.
**Implementation summary:** Not implemented, by design. `IMPLEMENTATION_NOTES.md` proves
that for a *deterministic* proposal (DFlash drafts greedily, `_greedy_sample` = argmax),
exact-match is already the lossless-optimal accept (max accept prob = p_i(proposed)); block
verification gains only exist for a *stochastic* draft.
**Missing pieces:** None to pursue as specified. Would require switching DFlash to
stochastic drafting first (uncertain net gain).

### Item 4 — Per-position temperature calibration (Stage 2)
**Status:** NOT STARTED — **design-determined MOOT**  **Correct:** N/A  **Flag:** GREY
**Files inspected:** No `pos_temp.json`, no calibration script (NOT FOUND).
**Implementation summary:** Not implemented, by design. Temperature scaling is monotonic
→ preserves the argmax → for a greedy draft the proposed tokens are unchanged → no effect.
**Missing pieces:** None as specified (moot for greedy draft).

### Item 5 — Adaptive-K / K-push (Stage 3)
**Status:** NOT STARTED (profiling in progress)  **Correct:** N/A  **Flag:** GREY
**Files inspected:** No `throughput` objective, no greedy-stop rule, no pad-and-mask
adaptive-K in `dflash.py`/`llm_base_proposer.py` (NOT FOUND). `c_verify_linear.json` and
`decision_gate.md` NOT FOUND. (The grep hit for "adaptive" in `gdn_linear_attn.py:326` is
an unrelated `qkvz_proj` comment.)
**Implementation summary:** Not implemented. The C_verify(k) k-sweep (k=1,2,4,6,8,10,12)
is running now (Stage -1); its convex-vs-flat verdict will decide adaptive-K vs K-push.
**Missing pieces:** Save the C_verify curve; then implement adaptive-K (if convex) or
test higher K (if flat).

### Item 6 — Fat-chain multi-candidate verification (Stage 4)
**Status:** NOT STARTED — **design-determined MOOT**  **Correct:** N/A  **Flag:** GREY
**Files inspected:** No `fat_chain` / `DFLASH_FAT_CHAIN` / multi-candidate OT loop in
`rejection_sampler.py` (NOT FOUND). (A *tree* selective path exists — `DFLASH_SELECTIVE`,
`ddtree.py` — but that is tree code, out of scope for the linear track.)
**Implementation summary:** Not implemented, by design. At the first reject position j the
token fat-chain would recover *is the target's argmax at j* — already emitted for free as
the correction/bonus; continuing past j needs the target conditional under the corrected
prefix, which the single verify pass did not compute. No continuation → no gain.
**Missing pieces:** None as specified (moot for this pipeline).

### Item 7 — Typical acceptance (Stage 5)
**Status:** IMPLEMENTED — benchmarked **in-process only**  **Correct:** **NO — missing
T=0 greedy guard**  **Flag:** YELLOW
**Files inspected:**
  - `rejection_sampler.py:140-143`: trigger `if _eps_lin > 0 and draft_token_ids is not
    None and len(num_draft_tokens)==1: return self._chain_typical_accept(...)`.
  - `_chain_typical_accept` (233-264): `tprob = softmax(target_logits)`; accept draft[i]
    iff `tprob[i, draft[i]] >= eps`; else stop + emit target argmax bonus. Batch==1 only.
  - Tree typical path (285-291) mirrors it for the tree accept.
  - Flag name is **`DFLASH_ACCEPT_EPS`**, not `DFLASH_TYPICAL`.
**Implementation summary:** Medusa-style threshold acceptance on the lean linear chain;
measured +11% tok/s / +17% τ at T=0.3 in the in-process harness.
**Correctness notes (RED-adjacent):** The trigger checks only `eps>0`, **not temperature**.
At **T=0 with eps>0 it still runs typical acceptance on softmax probs** — it does NOT reduce
to greedy and would NOT be byte-identical to the greedy baseline. The directive's Stage-5
gate ("at T=0 with flag ON, byte-identical to greedy") is therefore **not satisfied**. In
practice eps>0 has only ever been used at T>0 (eps=0 at T=0), so no incorrect output was
produced, but the guard is missing. Also: **batch==1 only** (falls back under batching), and
the win is validated only in the **in-process micro-harness, never via the production serve**.
**Missing pieces:** (a) add `is_greedy`/temperature guard so eps>0 reduces to exact greedy
at T=0; (b) generalize beyond batch==1; (c) validate via the real serve + bench.py.

### Item 8 — GDN state-rollback correctness (Stage A1)
**Status:** PARTIAL (implicit via align-mode)  **Correct:** NEEDS RUNTIME VERIFICATION
**Flag:** YELLOW
**Files inspected:**
  - `gpu_model_runner.py`: `num_accepted = valid_sampled_token_count - 1` (line 1472) is
    derived from the sampler output; `rollback`/optimistic-correction logic present.
  - `gdn_attn.py` / `gdn_linear_attn.py`: align-mode `num_accepted_tokens` promotion.
  - No `recompute_from_scratch` (NOT FOUND); no explicit `restored==recomputed` assertion
    for the **linear** path (the tree work's INV3 assertions are tree-specific).
**Implementation summary:** Rollback is implicit: `num_accepted` tracks the sampler's accept
count and align-mode promotes the corresponding state. No standalone rollback fn/assertion.
**Correctness notes:** Almost certainly correct for the baseline path (num_accepted is
sampler-derived), but unproven for arbitrary accept counts; static inspection cannot confirm.
**Missing pieces:** An explicit per-layer `restored_state == recompute_from_scratch` assert
over the seed set and all accept counts (Stage A1).

### Item 9 — Verify-path kernel fusion (Stage A2)
**Status:** NOT STARTED (for the linear path)  **Correct:** N/A  **Flag:** GREY
**Files inspected:** `/designs/verify_fusion.md` NOT FOUND. The linear GDN verify uses the
existing `fused_sigmoid_gating_delta_rule_update` kernel (already a single fused call per
layer); no extra K-token launch-grouping work for linear. (The *tree* per-branch fusion in
`gdn_linear_attn.py` is tree-mode only.)
**Missing pieces:** Design doc + linear K-token fusion, if profiling shows launch overhead.

### Item 10 — MoE expert prefetch (Stage E)
**Status:** NOT STARTED  **Correct:** N/A  **Flag:** GREY
**Files inspected:** No `prefetch`/`expert_prefetch` anywhere; `/designs/expert_prefetch.md`
NOT FOUND.
**Missing pieces:** Design doc + routing-ahead/prefetch implementation.

### Item 11 — CPU suffix/n-gram drafter (Stage B)
**Status:** NOT STARTED  **Correct:** N/A  **Flag:** GREY
**Files inspected:** `DFLASH_CPU_SUFFIX` absent. The `suffix`/`ngram` grep hits are vLLM's
built-in references in `gpu_model_runner.py` (and unrelated attention files), **not** a
DFlash CPU-side drafter. No async CPU draft path.
**Missing pieces:** The entire CPU suffix-drafter stage.

### Item 12 — Draft top-p / top-k tuning (Stage D)
**Status:** NOT STARTED  **Correct:** N/A  **Flag:** GREY
**Files inspected:** `top_p`/`top_k` hits are the standard sampler params, not a restriction
applied to DFlash *draft marginals*. `draft_topp.json` NOT FOUND.
**Missing pieces:** The draft-marginal top-p/k sweep + saved params.

### Item 13 — GDN prefix caching (Stage C)
**Status:** NOT STARTED  **Correct:** N/A  **Flag:** GREY
**Files inspected:** No `sub_block`/`partial_prefix`/`DFLASH_GDN_APC` (NOT FOUND).
`num_speculative_blocks` exists (`abstract.py`, `mamba_utils.py`, `gpu_model_runner.py`)
from the tree work, but the #39809 coexistence fix and sub-block caching are absent.
`/designs/prefix_caching.md` NOT FOUND.
**Missing pieces:** Design doc + sub-block GDN-state caching + coexistence fix.

### Item 14 — CUDA graph capture tuning (Stage G)
**Status:** PARTIAL (harness only)  **Correct:** NEEDS RUNTIME VERIFICATION  **Flag:** YELLOW
**Files inspected:** `tests/harness.py` sets `cudagraph_capture_sizes=[1, nspec+1]` for the
in-process LLM; `gpu_model_runner.py` has 10 capture-size/uniform-decode references (vLLM
stock). The serve path uses `COMPILATION_CONFIG` env (`_common.sh`). No deliberate capture-
range tuning for adaptive-K pad-and-mask or batch>1 headroom.
**Missing pieces:** Deliberate capture-range tuning once adaptive-K/batch shapes are defined.

### Item 15 — Companion repo documentation state
**Status:** PARTIAL  **Flag:** YELLOW
**Files present (no_train_suite + benchmark_results):**
  - `no_train_suite/final_benchmark_plan.md` (2026-06-07)
  - `no_train_suite/backups/manifest.md` (2026-06-07, header only — no entries)
  - `no_train_suite/LINEAR_OPT_AUDIT.md` (this file)
  - `IMPLEMENTATION_NOTES.md` (repo root — linear-opt finding chain)
  - `benchmark_results/`: `ddtree-gdn.md`, `b-sweep-crossover.md`,
    `staged-optimization-final.md`, `stage5-tree-wy-flop-analysis.md`, `raw_bench.log`
**Files missing (expected by the no-train suite):**
  - `no_train_suite/SUITE_NOTES.md` — NOT FOUND
  - `no_train_suite/decision_gate.md` (Stage D0) — NOT FOUND
  - `no_train_suite/profiles/thor/*` (c_verify, fp4 audit, expert share) — EMPTY
  - `no_train_suite/designs/*` (prefix_caching, gdn_rollback, verify_fusion, expert_prefetch) — EMPTY
  - `no_train_suite/benchmarks/*` — EMPTY
  - `no_train_suite/readable_code/*` — EMPTY
  - `no_train_suite/SUITE_SUMMARY.md` — NOT FOUND

---

## Companion Repo State

**Present:** see Item 15.
**Missing (expected but not found):** `SUITE_NOTES.md`, `decision_gate.md`,
`SUITE_SUMMARY.md`, and all of `profiles/thor/`, `designs/`, `benchmarks/`,
`readable_code/` (dirs exist, empty). `cost_models/thor/` (in `~/dflash-dev`) empty.

---

## Next Actions

1. **Item 5/0 — save C_verify(k) curve** (the k-sweep is running). Persist to
   `cost_models/thor/c_verify_linear.json` + `no_train_suite/decision_gate.md`; report
   convex vs flat. *Blocked by:* the running Stage -1 job. *Effort:* TRIVIAL (write-out).
2. **Item 7 — add T=0 greedy guard** so `eps>0` reduces to exact greedy at T=0 (satisfy the
   lossless gate), then **generalize beyond batch==1**, then validate via the production
   serve + bench. *Effort:* SMALL (guard) / MEDIUM (batch generalization + serve bench).
3. **Item 2 (Stage F) — NVFP4 verify-path audit.** Profile per-layer kernel dispatch; this
   is the validity guard for every later benchmark. *Effort:* MEDIUM (profiling infra).
4. **Item 8 (Stage A1) — explicit GDN rollback assertion** (`restored==recomputed` per layer,
   all accept counts). *Blocked by:* nothing. *Effort:* MEDIUM, fail-quiet → design-first.
5. **Item 13 (Stage C) — GDN prefix caching** (design doc first). Likely the largest win for
   agentic/shared-prompt; fail-quiet. *Effort:* LARGE, design-first.
6. **Items 9/10 (Stage A2/E) — verify fusion + expert prefetch** (design docs first;
   gated on Item 8 correct rollback for A2). *Effort:* LARGE each, fail-quiet.
7. **Items 11/12/14 (Stage B/D/G) — CPU suffix drafter / draft top-p sweep / graph tuning.**
   Independent, lower risk. *Effort:* MEDIUM (B) / SMALL (D) / SMALL (G).
8. **Items 3/4/6 — no action**; design-determined moot for DFlash's greedy+factorized draft
   (documented). Revisit only if the draft is made stochastic.
9. **Item 0 — per-stage backup snapshots** as the suite proceeds (BACKUP PROTOCOL).
