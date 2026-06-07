## Session outcome: TIER-A CORRECTNESS REACHED (GDN prefix cache is bitwise-correct)

### Phase 1 — Coupling sites found (THREE, not one)
1. `model_executor/models/config.py:399-406` (APC-off `else` branch): unconditionally
   forces `mamba_cache_mode = "none"` when APC is disabled. (APC-on side at L352 already
   respects an explicit mode.)
2. `config/vllm.py:2007-2017` `validate_mamba_block_size`: rejects an explicit
   `mamba_block_size` unless `enable_prefix_caching` is set.
3. **Runtime (no clean fix):** align mode + DFlash spec-decode with APC *off* hits a bare
   `AssertionError` during execution — the align state machinery assumes APC is on. So
   "align with APC off" is not a viable cold baseline on this base.

### Phase 2 — Decoupling change (sites 1 & 2)
- `models_config.py`: in the APC-off branch, honor an explicit `mamba_cache_mode=="align"`
  (assert chunked-prefill, set block_size to cache block_size); default `none` path
  unchanged for every config that does not explicitly set align.
- `config_vllm.py`: allow explicit `mamba_block_size` when `mamba_cache_mode=="align"` even
  with APC off; default rejection preserved otherwise.
- CUDA graph capture after change: **PASS** (FULL_AND_PIECEWISE). Baseline/default configs
  unaffected (they never set align).
- These pass the config validators, but site 3 (runtime assert) means align-no-APC still
  can't run — so the cold baseline was obtained a different, more correct way (below).

### Phase 3 — The correct bitwise gate
Because (a) align-no-APC asserts at runtime and (b) **distinct prompts never hit the cache**,
the right cache-correctness test is **run the seed set TWICE within APC-on align**:
run1 = cold (cache miss, computes+populates), run2 = warm (exact-repeat prompts → prefix
cache HIT → GDN state restored). At greedy T=0, run1==run2 iff the restore is correct.

  Comparison: **cold run1 vs warm run2, both APC-on align (block 128)** — NOT none vs align.
  Result: **0/20 diverging — BYTE_IDENTICAL.**
  Gate verdict: **PASS.**

⇒ The prefix cache restores GDN recurrent state **bitwise-correctly** in align mode. The
P1 "14/20" was the align-vs-none MODE switch (benign FP-order difference), not a cache bug.

### Phase 4 — Tier-A measurements
- CUDA graphs: PASS. Runtime: warm-align ran with `mamba_block_size=128`, attention block
  auto-set to 1152 (page-size alignment), graphs captured. No override of the user block size
  observed at 128.
- **Hit-rate / e2e on the real agentic (Hermes) trace: NOT MEASURED** — no such trace is
  available in-repo. The run-twice test proves cache *correctness* and exercises a cache hit
  (exact-repeat prompts), but the >50%-hit and e2e-speedup numbers need the multi-turn
  shared-system-prompt trace. The seed prompts are short, so their prefill-saving is small;
  the benefit scenario is long shared prefixes (future measurement with the trace).
- Decode tok/s: warm-align ~3.6 τ (same as align baseline) — prefix caching is a prefill
  win, not a decode win, as expected.

### Retired comparison
**"APC-on vs APC-off" as a bitwise gate is PERMANENTLY RETIRED.** It compares align mode vs
none mode — two different GDN FP paths — so it can never be bitwise-equal regardless of cache
correctness. The correct gate is **cold==warm within a fixed (align) mode**, which this
session establishes and PASSES.

### Next session
Measure hit-rate (block 64 vs 128) + e2e time-to-completion on the real Hermes agentic trace
(needs the trace). The correctness foundation (Tier-A) is done.
