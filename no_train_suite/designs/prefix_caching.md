# Stage C — GDN prefix caching: DEFERRED (design sketch)

**Deferred** per the decision gate: the win is agentic/shared-prompt specific, and no
representative multi-turn trace + hit-rate measurement is available to justify its large,
fail-quiet cost. Sketch retained for when a trace exists.

**Problem:** align-mode caches only fully-completed blocks (528/2096 tok) → sub-block
prompts get 0% hit and recompute the full prefix every turn; plus #39809 (prefix-caching +
spec-decode crashes on hybrid Mamba: missing num_speculative_blocks in the cudagraph buffer).

**Approach (when pursued):** sub-block GDN-state serialization at non-block-aligned
boundaries + the num_speculative_blocks coexistence fix. CORRECTNESS GATE: a warm (cached)
run must be byte-identical at T=0 to a cold run; per-layer assert restored-state ==
recomputed-state. Fail-quiet → design-first + hard cold==warm gate before enabling.

**Prerequisite:** an agentic transcript with the shared system prompt to measure hit-rate
before/after (D0 part i, currently NOT MEASURED).
