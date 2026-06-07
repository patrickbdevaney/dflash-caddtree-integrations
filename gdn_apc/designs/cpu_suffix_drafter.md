# Stage B — CPU suffix/n-gram drafter (design + blocker analysis)

## Blockers found (P0 investigation)
- vLLM's built-in suffix decoding (`method="suffix"`) needs **`arctic_inference`**, which is
  **NOT installed** in the image. (`ngram_proposer.py` IS present and needs no extra dep.)
- vLLM spec-decode runs **exactly one** proposer method; there is **no native "pick better of
  {DFlash, CPU suffix}" two-drafter path**. The hybrid is the part that must be built.

## Tractable design (lossless, fail-loud, overlay-reachable)

Key property: **the target verifies whatever draft is proposed**, so at T=0 (greedy) the
emitted tokens are the target's deterministic continuation **regardless of the draft source**.
⇒ swapping/augmenting the draft changes acceptance length (speed) but is **byte-identical at
T=0 by construction** (the Stage-B gate is automatic).

Plan (in `llm_base_proposer.propose` / `dflash.py`, behind `DFLASH_CPU_SUFFIX=1`):
1. Maintain a per-request suffix/n-gram index over the running token context (CPU, ~µs).
2. Each round, after the DFlash draft is computed, run a CPU match: if the recent context
   suffix recurs earlier with a continuation of length ≥ L at confidence ≥ c, take the
   suffix continuation as the draft for those positions; else keep the DFlash draft.
   ("pick better" = longest high-confidence match, else DFlash.)
3. Return the chosen draft (shape unchanged). The target verify + GDN state machinery are
   untouched (num_accepted still derives from the sampler → GDN promotion auto-tracks).

Risk profile: **fail-loud** (verify gates output); the only real hazard is async/threading
correctness (deadlock/race) if the CPU match runs on a side thread. A synchronous CPU match
(no thread) is simplest and still ~µs for short contexts — recommended first.

Expected gain: large on repetitive/agentic spans (suffix reuse), ~zero on novel reasoning
(report both honestly). No GPU state slot, no extra GDN pass.

## Status
DESIGNED. Implementation is a moderate proposer-side change (context-index plumbing + match +
override). Lossless gate is automatic at T=0; the work is the index + selection + (optional)
async. Deferred to a focused session rather than rushed on the spec-decode proposer path.
