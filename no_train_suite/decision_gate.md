# Stage D0 — Decision Gate

## (ii) C_verify(k) breakdown — MEASURED (aggregate)

CUDA graphs, T=0, 20-prompt mixed seed set. `per_step_ms = C_draft + C_verify(k)`:

| k | per_step_ms | τ | tok/s |
|---|---|---|---|
| 1 | 47.9 | 1.82 | 38.0 |
| 2 | 50.8 | 2.44 | 48.0 |
| 4 | 57.5 | 3.13 | 54.4 |
| 6 | 63.5 | 3.35 | 52.7 |
| 8 | 65.1 | 3.54 | 54.3 |
| 10 | 68.8 | 3.76 | 54.6 |
| 12 | 70.8 | 3.99 | **56.3** |

**Verdict: C_verify(k) is INCREASING, ≈linear at +2.1 ms/token. NOT flat/bandwidth-bound.**
Per the directive's Stage-3 rule, this means **adaptive-K is applicable** (not K-push).
BUT tok/s **plateaus from k≈4 to 12** (52.7–56.3, max at k=12) — so the adaptive-K upside is
**bounded**: the best static k (12) already sits at the plateau top; adaptive-K can only save
verify cost on low-acceptance rounds (use k<12 when the draft is unconfident). Expected gain
is modest (single-digit %), not transformational.

Note: this is the *aggregate* per-step cost. A per-layer-type breakdown (MoE expert fetch vs
GDN state vs attention) was NOT measured here — that requires per-layer profiling (Stage F),
which is the next profiling task and is needed to confirm where the +2.1 ms/token goes.

## (i) Prefix-cache hit rate — NOT MEASURED

Requires a representative multi-turn agentic transcript with the shared system prompt; no such
trace is available in-repo. **NOT MEASURED** (honest gap). Cannot compute the agentic vs
single-turn branch from real data.

## Chosen order

Branch is undecidable without (i). Defaulting to the **single-turn / batch=1** ordering
(our actual benchmark regime), and deprioritizing Stage C (prefix caching) until an agentic
trace is available to justify its (large, fail-quiet) cost:

  order = [F (fp4 audit), A+E (verify internals), B (cpu suffix), D (top-p), G (graph)]
  C (prefix caching) — deferred pending an agentic trace + hit-rate measurement.
