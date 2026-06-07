# GDN-APC — summary (real results)

Base: `vllm-dflash-thor:fa-native` = vLLM 0.20.0.dev0+dflash (prebuilt overlay; not nightly).

| Config | launches? | T=0 bitwise APC==off | hit rate | notes |
|--------|-----------|----------------------|----------|-------|
| baseline (no APC) | n/a | (ref) | 0% | — |
| align + block 128 (P1) | **YES** (no #39809 crash, graphs capture) | **NO — 14/20 diverge** | not measured | wrong-state (cache already bf16; not a dtype fix) |
| P2/P3/P4 | not attempted | — | — | gated on P1 correctness, which failed |

**Tier reached: neither A nor B.** P1 **launches** but fails the **sacred bitwise T=0 gate**
(wrong-state divergence on a base that predates the nightly #39809 Bug-1 fix). Per directive,
STOPPED — divergent output must not ship. dtype is already bf16/"auto" (not the cause).

**Path to Tier-A (future, fail-quiet, design-first):** port the nightly #39809 Bug-1/Bug-2
.py fixes + the correct align-mode GDN-state save/restore for DFlash spec-decode into the
overlay, then re-run the bitwise gate. agentic hit-rate measurement also pending (no trace).

**Stage B (CPU suffix drafter):** blocked natively (no arctic_inference; no two-drafter path);
designed as a lossless proposer-side n-gram override (designs/cpu_suffix_drafter.md), deferred.

No fabricated numbers. Bitwise gate enforced by real token-ID diff. Zero divergent output shipped.
