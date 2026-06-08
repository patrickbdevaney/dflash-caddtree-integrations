# Result: RULER @ 1M — DroPE (Qwen3.6-35B-A3B)  [DEFERRED]
Owner: (ruler harness, TBD) / run_ruler.sh. Written ONLY by the RULER run.

DEFERRED (2026-06-08): a container-contention mistake (duplicate disc launches competing for the
GPU) cost ~6h, so RULER is deferred per user decision. RULER at 1M is ~12-18h (quadratic prefill)
and must run as a single clean detached container after LongPPL completes. Standard pipeline:
hsiehjackson/RULER pinned commit, NIAH + multi-hop/aggregation/QA, pass@1, Wilson CI, attestation
prefix scores/ruler_*, log ruler_status.log. Do not start until GPU/docker healthy + LongPPL done.
