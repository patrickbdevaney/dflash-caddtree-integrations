# Phase B3 (1M retrieval) — cross-session monitoring

A detached `setsid` run drives the 1M proof so it survives session/token boundaries.

## To monitor from a NEW session
```
tail -40 ~/dflash-dev/innovation_b3_status.log     # overall progress
pgrep -f run_b3.sh                                  # is the orchestrator alive?
docker ps                                           # is a vllm container running?
cat /tmp/niah_1000000.log 2>/dev/null | tail -30    # per-context NIAH detail (or 750000/512000)
```

## What it does (run_b3.sh, detached)
1. Builds the overlay (cache-based DroPE).
2. **GATE A** — within-native bitwise: DroPE+graphs vs baseline (must be BYTE_IDENTICAL).
   If FAIL → aborts the 1M run (DroPE cache fix wrong); status logs `GATE_A=FAIL`.
3. **Phase B3** — S-NIAH at ctx 1M→750k→512k (OOM step-down), DroPE-extended, **BF16 KV,
   NO spec-decode** (P1/P2), synthetic magic-number needles at {5%,50%,95%}, N=5/pos,
   Wilson 95% CI. Writes `NIAH_RESULT` JSON + per-position `NIAH_POS ... Wilson95 [lo,hi]`.

## Success / claim criteria
- GATE_A=PASS required first (DroPE correct + graph-safe).
- Headline claim licensed only if p95 (~950k or proportional) Wilson lower-bound > 0.8
  (needs N≥13; first pass is N=5 scouting -> if p95 ≥4/5, RE-RUN p95 at NIAH_N=20 next session
  for the licensed claim). p05 must pass (sanity); p50 weakest expected (lost-in-middle).
- Honest framing: retrieval measured WITHOUT spec-decode, BF16 KV, DroPE inference-time
  extension (no recalibration checkpoint) -> "zero-shot extrapolation baseline".

## If GATE_A FAILS
The cache-based DroPE has a layout bug. Check `cos_sin_cache` cos/sin half ordering in the
installed vLLM rotary (the surgery assumes cos=first half, sin=second half). Fix the identity
write accordingly, rebuild, re-run.
