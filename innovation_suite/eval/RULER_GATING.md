# RULER NIAH eval — GATED, not started (2026-06-08)

PF-1 not met: RULER requires the S-NIAH `NIAH_RESULT`, which is NOT yet written.
- S-NIAH 1M run is RUNNING (pid 2337256, container keen_jones), GPU 98% (computing, not hung),
  but SLOW: the 1M prefill is quadratic in context (10 attention layers) -> ~30-60 min/instance
  in eager, ~many hours for 15 instances (matches the directive's ~14h estimate for 1M).
- Therefore RULER (another ~18h of GPU, one-container rule) is a FUTURE SESSION after S-NIAH
  completes. Starting it now would violate PF-1 and the one-container rule.

When S-NIAH completes (NIAH_RESULT in ~/dflash-dev/niah_status.log), the next session:
1. Commit the S-NIAH result with attestation (hardware/config/raw_outputs).
2. Then run RULER per the directive (Steps 1-5): clone RULER pinned, serve DroPE@MAX_VIABLE,
   NIAH subset at 262k then 1M, Wilson CI, signed commits.

GPG: no key configured. Use DCO (-s) for now; the user should set up a GPG key for -S, OR
explicitly authorize auto-generating one + enabling global commit.gpgsign (invasive).

Attestation infra (dirs + attest_functions.sh) is staged here, ready for the RULER session.
