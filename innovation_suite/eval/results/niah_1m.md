# Result: S-NIAH @ 1M — DroPE (Qwen3.6-35B-A3B-NVFP4, SM110a)  [FROZEN]
Owner: niah_1m.py / run_niah.sh. Written ONLY by the NIAH run. Do not edit from other evals.
Date: 2026-06-08 | DroPE factor=4 zero-shot inference-time (no recal), BF16 KV, no spec-decode.

| Position | Hits | Wilson 95% CI |
|---|---|---|
| p05 (~50k)  | 5/5 | [0.566, 1.0] |
| p50 (~500k) | 4/5 | [0.376, 0.964] (one mid-window miss, expected) |
| p95 (~950k) | 5/5 | [0.566, 1.0] (deep-tail headline cell) |
Overall 14/15. Claim tier: scouting (N=5); p95 lo=0.566 -> N>=13 for lo>0.8 headline.
Attestation: eval/attestation/{scores/sniah_1m_*,configs,hardware,raw_outputs}. Log: niah_status.log
