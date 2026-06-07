# Stage A1 — GDN state-rollback correctness assertion (design)

**Goal:** prove (not just assume) the GDN recurrent state after a verify round equals the
state of a from-scratch recurrence over exactly the accepted tokens, for ANY accept count.

**Current mechanism (implicit, believed correct):** the fused kernel evolves state for all
K spec tokens; `num_accepted = valid_sampled_token_count - 1` (`gpu_model_runner.py:1472`,
sampler-derived) selects the promotion position via align-mode `ssm_state_indices` /
`num_accepted_tokens`. So rollback = "promote the slot at the accepted position." No state
is mutated in place past the accepted token in a way that corrupts the next round, because
the canonical slot is chosen by num_accepted.

**The assertion to add (verification harness, env-gated, NOT production):**
For the first N rounds, for each GDN layer:
  1. Capture the promoted state `S_promoted` (slot at num_accepted-1) after the round.
  2. Recompute from scratch: run `fused_sigmoid_gating_delta_rule_update` over ONLY the
     `num_accepted` accepted tokens starting from the round's seed state → `S_recompute`.
  3. assert `allclose(S_promoted, S_recompute, atol=tol)`; log `DBG_ROLLBACK_MISMATCH` else.
Cover the accept-count distribution actually produced (baseline K, and — when enabled —
typical-acceptance counts). Adaptive-K/fat-chain are MOOT here (not implemented), so their
accept-count ranges are N/A.

**Risk:** assertion-only (no behavior change); the recompute is expensive → gate to first N
rounds under `DFLASH_ROLLBACK_CHECK=1`. Fail-quiet surface, so this is the right guard.

**Status:** DESIGNED. Implementation deferred (verification harness; safe but non-trivial);
the audit already rates rollback correctness as "needs runtime verification" — this doc is
the plan to close it.
