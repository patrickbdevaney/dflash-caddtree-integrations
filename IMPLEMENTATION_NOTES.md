# Linear DFlash optimization — implementation notes & finding chain

## Foundational finding (changes the whole plan)

`gpu_model_runner.py:3521-3523` calls the rejection sampler with **`draft_probs=None`**.
Consequence: the linear DFlash accept path is **exact-match**, even at T>0
(`NO_DRAFT_PROBS` → accept draft token i iff it equals the target's sampled token).
This is *why* linear τ degrades at temperature (6.33 → ~5.1 eager): at T>0 the
target sample diverges from the draft's top-1, and exact-match rejects it.

**Second finding — the draft is GREEDY (deterministic):** `_greedy_sample` returns
`compute_logits(h).argmax` (`llm_base_proposer.py:392-396`) at *all* temperatures.
DFlash's draft is also *factorized* (diffusion, position-independent marginals).

**This overturns the directive's lossless stages for DFlash.** For a *deterministic*
proposal x_i (a point mass), the maximum lossless acceptance probability at any
position is `p_i(x_i)` — exactly what exact-match already achieves (accept iff the
target's sample equals the proposed token, prob `p_i(x_i)`). Therefore:
- **Stage 1 (block verification / proper spec sampling): NO lossless gain.** With a
  point-mass proposal, min(1, p/q) = p(x_i) and resample-on-reject = the target's
  own correction — identical to exact-match. Block verification's gain exists only
  for a *stochastic* draft.
- **Stage 2 (per-position temperature): NO gain for a greedy draft.** Temperature
  scaling is monotonic → preserves argmax → the proposed tokens are unchanged.
- **Stage 4 (fat-chain top-W at the reject): NO gain.** The token it would recover
  at reject position j *is the target's argmax at j* — which exact-match already
  emits for free as the correction/bonus. Continuing past j would need the target's
  conditional under the corrected prefix, which the single verify pass did not
  compute (factorized draft doesn't help: the target is autoregressive). So no
  continuation, no gain.

**The only lossless levers that fit DFlash:**
1. **Stochastic drafting + proper spec sampling** (sample x_i ~ q_i at T>0 instead
   of argmax; then min(1,p/q) acceptance). Lossless; accept rate becomes the
   overlap Σ min(p_i, q_i), which can exceed the greedy peak `p_i(top1)` when the
   target is *not* peaked (higher T). Gain is uncertain and could net-hurt if the
   draft is well-tuned for greedy. **This is the one worth measuring.**
2. **Higher K** if `C_verify(k)` is flat (bandwidth-bound on Thor) — drafting more
   tokens is then nearly free. (Prior k-sweep found k=12 optimal, k=15 slower, so
   likely already at the ceiling — the Stage -1 C_verify curve will confirm.)

At T=0 greedy, everything stays exact-match → byte-identical by construction.

## GDN state hazard (Stage 1d) — resolved by construction

`num_accepted = valid_sampled_token_count - 1` (`gpu_model_runner.py:1472`) is
derived from the rejection-sampler output. So when the sampler accepts more (proper
spec sampling / block verification), `num_accepted` auto-tracks it, and the GDN
align-mode promotion uses that same count. The #39273-class hazard is therefore
handled — but Stage 1 will still assert `promoted_state_position == accept_count`
explicitly on every round.

## Marginals are retainable

The proposer computes draft logits via `self.model.compute_logits(sample_hidden_states)`
(`llm_base_proposer.py:490/522`). `q = softmax(logits)` is `[K, vocab]` and can be
stored at propose time and threaded to the verify-step rejection sampler (parallel
to how `draft_token_ids` are carried across the propose→verify step boundary).

## Stage plan (revised, gated)

- **-1 baseline**: lock baseline table + token IDs + C_verify(k) curve. (running)
- **0 instrumentation**: per-round survival/accept debug (env-gated).
- **1a plumb q + proper spec sampling**: lossless T>0 τ gain over exact-match.
  Gate: T=0 byte-identical; T>0 τ ≥ baseline; state-mismatch asserts = 0.
- **1b block verification**: optimal accept count over the K-block. Gate as 1a.
- **2 per-position temp calibration**: sharpen draft proposals; lossless@T=0.
- **3 adaptive-K**: only if C_verify(k) convex; else push K to graph ceiling.
- **4 fat-chain top-W at first reject**: SpecTr OT acceptance, lossless. Novel.
- **5 typical acceptance**: DONE in prior work (lossy, T>0, opt-in; +11% tok/s,
  +17% τ at T=0.3 on the lean linear path via `_chain_typical_accept`).

## Honest note on relation to prior work

The prior session already shipped **typical (Medusa) acceptance** (lossy) on the
linear path: +11% tok/s, +17% τ at T=0.3. Stages 1/1b here pursue the **lossless**
alternative (proper spec sampling / block verification with the real draft probs),
which preserves the target distribution exactly. If lossless 1a/1b approach the
lossy typical gain, prefer them (no quality cost).
