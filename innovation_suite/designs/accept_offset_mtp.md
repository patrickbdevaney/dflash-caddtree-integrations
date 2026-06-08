# Stage 3 — accept-offset rollback for MTP + DFlash proposers (design-first)

## Background
PR #40738 fixed GDN/Mamba-2 SSM-state corruption under **ngram** spec-decode by selecting
the correct intermediate state slot to persist via `spec_decode_src_indices` /
num_accepted offset in `worker/mamba_utils.py:postprocess_mamba`. Scope was ngram only.
DFlash and MTP go through different proposer paths (#40880) and need the same invariant.

## The invariant (both paths)
After a verify round that accepted M of K drafted tokens, the persisted SSM/recurrent state
MUST be the state **after exactly M accepted tokens** — not after K (all drafted) and not
after M-1. Formally: `promoted_state_position == num_accepted_tokens`, every round.

## Where it already holds vs. where to verify
- The image's `postprocess_mamba` (verified earlier) carries `num_accepted_tokens` offset
  logic (L214/247/251/272) — i.e. #40738 is PRESENT for the path that uses postprocess_mamba.
- DFlash linear: `num_accepted = valid_sampled_token_count - 1` (gpu_model_runner:1472) is
  sampler-derived, and align-mode promotion uses it. Prior cold==warm bitwise (gdn_apc) +
  GDN INV3/INV5 tree tests already exercised partial acceptance with NO state corruption.
  So the DFlash path is BELIEVED correct on this base — the stress test below PROVES it.
- MTP: the Nemotron serve script omits spec-decode ("MTP omitted for stability") and MTP is
  a different proposer; testing requires the 120B model + a working MTP method name. Treated
  as Block-B smoke (stability bar), not a full gate.

## Stress test (the gate)
Force M < K acceptance and assert no corruption:
  - Hook (DFLASH_FORCE_ACCEPT=M): in the linear rejection sampler, truncate the accepted
    prefix to M before emitting (M in {1,3,8}). The GDN promotion then sees num_accepted=M.
  - Gate: over the 20-prompt seed set, output is NON-degenerate (lossy_gate: no repeated-token
    / fixation), AND a per-round assert `promoted_pos == M` fires 0 violations.
  - This isolates accept-offset correctness from acceptance *rate*.

## Status
DESIGN. DFlash path: stress-testable on this base (Qwen). MTP path: Block-B Nemotron smoke
(stability only). If the DFlash stress test shows degenerate output at forced M<K, that is the
#40880 bug reproduced -> port the postprocess_mamba offset selection to the DFlash promotion.
