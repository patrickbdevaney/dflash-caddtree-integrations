# Stage A2 — verify-loop kernel fusion: ALREADY SATISFIED (linear)

**Finding: no work needed.** The linear DFlash GDN verify already fuses all K spec
tokens into ONE kernel launch per GDN layer:
`gdn_linear_attn.py:1343-1363` (the `elif spec_sequence_masks is not None` path) calls
`fused_sigmoid_gating_delta_rule_update(...)` once, with `cu_seqlens` covering the
`num_spec_decodes` sequences (each K tokens) and `ssm_state_indices`/`num_accepted_tokens`.
There is no per-token Python loop on the linear path (that loop only existed in *tree*
mode, which the prior work fused per-branch). So A2's goal — "fuse per-token GDN updates
across K verify tokens into fewer launches" — is met by the existing kernel call.
**Status: COMPLETE (pre-existing). No change, no risk.**
