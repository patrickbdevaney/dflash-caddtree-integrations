# [RFC] Tree-based speculative decoding for GDN/Mamba hybrid models

AI-assisted (Claude); human reviews/defends end-to-end. RFC issue (design + reference code) —
NOT a feature PR. The result is honest-negative on the model tested; the contribution is the
abstraction + the cost finding, not a speedup claim.

## Problem / gap
vLLM speculative decoding for hybrid (GDN/Mamba-2) models verifies a single LINEAR draft chain.
There is no tree-based candidate verification — i.e. verifying multiple branching continuations
in one target forward — for hybrids, because the recurrent (SSM/GDN) state must be promoted to
the correct accepted *branch*, which the linear `num_accepted_tokens` machinery does not model.

## Reference implementation (out-of-tree, Qwen3.6-35B-A3B, SM110a)
- DDTree (215 LOC): best-first tree construction over a parallel-draft proposer's position-
  independent marginals (here DFlash, but the structure is proposer-agnostic given top-k
  marginals per depth). TreeNode/DDTree/DDTreeHeap + an [N][N] ancestor mask.
- Tree ancestor-mask attention on the hybrid's ATTENTION layers: each spec node attends to full
  context + its tree ancestors, computed as one eager GQA SDPA (`_tree_eager_attn`). The linear
  path is byte-identical (gated on tree mode).
- Per-branch recurrent-state routing: the GDN state slot is promoted along the accepted path
  (same accept-offset invariant as upstream `mamba_hybrid.num_accepted_tokens`, generalized
  from a chain to a tree path).

## Honest finding (negative — and that is the point)
On Qwen3.6-35B-A3B, tree spec did NOT beat linear DFlash end-to-end. It is depth-dominated, and
the tree-aware GDN recurrence is O(B²d) in branch count B — the quadratic verification cost eats
the acceptance-length gain. Linear DFlash + typical (Medusa) acceptance was the better lever
(+11% in-proc, +26-28% prod @ T=0.3). Reporting this so others don't re-derive the O(B²d) wall.

## Ask to maintainers
Is a *general* per-branch recurrent-state-routing API (tree path promotion for hybrid spec
decode, decoupled from any specific drafter and usable with eagle/MTP) worth landing as infra —
even though tree drafting is model-dependent and showed no speedup here? If yes, the reference
code can be ported to target an upstream drafter and submitted as a WIP PR against the RFC.
If no, it stays out-of-tree on the author's compound fork. Cost analysis + invariants on request.

## Why this is not a feature PR
It does not beat linear on the tested model, and the reference is coupled to DFlash (not
upstream). Filing as RFC first is the correct discipline (vLLM AGENTS.md: no speedup claims
without data; gauge design interest before a large PR). Reference branch can be attached on request.
