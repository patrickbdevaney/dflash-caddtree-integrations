# [RFC] Tree speculation for GDN-hybrid spec decoding (spec_state_indices routing)

AI-assisted (Claude); human reviews/defends end-to-end. Not a PR — design discussion first.

## Problem
vLLM spec decode is linear (single draft chain). Tree drafting (multiple candidate
branches verified in one forward) can raise accepted-tokens/step. For GDN/Mamba-2 hybrids
the recurrent SSM state must be promoted to the correct accepted branch — a tree needs
per-node state routing.

## Design explored (out-of-tree, Qwen3.6-35B-A3B)
- spec_state_indices tensor: route each tree node to its SSM state slot; promote the
  accepted path's slot by num_accepted (the accept-offset invariant, same machinery as
  upstream mamba_hybrid num_accepted_tokens).
- Tree-WY attention mask for branch verification in one pass.

## Honest finding (the contribution is the abstraction, not a universal speedup)
On this model tree spec did NOT beat linear DFlash end-to-end — depth-dominated; the
Tree-WY GDN recurrence is O(B²d) in branch count B, which eats the acceptance gain. Linear
DFlash + typical acceptance was the better lever (+11% in-proc, +26-28% prod @ T0.3).
The reusable contribution is the per-node state-routing abstraction + the cost analysis,
so the next hybrid model can decide tree-vs-linear from the O(B²d) bound rather than
re-discovering it.

## Ask
Is a per-node SSM-state-routing API worth landing as infra (even if tree drafting is
model-dependent), or kept out-of-tree? Cost analysis + invariants available on request.
