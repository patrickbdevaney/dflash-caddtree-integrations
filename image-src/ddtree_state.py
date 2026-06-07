# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN branch recurrent-state machinery for DDTree tree verification.

Implements the branch-state solution the DDTree paper deferred for recurrent
architectures. The key result (validated in tests/test_branch_kernel.py): the
existing batched varlen `fused_sigmoid_gating_delta_rule_update` kernel already
supports tree verification with NO kernel surgery, by treating each root->leaf
path as an independent "sequence":

  INVARIANT 1/2: allocate disjoint node slots per tree; seed each branch root
                 slot with a COPY of the canonical (accepted) state.
  INVARIANT 3:   canonical slot is never referenced by the kernel's index set,
                 so it is untouched even with inplace_final_state=True.
  INVARIANT 5:   promotion = copy the accepted branch's leaf slot -> canonical.

The reference implementation runs the W branches as W batched sequences; shared
prefixes are recomputed per path (correct; redundant at node_budget<=22). A fused
multi-branch kernel that dedups shared-prefix recompute is future work.
"""
from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)


def seed_branch_roots(pool: torch.Tensor, canonical_slot: int,
                      branch_root_slots: list[int]) -> None:
    """INV2: initialize every branch root slot to a copy of canonical state."""
    canon = pool[canonical_slot]
    for s in branch_root_slots:
        pool[s].copy_(canon)


def promote_branch(pool: torch.Tensor, canonical_slot: int,
                   accepted_leaf_slot: int) -> None:
    """INV5: atomic exact promotion of the accepted branch leaf to canonical."""
    pool[canonical_slot].copy_(pool[accepted_leaf_slot])


def build_branch_index_tensors(
    branch_paths: list[list[int]],
    node_to_slot: list[int],
    device: torch.device,
    spec_width: int,
):
    """Map DDTree branch paths -> per-branch ssm_state_indices rows + cu_seqlens.

    branch_paths: list of root->leaf flattened node indices (from DDTree).
    node_to_slot: physical pool slot reserved for each tree node.
    Returns (ssm_state_indices [N, spec_width], cu_seqlens [N+1],
             num_accepted [N], path_lengths). Each row's first slot is the
             branch root (pre-seeded with canonical); subsequent slots are the
             path's node slots in order. num_accepted=1 -> kernel seeds from the
             root slot (== canonical copy).
    """
    n = len(branch_paths)
    ssi = torch.zeros(n, spec_width, dtype=torch.int32, device=device)
    lens = []
    for i, path in enumerate(branch_paths):
        slots = [node_to_slot[node] for node in path]
        L = min(len(slots), spec_width)
        ssi[i, :L] = torch.tensor(slots[:L], dtype=torch.int32, device=device)
        lens.append(L)
    cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=device), 0)
    num_accepted = torch.ones(n, dtype=torch.int32, device=device)
    return ssi, cu, num_accepted, lens


def run_branch_recurrence(
    *, A_log, a, b, dt_bias, q, k, v, pool,
    ssm_state_indices, cu_seqlens, num_accepted,
    use_qk_l2norm_in_kernel=True,
):
    """Run all tree branches as batched sequences through the existing kernel.

    q/k: [1, T, H, K]; v: [1, T, HV, V]; a/b: [1, T, HV] flattened in branch
    order matching cu_seqlens. Writes each node's state into its reserved pool
    slot (inplace over the disjoint node slots); canonical is untouched.
    """
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
        initial_state=pool, inplace_final_state=True,
        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
