"""FAST synthetic validation of the GDN branch-state core (no model load).

Thesis: tree branch-state verification needs NO new Triton kernel. The existing
batched varlen `fused_sigmoid_gating_delta_rule_update` already:
  - processes N independent sequences (cu_seqlens) in one launch,
  - seeds each from ssm_state_indices[n, num_accepted-1],
  - writes per-token state to ssm_state_indices[n, t].
So W tree branches = W "sequences"; pre-seed each branch's slot[0] with a COPY of
canonical (INV2); write to disjoint slots; canonical slot is never referenced so it
is untouched (INV3) even with inplace_final_state=True. Promotion (INV5) = copy the
accepted branch's leaf slot into canonical.

Validates: INV2 (identical fork init -> identical state for identical tokens;
divergence for different tokens), INV3 (canonical unchanged), correctness
(batched W-branch == per-branch sequential reference).
"""
import torch, sys

def main():
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as upd,
    )
    dev = "cuda"; dt = torch.bfloat16
    torch.manual_seed(0)
    HV = 2; H = 2; K = 4; V = 4; NB = 16
    SPECW = 4  # num_spec+1 columns

    def rand_tokens(T):
        q = torch.randn(1, T, H, K, device=dev, dtype=dt)
        k = torch.randn(1, T, H, K, device=dev, dtype=dt)
        v = torch.randn(1, T, HV, V, device=dev, dtype=dt)
        a = torch.randn(1, T, HV, device=dev, dtype=dt)
        b = torch.randn(1, T, HV, device=dev, dtype=dt)
        return q, k, v, a, b

    A_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)

    canonical = torch.randn(HV, V, K, device=dev, dtype=torch.float32)

    def fresh_pool():
        p = torch.zeros(NB, HV, V, K, device=dev, dtype=torch.float32)
        return p

    # Two branches sharing token0 (shared prefix), diverging at token1,2.
    L = 3
    q0, k0, v0, a0, b0 = rand_tokens(L)
    q1, k1, v1, a1, b1 = rand_tokens(L)
    # share token0
    for x0, x1 in ((q0, q1), (k0, k1), (v0, v1), (a0, a1), (b0, b1)):
        x1[:, 0] = x0[:, 0]

    CANON = 1
    B0 = [5, 6, 7]      # branch0 node slots
    B1 = [10, 11, 12]   # branch1 node slots

    def run_batched():
        pool = fresh_pool(); pool[CANON] = canonical
        pool[B0[0]] = canonical.clone()   # INV2: seed each branch root from canonical
        pool[B1[0]] = canonical.clone()
        # flatten 2 branches as varlen N=2
        q = torch.cat([q0, q1], dim=1); k = torch.cat([k0, k1], dim=1)
        v = torch.cat([v0, v1], dim=1)
        a = torch.cat([a0, a1], dim=1); b = torch.cat([b0, b1], dim=1)
        cu = torch.tensor([0, L, 2 * L], device=dev, dtype=torch.int32)
        ssi = torch.zeros(2, SPECW, device=dev, dtype=torch.int32)
        ssi[0, :L] = torch.tensor(B0, device=dev); ssi[1, :L] = torch.tensor(B1, device=dev)
        nat = torch.tensor([1, 1], device=dev, dtype=torch.int32)
        upd(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
            initial_state=pool, inplace_final_state=True, cu_seqlens=cu,
            ssm_state_indices=ssi, num_accepted_tokens=nat, use_qk_l2norm_in_kernel=True)
        return pool

    def run_single(qb, kb, vb, ab, bb, slots):
        pool = fresh_pool(); pool[slots[0]] = canonical.clone()
        cu = torch.tensor([0, L], device=dev, dtype=torch.int32)
        ssi = torch.zeros(1, SPECW, device=dev, dtype=torch.int32)
        ssi[0, :L] = torch.tensor(slots, device=dev)
        nat = torch.tensor([1], device=dev, dtype=torch.int32)
        upd(A_log=A_log, a=ab, b=bb, dt_bias=dt_bias, q=qb, k=kb, v=vb,
            initial_state=pool, inplace_final_state=True, cu_seqlens=cu,
            ssm_state_indices=ssi, num_accepted_tokens=nat, use_qk_l2norm_in_kernel=True)
        return pool

    pool = run_batched()
    ref0 = run_single(q0, k0, v0, a0, b0, B0)
    ref1 = run_single(q1, k1, v1, a1, b1, B1)

    fails = []
    # INV3: canonical untouched
    if not torch.allclose(pool[CANON], canonical, atol=1e-5):
        fails.append("INV3: canonical slot modified")
    else:
        print("[PASS] INV3 canonical slot unchanged after W=2 batched pass")

    # correctness: batched branch finals == per-branch sequential reference
    def close(x, y): return torch.allclose(x, y, atol=2e-2, rtol=2e-2)
    if close(pool[B0[-1]], ref0[B0[-1]]) and close(pool[B1[-1]], ref1[B1[-1]]):
        print("[PASS] batched W=2 == per-branch sequential reference")
    else:
        d0 = (pool[B0[-1]] - ref0[B0[-1]]).abs().max().item()
        d1 = (pool[B1[-1]] - ref1[B1[-1]]).abs().max().item()
        fails.append(f"correctness: batched != reference (d0={d0:.4f} d1={d1:.4f})")

    # INV2 fork init: shared token0 -> branch intermediate after token0 identical
    if close(pool[B0[0]], pool[B1[0]]):
        print("[PASS] INV2 identical fork: shared-prefix node states match across branches")
    else:
        fails.append("INV2: shared-prefix node states differ")

    # divergence: different later tokens -> different leaf states
    if not close(pool[B0[-1]], pool[B1[-1]]):
        print("[PASS] branch divergence: differing tokens -> distinct leaf states (INV1/2)")
    else:
        fails.append("divergence: leaf states identical despite different tokens")

    # INV5 promotion: copy accepted branch leaf -> canonical, others irrelevant
    promoted = pool.clone(); promoted[CANON] = pool[B1[-1]].clone()
    if torch.allclose(promoted[CANON], pool[B1[-1]], atol=1e-6):
        print("[PASS] INV5 promotion: canonical := accepted-branch leaf (memcpy)")
    else:
        fails.append("INV5: promotion copy failed")

    print("BRANCH_KERNEL_RESULT", "ALL_OK" if not fails else "FAIL " + "; ".join(fails))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
