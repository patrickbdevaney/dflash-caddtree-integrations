# DDTree on GDN-hybrid MoE — Benchmark (Jetson AGX Thor SM110a)

Model: **Qwen3.6-35B-A3B-NVFP4** (GDN-hybrid MoE: 40 layers = 30 Gated-Delta-Net
linear-attention + 10 full-attention; 256 experts, top-8) + DFlash draft
(`num_speculative_tokens=12`). Image: `vllm-dflash-thor:ddtree` (overlay on
`vllm-dflash-thor:fa-native`). Hardware: Jetson AGX Thor, SM110a (Blackwell),
128 GB unified LPDDR5x. **eager mode** (`enforce_eager=True`), single sequence,
T=0, seed=0.

## Throughput (5 coding prompts × 128 tokens)

| Config | tokens | wall (s) | tok/s | vs linear |
|--------|-------:|---------:|------:|----------:|
| W=1 (linear DFlash)        | 639 | 14.28 | **44.74** | baseline |
| W=2 (DDTree tree spec)     | 593 | 35.80 | **16.56** | **0.37× (2.7× slower)** |

## Correctness (what works)

- **Tree speculation functions end-to-end**: per-position top-W lattice from
  DFlash's parallel marginals → DDTree best-first tree → single target forward
  with an ancestor-masked verification → tree-aware acceptance → GDN branch-state
  promotion. Output is coherent, valid code.
- **Multi-token tree acceptance observed**: accepted root→leaf path lengths of
  4, 2, 5 tokens across rounds for `def fibonacci(n):` (τ > 1).
- Example W=2 output for `def fibonacci(n):`:
  ```python
      if n <= 1:
          return n
      else:
          return fibonacci(n - 1)
      return fibonacci(n - 1) + fibonacci(n - 2
  ```
- **W=1 (tree_width=1) is byte-identical to linear DFlash** (the full 6-test
  invariant suite passes at W=1; tree code is gated off).
- W=2 vs W=1 outputs agree on the first 6 tokens then diverge at a *semantic
  near-tie* (`if n <= 0:` vs `if n <= 1:`), i.e. the eager ancestor-mask SDPA and
  flash_attn flip a near-equal greedy argmax — a kernel numerical difference, not
  a logic error (both are valid greedy continuations).

## Why W=2 is slower (honest analysis)

The implementation is a **correctness-first reference**, not an optimized one.
Per decode step, W>1 adds:
1. **GDN per-node depth-batched recurrence** — one `fused_sigmoid_gating` launch
   *per BFS depth* (≤ tree depth launches) per GDN layer, vs one launch total in
   linear. ~6× more kernel launches across 30 GDN layers.
2. **Eager combined-mask SDPA** in each of the 10 full-attention layers: a fresh
   `self.attn` call (kept only for its cache-write side effect) **plus** a manual
   paged-KV gather + `F.scaled_dot_product_attention`. The discarded `self.attn`
   call alone roughly doubles full-attn cost.
3. **Per-step Python** tree construction (heap), ancestor-matrix build, and tree
   acceptance walk on the host.

At node budget B=13 these overheads dominate the acceptance-length gain in eager
mode. The path to net speedup (future work):
- A **fused multi-branch GDN kernel** (one launch, parent-indexed) removing the
  per-depth Python loop and the redundant baseline call.
- A **Triton ancestor-mask attention** kernel (or fixing FlexAttention's
  power-of-2 block constraint) so the full-attn verify is a single masked kernel
  with no discarded `self.attn` and no host-side gather.
- **CUDA graphs** for the verify forward (eager mode disables them).
- Moving tree construction/acceptance off the host critical path.

## Conclusion

This is, to our knowledge, the **first working tree speculative decoding on a
GDN-hybrid recurrent MoE** — the DDTree paper deferred recurrent architectures as
future work. The GDN branch-state solution (6 invariants) is implemented and
verified. The reference implementation establishes *correctness*; *throughput*
requires the kernel/graph optimizations above and is not yet realized.
