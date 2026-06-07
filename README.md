# DDTree tree speculative decoding on GDN-hybrid MoE (DFlash / Qwen3.6-35B-A3B)

First working implementation of **tree speculative decoding on a Gated-Delta-Net
(GDN) hybrid recurrent MoE** — the `Qwen3.6-35B-A3B` family — on top of **DFlash**
block-diffusion drafting, running on **NVIDIA Jetson AGX Thor (SM110a)**.

The DDTree paper (arXiv:2604.12989) and CaDDTree (arXiv:2606.01813) evaluated only
pure-attention models and **explicitly deferred recurrent / hybrid architectures
as future work**, because a recurrent layer's state `S` evolves *sequentially* and
cannot be naively shared across the branches of a verification tree. This repo
implements and verifies the **GDN branch-state solution** that makes it possible,
and documents the full build + debugging process end-to-end.

> **Status (honest):** the algorithm is **implemented and verified correct**
> (coherent output, multi-token tree acceptance τ>1, W=1 byte-identical to linear
> DFlash). The **eager reference implementation is not yet faster** than linear
> (2.7× slower in eager mode) — net throughput needs fused kernels + CUDA graphs,
> documented under *Limitations / future work*. No speedup is claimed.

---

## 1. The idea

DFlash drafts `K` tokens in **one parallel forward** by predicting `K` masked
positions; it then keeps only the top-1 token per position and verifies a single
linear chain. But that one forward already produces `K` rich per-position marginal
distributions `q_1..q_K`. DDTree turns those marginals into a **tree** of candidate
continuations and verifies the whole tree in **one** target forward, accepting the
best matching root→leaf path — more accepted tokens per step, same draft cost.

Per-position **top-W lattice with factorized joint** `Q(Y_{1:L}) = ∏_i q_i(Y_i)`:
this is exactly what DFlash's parallel marginals provide, so DDTree's best-first
heap (Algorithm 1) applies directly — no conditional re-drafting.

### Why GDN makes this hard

The 30 GDN (linear-attention) layers maintain a fixed-size recurrent state that
updates sequentially per token: `S_new = α·S_old + β·(k⊗v)`. In a verification
**tree**, each branch is a *different* continuation of `S`. If two branches share
a state slot, the second write corrupts the first. Standard tree attention (for
the 10 full-attention layers) does not solve this — the recurrent state needs its
own branch-correct bookkeeping. That is the contribution here.

---

## 2. The six GDN branch-state invariants

| # | Invariant | How it's satisfied |
|---|-----------|--------------------|
| 1 | One state slot per tree **node** (children read parent's state), not per sequence | `abstract.py` reserves `node_budget+1` GDN state blocks/seq; `gdn_attn.py` lays out per-node slots |
| 2 | All branches initialised from a **copy** of the canonical (accepted) state at the fork | root node = canonical slot; depth-1 nodes seed from it |
| 3 | Recurrence must **not** write in place over canonical during tree verify | per-node depth-batched kernel seeds from the parent slot and writes the node slot; canonical only read (verified: 273/273 untouched) |
| 4 | Branch identity threaded to the GDN forward | via `GDNAttentionMetadata` (not `ForwardContext`): `tree_flat_pos/node_slots/parent_slots` |
| 5 | Atomic, exact state **promotion** on acceptance | accepted leaf's state copied to `slot[num_accepted-1]` so existing align-mode `postprocess_mamba` carries it across the buffer rotation |
| 6 | `tree_width=1` is **byte-identical** to linear DFlash | all tree code gated on `tree_width>1`; the 6-test suite passes byte-identical at W=1 |

**Key kernel result (no new CUDA needed for correctness):** the existing batched
`fused_sigmoid_gating_delta_rule_update` already supports tree branch-state. Each
tree node is processed as a 1-token "sequence" seeded from its parent's slot via a
`num_accepted=2` trick (`ssm_state_indices=[node_slot, parent_slot]`, T=1 → kernel
seeds slot[1]=parent, writes slot[0]=node), **batched per BFS depth** so every
parent is written before its children. Validated to **0.0 diff** vs a sequential
reference chain (`tests/test_branch_kernel.py`, and the in-situ W=2 state test).

---

## 3. Architecture / data flow

```
DFlash draft (1 parallel fwd)                 Target verify (1 fwd over the tree)
  K position marginals q_1..q_K                 30 GDN layers: per-node depth-batched
        │                                          recurrence (parent→child slots)
        ▼                                       10 full-attn layers: eager combined
  DDTreeHeap.build  ──►  prefix-closed tree T*     mask = [context(all) | ancestor]
   (ddtree.py)          B≤node_budget nodes               (qwen3_next.py)
        │                     │                              │
        ▼                     ▼                              ▼
  flat draft tokens     ancestor matrix              target logits per node
  (proposer Change 0)   + per-node GDN schedule              │
        │               (gdn_attn.py build)                  ▼
        ▼                                            tree-aware acceptance
  scheduler schedules B-1 draft tokens               (rejection_sampler _tree_accept):
        │                                            walk T* following target greedy
        ▼                                            after each node → accepted path
  runner: tree RoPE positions (depth-based),                 │
   decoupled from KV slot-mapping (Change A)                 ▼
                                                     promote accepted leaf's GDN
                                                     state → canonical (Change D)
```

**Flat-position convention (the one fact to remember):** the verify query has `B`
positions where **flat pos `i` == tree node `i`**, and **flat pos 0 = root = the
bonus/previously-accepted token**. Every module uses this identity mapping. (See
`CHANGES.md`; an early off-by-one here caused incoherent output + zero acceptance.)

---

## 4. Repository layout

```
README.md                     ← this file
CHANGES.md                    ← per-file change map + flat-pos convention
docs/IMPLEMENTATION_PLAN.md   ← the original contract / invariant→file plan
Dockerfile.ddtree            ← overlay build (FROM fa-native, COPY .py files)
src/vllm/...                  ← FULL modified + new files at real vLLM paths
src_original/vllm/...         ← pre-change originals (so diffs are self-contained)
patches/*.diff               ← unified git-style diffs (apply with `git apply`)
patches/ALL.combined.diff    ← all diffs concatenated
tests/                       ← invariant suite + fast synthetic + spike tests
benchmark_results/           ← ddtree-gdn.md (table + analysis) + raw log
```

The 10 changed/new files (8 modified + 2 new) are listed with roles in
`CHANGES.md`. New modules: `ddtree.py`, `ddtree_state.py`.

---

## 5. How it was built (process narrative)

This was developed against a **frozen** base image (`vllm-dflash-thor:fa-native`)
whose installed vLLM is **not reproducible from the `~/vllm` fork HEAD** (HEAD has
the #41126 mamba refactor the image predates; the image's file blobs are not in the
repo). So the strategy was: **author against byte-identical copies of the image's
own files** and ship an **overlay image** (`Dockerfile.ddtree`, COPY only — no
recompile). Forward-porting to the public fork HEAD is a separate diff.

Order of work and gates:

1. **Characterise** the image: GDN mixer (`gdn_linear_attn.py`), the spec GDN
   kernel (`fused_sigmoid_gating`), `GDNAttentionMetadata` (already has a *linear*
   multi-slot spec mechanism — `spec_state_indices`, `num_accepted_tokens`), the
   DFlash proposer (`llm_base_proposer.propose`, parallel drafting), and the
   runner. Found: **no tree-attention infra**; runner hardwires a flat
   `num_spec_tokens=12` draft → **node budget B≤13** without a scheduler rewrite.
2. **Test harness first**: in-process `vllm.LLM` (single-process), GDN state
   captured by monkeypatching the spec kernel. 6-test invariant suite established
   on linear DFlash (all pass; W=1 reference saved).
3. **Validate the two cores fast (no model load):** the per-node parent-seed
   kernel trick (0.0 diff) and the DDTree builder (W=1→linear, prefix-closed,
   budget-capped, correct ancestor mask).
4. **Implement** in order, rebuilding the overlay + gating after each:
   `4a` allocator → `4c` GDN metadata builder → `Change 0` proposal →
   `D10` GDN per-node depth-batched recurrence (W=2 state test: INV3 273/273
   canonical-untouched, 273/273 node divergence) → `A/B/C/D` positions / metadata
   / tree acceptance / promotion → `D11` ancestor-masked verify attention.
5. **D11 was the hard part.** FlexAttention (the obvious ancestor-mask route) is
   **framework-incompatible** here: the hybrid forces a non-power-of-2 attention
   block size (1152) and FlexAttention's Triton kernel requires power-of-2
   `BLOCK_N` (`arange's range must be a power of 2`). Pivoted to an **eager
   combined-mask SDPA** inside `Qwen3NextAttention.forward`: gather context K/V
   from the paged cache, concat the spec K/V, one GQA `scaled_dot_product_attention`
   with mask `[context=all | spec=ancestor]`.
6. **First W=2 run was incoherent** → diagnosed via instrumentation to an
   **off-by-one** in the flat-pos↔node mapping (query is `[root, node1..node12]`,
   13 positions, not 12). Fixing all four sites to the identity mapping →
   **coherent valid code + multi-token acceptance**.

---

## 6. Configuration

Development uses env vars (config-field wiring is future work):

| env | meaning | default |
|-----|---------|---------|
| `DFLASH_TREE_WIDTH` | top-W per position; `1` = linear DFlash | `1` |
| `DFLASH_NODE_BUDGET` | max GDN state slots reserved (tree node budget) | `num_speculative_tokens` |

Engine config (mirrors `serve-35b.sh`): `quantization=compressed-tensors`,
`kv_cache_dtype=auto` (BF16), `num_speculative_tokens=12`, `gpu_memory_utilization=0.78`,
`enforce_eager=True` (tree path is eager; CUDA graphs are future work),
`VLLM_ENABLE_V1_MULTIPROCESSING=0` for the in-process test harness.

---

## 7. Benchmark (Jetson AGX Thor SM110a, eager, T=0)

| Config | tok/s | vs linear |
|--------|------:|----------:|
| W=1 (linear DFlash) | **44.74** | baseline |
| W=2 (DDTree tree spec) | **16.56** | 0.37× (2.7× slower) |

Correct, coherent output; multi-token acceptance (paths of 4/2/5 tokens observed);
W=1 byte-identical to linear. **Slower** because the eager reference adds large
per-step overhead (per-depth GDN launches, a discarded `self.attn` + manual gather
+ eager SDPA in every full-attn layer, host-side tree build/accept). See
`benchmark_results/ddtree-gdn.md` for the full analysis.

---

## 8. Limitations / future work (to realize speedup)

- **Fused multi-branch GDN kernel** (one parent-indexed launch) to remove the
  per-depth Python loop and the redundant baseline call.
- **Triton ancestor-mask attention** (or fix FlexAttention's power-of-2 block
  constraint for hybrid models) so the full-attn verify is a single masked kernel
  with no discarded `self.attn` and no host gather.
- **CUDA graphs** for the verify forward (disabled by `enforce_eager`).
- Node budget capped at `B≤13` (the runner's flat `num_speculative_tokens`
  assumption); larger trees need a scheduler change.
- Batch size 1 (single-sequence DFlash decode) in the current tree path.
- Tree (eager) vs linear (flash) can differ at greedy near-ties (kernel numerics).

---

## 9. Reproduce

```bash
# Build the overlay image (no recompile; ~seconds)
docker build -f Dockerfile.ddtree -t vllm-dflash-thor:ddtree .

# W=1 invariant suite (must be byte-identical to linear)
docker run --rm --runtime nvidia --gpus all \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e DFLASH_TREE_WIDTH=1 \
  -v $MODEL:/model:ro -v $DRAFT:/drafter:ro -v $PWD/tests:/tests \
  --entrypoint /opt/venv/bin/python vllm-dflash-thor:ddtree \
  /tests/test_gdn_tree_invariants.py

# W=2 tree generation
... DFLASH_TREE_WIDTH=2 ... /tests/_w2_smoke.py
```

Apply the diffs to a matching vLLM tree:
```bash
git apply patches/ALL.combined.diff     # against src_original/ layout
```

## References
- DDTree — arXiv:2604.12989
- CaDDTree — arXiv:2606.01813
- DFlash — arXiv:2602.06036
