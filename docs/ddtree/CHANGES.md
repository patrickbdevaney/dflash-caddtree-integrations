# Code change map

Authoring surface was `~/dflash-dev/image-src/` — copies byte-identical to the
files inside `vllm-dflash-thor:fa-native` at
`/opt/venv/lib/python3.12/site-packages/vllm/...`. The overlay image
(`Dockerfile.ddtree`) COPYs the changed/new files back onto those paths (no
C++/CUDA recompile). Full files are under `src/vllm/...`; pre-change originals
under `src_original/vllm/...`; unified diffs under `patches/`.

| File | vLLM path | New? | Diff lines | Role in DDTree |
|------|-----------|:----:|-----------:|----------------|
| `ddtree.py` | `v1/spec_decode/ddtree.py` | **new** | — | DDTree best-first tree builder over DFlash's per-position marginals (`DDTreeHeap`, `build_ddtree`, `ancestor_mask`). W=1 ⇒ linear chain. |
| `ddtree_state.py` | `v1/spec_decode/ddtree_state.py` | **new** | — | GDN branch recurrent-state helpers: seed branches from canonical, promote accepted leaf, build per-branch index tensors. The "branch-state solution." |
| `abstract.py` | `model_executor/layers/mamba/abstract.py` | mod | 48 | INV1: per-sequence GDN state-block reservation (`num_speculative_blocks`) → tree node budget (env-gated, W=1 no-op). |
| `gdn_attn.py` | `v1/attention/backends/gdn_attn.py` | mod | 143 | INV1/2/4: widen `spec_state_indices` buffer; build the per-node depth-batched tree schedule (`tree_flat_pos/node_slots/parent_slots`, `tree_parent_index`); stash `node_to_slot` on the drafter. |
| `gdn_linear_attn.py` | `model_executor/layers/mamba/gdn_linear_attn.py` | mod | 54 | INV3: D10 per-node depth-batched recurrence — each tree node seeds from its parent's slot (`num_accepted=2` trick), batched per BFS depth; canonical untouched. |
| `dflash.py` | `v1/spec_decode/dflash.py` | mod | 19 | Proposer state fields (`tree_width`, `current_tree`, `current_node_to_slot`, `current_context_lengths`). |
| `llm_base_proposer.py` | `v1/spec_decode/llm_base_proposer.py` | mod | 56 | Change 0: at the parallel-draft sample point, build the DDTree from the K marginals and return its non-root nodes as the flat draft proposal. |
| `gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` | mod | 145 | Change A (tree RoPE positions, decoupled from KV slot-mapping), Change B (attach tree structure to `SpecDecodeMetadata`), Change D (branch promotion + `_find_accepted_leaf`, `_get_gdn_state_pool`), and the D2 wiring (`builder.drafter`). |
| `rejection_sampler.py` | `v1/sample/rejection_sampler.py` | mod | 77 | Change C: tree-aware acceptance (`_tree_accept`) — walk the tree following the target's greedy after each node; emit accepted path + bonus; build `SamplerOutput` directly. |
| `qwen3_next.py` | `model_executor/models/qwen3_next.py` | mod | 144 | D11: eager combined-mask attention in `Qwen3NextAttention.forward` — each spec node attends to all context (gathered from paged KV) + its tree ancestors via one GQA `scaled_dot_product_attention`. Gated on tree mode; linear path byte-identical. |

## Flat-position ↔ tree-node convention (the key correctness fact)

The verify query has **`B` (=node budget+1) positions**: `flat pos i == tree node i`,
with **flat pos 0 = root = the bonus/previously-accepted token**. Draft nodes
`1..B-1` occupy flat pos `1..B-1`. Every module uses this identity mapping
(`gdn_attn` schedule, `qwen3_next` ancestor mask, runner positions, rejection
sampler acceptance). An early off-by-one (assuming pos `i` = node `i+1`) produced
incoherent output and zero acceptance; fixing it to the identity mapping yielded
coherent output and multi-token acceptance.

## Explored-but-not-shipped

`flex_attention.py` (`.view`→`.reshape`) was patched while attempting the
ancestor mask via vLLM's FlexAttention backend. That route is **blocked** on this
model: the GDN-hybrid forces a non-power-of-2 attention block size (1152) and
FlexAttention's Triton kernel requires power-of-2 `BLOCK_N`. It is **not** in the
final overlay; D11 uses the eager combined-mask path in `qwen3_next.py` instead.
