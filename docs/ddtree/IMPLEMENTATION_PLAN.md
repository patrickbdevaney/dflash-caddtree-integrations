# DDTree on GDN-hybrid MoE (Qwen3.6-35B-A3B) — Part 2 Implementation Plan (CONTRACT)

Authoring surface: `~/dflash-dev/image-src/*.py` (byte-identical to the running
`vllm-dflash-thor:fa-native` `/opt/venv/.../vllm/...`). All edits here; overlay image
`COPY`s these onto the same `/opt/venv` paths. Forward-port to `~/vllm` HEAD after the
image is green (Strategy 1).

Tree semantics (confirmed): **per-position top-W lattice** with factorized joint
`Q(Y_1:L)=∏ q_i(Y_i)` from DFlash's single parallel draft; DDTree best-first heap
(arXiv 2604.12989 Alg.1), node budget **B ≤ 22**; prefix-closed tree; ancestor-only
verification mask. No conditional re-expansion.

Path-length convention: `K = num_speculative_tokens = 12`. A tree "branch" = a
root→leaf path. Tree has up to B nodes total (not per branch). One GDN recurrent
state slot is needed **per tree node** (children read parent's state), plus the
canonical slot — so the per-sequence reservation is `B+1`, not `W`.

---
## Invariant → file → change map

### INVARIANT 1 — one state slot per live branch (→ per node)
**Already partially present:** the spec path reserves `num_spec+1` GDN state slots per
sequence and indexes them via `spec_state_indices_tensor` `[batch, num_spec+1]`.
- `model_executor/layers/mamba/abstract.py:53-58` — `num_speculative_blocks =
  num_speculative_tokens`. **Change:** `= max(num_speculative_tokens, tree_node_budget)`
  (B≤22). This bumps the physical per-sequence reservation in
  `v1/core/single_type_kv_cache_manager.py:913-953` (consumes `num_speculative_blocks`)
  and the spec metadata buffers.
- `v1/attention/backends/gdn_attn.py:107-150` — index/buffer sizing uses
  `(self.num_spec + 1)`. **Change:** size to `(node_budget + 1)`.
- `v1/attention/backends/gdn_attn.py:156-300+` `build()` — currently slices
  `block_table_tensor[spec_mask, :num_spec+1]` into a per-position **chain**.
  **Change:** produce a per-**node** slot index tensor `tree_state_indices [batch, B]`
  laid out in the tree's flatten order.

### INVARIANT 2 — all branch states init = copy of canonical
**Mechanism present:** `fused_sigmoid_gating.py:103-120` seeds the chain's initial state
from `ssm_state_indices[i_n, num_accepted-1]` (the prior round's accepted/canonical slot).
**Change:** each tree **root child** seeds from that same canonical slot; interior nodes
seed from their parent's slot. Implemented via a new `parent_state_indices` tensor (see
Inv 4). No "copy to all W" needed if roots all read the one canonical slot.

### INVARIANT 3 — kernel must NOT write in-place during tree verify
**Capability present:** `fused_recurrent_gated_delta_rule_fwd(..., inplace_final_state=
False)` (`fused_recurrent.py:178-252`) and the scatter-by-index writes in
`fused_sigmoid_gating.py:157-170`. The canonical slot is only *read*, never written, as
long as branch write-slots are disjoint from the canonical slot (guaranteed by the B+1
reservation: canonical slot index ∉ {node slots}).
- **Reference implementation (ship first, per directive):** `fused_sigmoid_gating.py` —
  drive **per root→leaf path** as an independent chain seeded from canonical, writing to
  that path's disjoint node slots. W (≤ leaf count) kernel calls. Shared ancestors are
  recomputed per path (redundant but correct; B≤22 so cost is trivial). Canonical
  untouched.
- **File:** `model_executor/layers/fla/ops/fused_sigmoid_gating.py` (primary spec
  kernel). `fused_recurrent.py` packed_decode only if a W=1 single-token decode path is
  hit. **No fused multi-branch kernel in v1** — future work (Part 6 limitations).

### INVARIANT 4 — branch identity threaded to GDN forward
**Re-targeted (confirmed):** route via `GDNAttentionMetadata`, NOT `ForwardContext`.
- `v1/attention/backends/gdn_attn.py:40-64` `GDNAttentionMetadata` — **add fields:**
  `tree_state_indices: Tensor|None` (per-node write slots, `[batch, B]`) and
  `parent_state_indices: Tensor|None` (per-node read/seed slots, `[batch, B]`;
  roots → canonical slot). Populate in `build()`.
- `model_executor/layers/mamba/gdn_linear_attn.py:~1120-1290` `forward_cuda` spec path
  (reads `spec_state_indices_tensor`, `num_accepted_tokens`; calls
  `fused_sigmoid_gating_delta_rule_update(...)` at ~L1255-1273). **Change:** when tree
  metadata present, drive the per-path kernel calls using `tree_state_indices` /
  `parent_state_indices`. When absent (W=1 / no tree): unchanged → linear path.

### INVARIANT 5 — atomic exact promotion on acceptance
**Mechanism present (linear):** align-mode copy spec
(`v1/worker/mamba_utils.py:299-370`, `get_temporal_copy_spec`) copies the accepted chain
slot → canonical block using `num_accepted_tokens-1` offset; invoked by
`postprocess_mamba` (`mamba_utils.py:222`) from `gpu_model_runner.py:1477`.
- **Change (mamba_utils.py):** promotion copies the **accepted-branch leaf node slot** →
  canonical, where the accepted leaf is the deepest accepted node on the winning path.
- **Change (gpu_model_runner.py:1456-1492):** linear `num_accepted = first(-1)` becomes
  tree path selection (see Acceptance below); derive accepted leaf slot + length; feed to
  `postprocess_mamba`. All other node slots are implicitly freed (overwritten next round).

### INVARIANT 6 — W=1 byte-identical to linear DFlash
W=1 ⇒ tree degenerates to one chain; `tree_state_indices` == today's
`block_table[:, :num_spec+1]`, `parent_state_indices[t]=slot[t-1]`, no ancestor mask
restriction. Code paths must hit the *unchanged* branches. Guarded by TEST 1.

---
## Tree proposal (Step 4d) — DFlashProposer / base proposer
- `v1/spec_decode/llm_base_proposer.py:474-479` (parallel-draft sample point):
  `sample_hidden_states → compute_logits → [batch*K, vocab]`. **Change:** `topk(W,-1)` →
  `[batch*K, W]`; run DDTree best-first heap over factorized `∏ q_i` with budget B≤22;
  emit `(tree_tokens, ancestor_mask, parent_node_idx, branch_id)`.
- `v1/spec_decode/dflash.py` — input-prep (`set_inputs_first_pass` L202-307) may need to
  carry tree positions; the draft itself stays one parallel forward (semantics unchanged).
- New helper module `v1/spec_decode/ddtree.py` for the heap + prefix-closed tree build +
  flatten (keeps proposer diffs small).

## Target-side verification (Step 4d/4e) — gpu_model_runner.py
The **target** verifies the tree; largest piece the original directive under-specified.
1. **Ancestor mask for the 10 full_attention layers — RESOLVED by spike.**
   No flash_attn arbitrary-mask support, but the image's **FlexAttention backend**
   (`v1/attention/backends/flex_attention.py`) supports swappable `logical_mask_mod` +
   `create_block_mask` + paged-KV (`get_paged_mask_mod`). GPU smoke test on Thor SM110
   (cap 11.0, torch 2.10) PASSED eager+compiled with a custom tree-ancestor mask_mod
   (compiled vs eager max diff 0.0156, bf16-normal).
   **Decision:** W>1 tree verification runs the target under `FLEX_ATTENTION` with a
   `tree_ancestor_mask_mod` (spec node i attends to context fully + spec node j iff j is
   ancestor of i or i==j). **W=1 routes through the ORIGINAL flash_attn linear path
   unchanged** (no Flex) → keeps TEST 1 byte-identical. Tree mode = serve with
   `--attention-backend FLEX_ATTENTION`; build the BlockMask per verify step from the
   parent map. Fallback if Flex perf is poor: split-attention (flash spec→context + eager
   spec→spec LSE-merge) — not needed unless benchmarks demand it.
2. **Tree-aware acceptance.** `RejectionSampler` (L287/588) + `num_accepted` logic
   (L1466-1492) are linear. **Change:** select the longest accepted root→leaf path
   (greedy/T=0 exact-match; T>0 typed acceptance along a path), output accepted tokens +
   accepted leaf → branch index → promotion (Inv 5). Feed tree node budget + parent map
   into `gdn_attn.build` via the `num_decode_draft_tokens` path (L~2280, 5558).

---
## Config knob
`tree_width W` / `node_budget B`: thread via `speculative_config` (new optional field,
default W=1) so serve flag `--speculative-config '{"method":"dflash",...,"tree_width":2}'`
selects it. W=1 = exact current behavior.

## Files to be modified (overlay COPY list → Dockerfile.ddtree)
1. `model_executor/layers/mamba/abstract.py`            (Inv1 slot count)
2. `v1/attention/backends/gdn_attn.py`                  (Inv1,2,4 metadata+build)
3. `model_executor/layers/mamba/gdn_linear_attn.py`     (Inv3,4 forward routing)
4. `model_executor/layers/fla/ops/fused_sigmoid_gating.py` (Inv3 per-path kernel calls)
5. `v1/worker/mamba_utils.py`                           (Inv5 promotion copy)
6. `v1/worker/gpu_model_runner.py`                      (verify mask + acceptance + build wiring)
7. `v1/spec_decode/llm_base_proposer.py`               (4d proposal)
8. `v1/spec_decode/dflash.py`                           (4d input prep, if needed)
9. `v1/spec_decode/ddtree.py`  (NEW)                    (heap + tree build)
(+`fused_recurrent.py` only if W=1 packed_decode path needs the same guard.)

## Test mapping (Part 3, built first, must pass on linear/W=1)
- TEST 1 W=1 identity → INV6   | TEST 2 branch isolation → INV1+2
- TEST 3 non-in-place → INV3   | TEST 4 promotion → INV5
- TEST 5 multi-round continuity| TEST 6 accepted-path == linear → INV6/correctness
File: `~/dflash-dev/tests/test_gdn_tree_invariants.py`. Run vs `vllm-dflash-thor:fa-native`
(forced W=1 / top-1) before any tree code.

## Empirical notes from Part 3 (harness probe + baseline run)
- In-process harness viable: `VLLM_ENABLE_V1_MULTIPROCESSING=0` → `UniProcExecutor`,
  worker in-process; kernel monkeypatch captures GDN state (360 calls/generate).
- Runtime spec layout confirmed: `ssm_state_indices [batch, 13]` (num_spec+1=13),
  `num_accepted_tokens [batch]`, `initial_state` per-layer pool `[~3803, 32, 128, 128]`,
  `inplace_final_state=True`. Per-GDN-layer pool distinguished by `data_ptr()`.
- **Align-mode already multi-buffers:** a sequence's GDN spec slots rotate through ~4
  block-ranges of 13 slots each across decode steps; revisiting a range reads back exactly
  the prior write (TEST4 52/52). So the per-sequence physical reservation already exceeds
  13; bumping `num_speculative_blocks` (abstract.py:54) from `num_spec` to the tree node
  budget B must preserve this rotation. Allocator change in 4a must size each buffer to
  B+1 slots, not disturb the multi-buffer rollback.
- All 6 baseline tests PASS at W=1 (`~/dflash-dev/tests/`, ref tokens saved).

## Known risks / honesty
- **Target ancestor mask under flash_attn is unsolved upstream here** — biggest risk;
  spike first (masked Triton verify for ≤22 tokens is the leading candidate).
- Per-path kernel calls recompute shared prefixes (acceptable at B≤22; fused kernel =
  future work).
- Forward-port to `~/vllm` HEAD spans the #41126 mamba refactor (different paths/classes);
  it is a separate, non-image-tested diff.
