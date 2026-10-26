# Speculative decoding, caching and long context on a single Jetson AGX Thor

**An experiment monorepo of hypotheses, verdicts and evidence.**

This repository collects inference experiments run in June 2026 on one NVIDIA Jetson AGX Thor (`sm_110a`, CUDA 13,
aarch64, 128 GB unified memory). The models are:
- three hybrid Gated-DeltaNet (GDN) mixture-of-experts models: Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.5-122B-A10B;
- one block-diffusion MoE: LLaDA2.1-mini.

Drafting uses DFlash block-diffusion speculative decoding (arXiv:2602.06036), served by a vLLM `0.20.0.dev0` fork.

Each study states a hypothesis, then reports a verdict: supported, refuted, or inconclusive. Each verdict links to
its evidence. Negative results are reported with the same weight as positive ones; several headline findings here
are negatives.

## Summary of findings

| # | Hypothesis | Verdict | Key measurement |
|---|---|---|---|
| H1 | Tree speculative decoding (DDTree) can be made correct on a recurrent GDN hybrid | **Supported** | 0 invariant violations; tree width 1 byte-identical to linear; branch kernel 0.0 diff against a sequential reference |
| H2 | A verification tree beats linear DFlash on a strong drafter | **Refuted** | Tree + typical 66.1 tok/s vs linear + typical 90.7; beating linear would need τ > 9.4 against a ceiling of ~7.5 |
| H3 | Relaxing the acceptance rule (typical acceptance) is worth more than tree breadth | **Supported** | +26% / +27% tok/s at T = 0.3 on 122B-A10B / 27B; byte-identical at T = 0 |
| H4 | Most training-free spec-decode optimizations still have headroom on this stack | **Refuted** | Of eleven candidate stages, one landed. The rest were already fused, provably moot for a factorized greedy draft, or irrelevant on unified memory |
| H5 | Prefix caching can coexist with DFlash on a recurrent hybrid without corrupting state | **Supported** | Cold == warm bitwise over an 11.8K-token, 4-turn agentic trace; 1.66× end to end |
| H6 | Speculative decoding can reproduce the base model's greedy tokens bitwise | **Refuted** (as a general property) | The base and DFlash-without-cache already diverge on near-ties in the verify forward |
| H7 | Inference-time RoPE manipulation improves long context beyond the native 262K on a GDN hybrid | **Refuted** | Far-region perplexity at 1M: standard RoPE 5.03, LongRoPE-approx 5.10, DroPE 7.07 |
| H8 | DroPE retrieves a needle at 1M tokens | **Supported, not specific to DroPE** | 14/15 retrievals at 1M; no standard-RoPE baseline was run, and the perplexity results suggest it would also retrieve |
| H9 | vLLM can serve LLaDA2.1-mini block diffusion faster than raw Transformers | **Supported** | 90.7 tok/s vs a 64.9 tok/s KV-cached floor (1.40×) |
| H10 | NVFP4 speeds up concurrency-1 block-diffusion decode | **Refuted** | 3.4× smaller, but under 1 tok/s: the FP4 grouped GEMM starves at ~32 tokens per forward |

## Studies

### 1. Tree speculative decoding on a recurrent hybrid (H1, H2)

**Question.** DDTree (arXiv:2604.12989) and CaDDTree (arXiv:2606.01813) evaluate pure-attention models and defer
recurrent and hybrid architectures to future work. A recurrent layer's state evolves sequentially, so it cannot be
shared naively across the branches of a verification tree.

**Method.** Six branch-state invariants for GDN layers:
- one state slot per tree node;
- branches seeded from a copy of the accepted state;
- the canonical state never written during verify;
- branch identity threaded to the GDN forward;
- atomic promotion of the accepted leaf's state;
- tree width 1 byte-identical to linear decoding.

The existing fused delta-rule kernel supports this without new CUDA. Each node is a one-token sequence seeded from
its parent's slot, batched per tree depth.

**Results** (Qwen3.6-35B-A3B NVFP4, 12-token drafts, CUDA graphs, T = 0.3):

| Configuration | τ | tok/s |
|---|---:|---:|
| Linear DFlash, rejection sampling (baseline) | 5.84 | 81.97 |
| Tree, typical acceptance | 6.35 | 66.1 |
| **Linear DFlash, typical acceptance** | **6.84** | **90.71** |

- **Correctness held.** 0 invariant violations, and 273/273 canonical-state checks untouched.
- **The tree got 3.6× faster (15.4 → 56.2 tok/s at width 2)** after three fixes: always keeping the full top-1
  chain (the spine) in the tree, one batched GDN launch per branch, and CUDA graphs.
- **Depth beats breadth for a strong drafter.** At the feasible budget (13 nodes, capped by the verify-token
  count), the tree reduces to the linear chain plus per-step overhead. An offline simulation bounds breadth's gain
  at +5% to +18% τ even with unlimited budget.
- **A tree-structured WY kernel would not help:** O(B²d), and at B = 13 it is latency-bound on single-tile GEMMs.

**Conclusion.** Tree speculation on a GDN hybrid is correct and implementable; to our knowledge this is the first
such implementation. It does not pay against a drafter this strong. Two independent efforts reached the same result
(a pure-CUDA Gemma-4 server, and CaDDTree in vLLM).

Evidence: [`docs/ddtree/`](docs/ddtree/README.md), [`benchmark_results/`](benchmark_results/),
[`tests/`](tests/), code in [`src/`](src/) and [`patches/`](patches/).

### 2. Training-free optimizations for linear DFlash (H3, H4)

**Typical acceptance** (Medusa-style) accepts draft token *i* if the target assigns it probability ≥ ε (ε = 0.09).
With ε = 0 it reduces exactly to greedy, so T = 0 output is byte-identical. It is enabled with
`DFLASH_ACCEPT_EPS`.

Production serving, 2-run medians over 4 coding tasks
([`no_train_suite/benchmarks/production_3model.md`](no_train_suite/benchmarks/production_3model.md)):

| Model | Baseline, T = 0.3 | Typical, T = 0.3 | Δ tok/s | Δ τ |
|---|---:|---:|---:|---:|
| Qwen3.6-27B (dense) | 43.0 (τ 6.00) | **54.8** (τ 7.66) | **+27%** | +28% |
| Qwen3.5-122B-A10B | 41.4 (τ 4.75) | **52.1** (τ 6.00) | **+26%** | +26% |
| Qwen3.6-35B-A3B | 127.5 (τ 6.31) | 137.8 (τ 6.56) | +8% (within noise) | +4% |

The gain concentrates where rejection sampling loses the most acceptance at temperature. Typical acceptance is a
lossy relaxation and applies only at T > 0.

**The rest of the suite was a negative-heavy audit** ([`no_train_suite/SUITE_SUMMARY.md`](no_train_suite/SUITE_SUMMARY.md)):
- **Already done.** The verify forward was already fused, and no FP4 dequantization fallback exists.
- **Moot on unified memory.** Expert prefetch has nothing to prefetch when weights are resident.
- **Provably moot.** Block verification, positional temperature, fat-chain drafting and top-p cannot improve on
  exact match for a deterministic factorized draft
  ([`docs/ddtree/IMPLEMENTATION_NOTES.md`](docs/ddtree/IMPLEMENTATION_NOTES.md)).
- **Bounded.** Adaptive draft length is limited by a tok/s plateau between draft lengths 4 and 12.

### 3. Prefix caching under speculative decoding on a hybrid (H5, H6)

**Question.** Can vLLM's automatic prefix caching restore GDN recurrent state exactly while DFlash speculation is
active?

**Results** ([`gdn_apc/`](gdn_apc/GDN_APC_SUMMARY.md)):
- **The cache is exact.** Cold and warm runs are bitwise-identical over an 11.8K-token, 4-turn agentic trace
  (20/20 prompts).
- **End to end:** 20.25 s against 33.71 s for the base model, 1.66×; caching adds 1.33× on top of DFlash.

**A methodological correction.** The obvious test, cache on against cache off, is confounded. Enabling the cache
switches the recurrent-state mode (`none` to `align`), which changes numerics without any corruption. The valid
test compares cold and warm runs within one mode.

**H6.** Bitwise agreement with the base model is not achievable, and not because of the cache. The base and DFlash
without any cache already diverge: the verify forward's near-tie numerics cascade. Every divergence observed
produced a valid greedy continuation, not corrupted state.

### 4. Long context past the native window (H7, H8)

**Setup.** Qwen3.6-35B-A3B; native context 262,144 tokens. Three inference-time position schemes, none
recalibrated:
- standard RoPE;
- a LongRoPE approximation (non-uniform rescaling);
- DroPE (positions beyond native set to identity rotation). DroPE was implemented as a new vLLM `rope_type`, about
  67 lines, cudagraph-safe and bitwise-equal to YaRN within native range.

Far-region perplexity on GovReport, tokens beyond 262K
([`innovation_suite/eval/results/`](innovation_suite/eval/results/)):

| Context | Standard RoPE | LongRoPE-approx | DroPE |
|---|---:|---:|---:|
| 288K (1.1× native) | **4.18** | 4.58 | 5.23 |
| 512K (2×) | **5.07** | 5.16 | 7.11 |
| 1M (4×) | **5.03** | 5.10 | 7.07 |

LongPPL on key tokens below 128K is about 1.252 in every condition. That is expected under causal attention, and it
validates the pipeline.

**Conclusion.** Standard RoPE wins at every scale and does not collapse at 4× native. Removing positional signal
(DroPE) is clearly worse; rescaling it (LongRoPE-approx) is close but not better.

**Interpretation, hypothesis-generating.** The GDN recurrent layers appear to carry long-range signal, which makes
the hybrid robust to out-of-distribution RoPE in its attention layers.

**DroPE needle retrieval at 1M** scored 14/15: 5/5, 4/5 and 5/5 at depths 5%, 50% and 95%. N = 5 per depth is a
scouting sample (Wilson 95% interval at the 95% depth: [0.57, 1.0]). No standard-RoPE baseline was run, and given
the perplexity table the result should not be read as DroPE-specific.

### 5. Serving a block-diffusion LLM (H9, H10)

**Setup.** LLaDA2.1-mini: 16B MoE, about 1.4B active, 256 experts with top-8, 32-token denoising blocks.

| Stack | tok/s |
|---|---:|
| Raw Transformers, BF16, no cache | 38.9 |
| Raw Transformers, BF16, KV cache (floor) | 64.9 |
| Speculative self-drafting (S2D2) | 34.3–39.7 (no gain) |
| vLLM diffusion plugin on the DFlash fork, eager, Triton attention | 72.1 |
| + FlashInfer non-causal attention | 73.3 |
| **+ commit threshold 0.55** | **90.7** (1.40× the floor; output coherent) |
| + commit threshold 0.40 | 104.2 (token-reordering artifacts) |

**What made it work.**
- Making vLLM generate at all required a gated 14-file port.
- The commit threshold is the dominant lever. Decoding is bound by the number of denoising steps, and per-token
  MoE cost falls 2.2× from 1 to 32 tokens per forward.

**What did not work.**
- **CUDA graphs** produce degenerate output with the diffusion plugin: a correctness incompatibility, not a
  hardware limit.
- **MoE autotuning** stalls at about 97% on `sm_110`, because the tuner has no per-config timeout.
- **NVFP4** loads after a loader patch at 10 GB against 32, but the CUTLASS FP4 grouped GEMM starves at batch ~32.
  It is a memory win only.
- **SGLang** is blocked on aarch64 by its prebuilt ABI.

Evidence: [`innovation_suite/eval/llada_mini/`](innovation_suite/eval/llada_mini/).

### 6. Upstream contribution audit

Six planned vLLM bugfixes were checked against upstream `main` before any pull request:
- four were already fixed or landed upstream;
- one, M-RoPE vision fallback, needs a full multimodal class;
- one, sparse-KV infrastructure, needs a port.

No redundant pull requests were opened. The single novel change, the DroPE `rope_type`, is on a personal fork
branch (`pr/drope-rope-type`) for human review. Given H7, its value claim is not supported.

Two environment findings:
- **FP8 KV fails on this stack under both attention backends:** a FlashInfer `kv_cache_sf` API mismatch and a
  Triton assertion.
- **Nemotron-3-Super-120B** ships on vLLM 0.16.2 with a 262K context, not 1M.

See [`innovation_suite/`](innovation_suite/INNOVATION_SUITE_SUMMARY.md).

## Methodology

- **Correctness gates on token IDs.** A lossless change must be byte-identical at T = 0 by real token-ID diff.
  Lossy changes are opt-in, apply only at T > 0, and are guarded at T = 0.
- **Valid comparisons.** A test must compare like with like. The cache study's on/off gate was replaced when it
  turned out to measure a mode switch.
- **Reporting.** Gates that could not run are marked not executed. Scouting sample sizes are labeled with their
  intervals.
- **Overlay images.** Changes are authored against byte-identical copies of a frozen base image's files and shipped
  as copy-only overlays ([`Dockerfile.ddtree`](Dockerfile.ddtree)). Pre-change originals are in
  [`src_original/`](src_original/), so every patch is self-contained.
- **One GPU job at a time.** Thor's memory is unified, so GPU work runs through a single-container guard
  ([`innovation_suite/tools/gpu_run.sh`](innovation_suite/tools/gpu_run.sh)): a flock mutex, graceful stop and a
  start watchdog.
- **Signed commits.** Commits from 2026-06-08 onward are GPG-signed with a DCO sign-off.

## Threats to validity

- **One device, one fork.** Results come from one Thor and one vLLM fork (`0.20.0.dev0+dflash`). Kernel-selection
  results may not transfer to other GPUs or vLLM versions.
- **Narrow throughput benchmarks.** They use 4–5 coding prompts at 128–400 tokens, with 2-run medians. Small
  differences, such as the 35B-A3B +8%, are within noise.
- **Typical acceptance changes the output distribution at T > 0.** It is a quality trade, measured here only for
  speed and acceptance, not for downstream task accuracy.
- **Long-context limits.** Perplexity uses one corpus. The 1M NIAH run has N = 5 per depth and no standard-RoPE
  control.

## Reproduction

The base image is `vllm-dflash-thor:fa-native` (vLLM `0.20.0.dev0+dflash`, built for `sm_110a`); the overlays add
the patched Python files without recompiling.

```bash
docker build -f Dockerfile.ddtree -t vllm-dflash-thor:ddtree .      # DDTree + typical acceptance overlay
# Linear DFlash with typical acceptance at T > 0:
DFLASH_ACCEPT_EPS=0.09 ...                                          # 0 = exact greedy (byte-identical)
# Tree speculation (correct, slower than linear on strong drafters):
DFLASH_TREE_WIDTH=2 DFLASH_GRAPHS=1 ...
python tests/test_gdn_tree_invariants.py     # inside the overlay container: the six GDN branch-state invariants
```

Each study directory lists its own serve scripts, configurations and raw result files.

## Repository layout

| Path | Contents |
|---|---|
| [`docs/ddtree/`](docs/ddtree/) | DDTree writeup, invariant-to-file plan, per-file change map, design notes and proofs |
| [`src/`](src/), [`src_original/`](src_original/), [`patches/`](patches/), [`image-src/`](image-src/), [`Dockerfile.ddtree`](Dockerfile.ddtree) | Modified vLLM files, their originals, diffs, and the overlay build |
| [`tests/`](tests/) | Tree invariant, branch-kernel and synthetic tests; benchmark harness |
| [`benchmark_results/`](benchmark_results/) | Tree vs linear, breadth vs depth, Tree-WY FLOP analysis |
| [`no_train_suite/`](no_train_suite/) | Training-free optimization audit and the 3-model typical-acceptance benchmark |
| [`gdn_apc/`](gdn_apc/) | Prefix caching under speculation: design, gate fix, results |
| [`innovation_suite/`](innovation_suite/) | Upstream audit, RFCs, long-context evaluation, LLaDA2.1-mini serving, GPU launch guard |

## License

Apache-2.0 ([`LICENSE`](LICENSE)). The modified files under `src/`, `src_original/`, `image-src/` and `patches/` are
derived from vLLM, which is Apache-2.0. Model weights are not distributed here.

## Citation

```bibtex
@misc{devaney2026thorspec,
  title  = {Speculative decoding, caching and long context on a single Jetson AGX Thor: an experiment monorepo},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/dflash-caddtree-integrations}
}
```

**References.**
- DFlash, arXiv:2602.06036.
- DDTree, arXiv:2604.12989.
- CaDDTree, arXiv:2606.01813.
- Medusa (Cai et al., 2024), for typical acceptance.
- LLaDA2.1 (inclusionAI).
- vLLM (Kwon et al., 2023).
