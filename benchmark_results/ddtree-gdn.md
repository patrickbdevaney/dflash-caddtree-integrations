# DDTree on GDN-hybrid MoE — Benchmark (Jetson AGX Thor SM110a)

Model: **Qwen3.6-35B-A3B-NVFP4** (GDN-hybrid MoE: 30 Gated-Delta-Net + 10
full-attention layers; 256 experts top-8) + DFlash draft (`num_speculative_tokens=12`).
Image: `vllm-dflash-thor:ddtree`. Hardware: Jetson AGX Thor SM110a (Blackwell),
128 GB unified LPDDR5x. Single sequence, T=0, 5 coding prompts × 128 tokens.
τ = mean accepted tokens/step. (eager = `enforce_eager=True`; graphs = CUDA graphs.)

## Optimization progression (W=2 DDTree)

| stage | mode | τ | steps | tok/s |
|-------|------|---:|---:|---:|
| bushy tree (best-first, no spine guarantee) | eager | 3.14 | 189 | 15.44 |
| **+ spine fix** (full top-1 chain always in tree) | eager | 5.48 | 104 | 22.51 |
| **+ GDN fusion** (single per-branch launch vs per-depth loop) | eager | 5.48 | 104 | 25.70 |
| **+ CUDA graphs** | graphs | 5.70 | 97 | **56.23** |

W=2 improved **3.6×** (15.4 → 56.2 tok/s) with **no τ regression** (τ tracks linear).

## Head-to-head (apples-to-apples, both CUDA graphs)

| Config | τ | steps | tok/s | vs linear |
|--------|---:|---:|---:|---:|
| **W=1 linear DFlash** | 5.75 | 102 | **78.0** | baseline |
| **W=2 DDTree (B=13)** | 5.70 | 97 | **56.23** | **0.72× (28% slower)** |

(eager baselines, for reference: W=1 ≈ 40–45 tok/s, W=2 ≈ 15–26.)

## The decisive finding

After the spine fix, **τ_tree (5.70) ≈ τ_linear (5.75)** — the tree no longer
wastes steps. But at **node budget B=13** the tree is the *spine-only chain*
(B = 12-deep spine + root = 13, **zero spare budget for branches**), so it is
effectively linear DFlash **plus** per-step tree overhead (0.10 vs 0.074 s/step:
per-branch GDN gather/scatter, the full-attn context gather + eager SDPA, host
tree build/accept). With equal τ and extra overhead, W=2 can only be *slower*.

**DDTree can only beat linear when τ_tree > τ_linear, which requires real
branches, which requires B ≫ K.** B is capped at K+1 = 13 by the runner's flat
`num_speculative_tokens=12` assumption. Lifting it (schedule >K tree tokens) is
the #42121-class scheduler rewrite — the path to an actual speedup, deferred.

Also: DFlash's linear draft is already strong (τ≈5.7/12 ≈ 48% acceptance);
tree breadth adds little when the draft is good and the budget is tight. DDTree
pays off for *weaker* drafts and *larger* budgets.

## What works / correctness

- End-to-end tree speculation on GDN-hybrid: per-position lattice → DDTree
  (spine + best-first branches) → single target forward with ancestor-masked
  verification (eager combined-mask SDPA) → tree acceptance → GDN branch-state
  promotion. Coherent, valid output.
- **Spine guarantee:** the full top-1 chain is always a branch ⇒ τ_tree ≥ τ_linear.
- **GDN INV3** 273/273 canonical-untouched, **273/273 node divergence** at W=2.
- **W=1 (tree_width=1) byte-identical** to linear DFlash (6/6 invariant tests).
- Single-launch per-branch GDN ≡ per-depth loop (τ unchanged 5.48 → 5.48).
- W=2-graphs vs W=1-graphs differ at greedy near-ties (eager-SDPA vs flash + graph
  numerics) — both valid greedy continuations.

## Optimizations applied (and why they helped / didn't)

1. **Spine fix** (ddtree.py) — τ 3.14→5.48. *The* algorithmic fix: stops the
   bushy tree from truncating depth below linear's chain.
2. **GDN per-branch fusion** (gdn_attn.py/gdn_linear_attn.py) — 5670 per-depth
   launches → 1 batched per-branch launch; eager tok/s 22.5→25.7; unblocks graphs.
3. **CUDA graphs** (static B=13 tree padding) — removes Python dispatch;
   tok/s 25.7→56.2. The single largest gain.
4. Overhead is no longer the gap; **τ is** — and τ is budget-bound (B≤13).
