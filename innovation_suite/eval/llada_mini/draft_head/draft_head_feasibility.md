# Draft-Head Speculative Decoding for / with LLaDA2.1-mini — Feasibility
Date: 2026-06-09 · Hardware: Jetson AGX Thor SM110a · Model: LLaDA2.1-mini (16B MoE)

## TL;DR (honest)
The directive framed the novel contribution as "a DFlash/DFLARE-style draft head whose
**TARGET** is a diffusion LLM." After grounding in current prior art (June 2026) and our
own Stage-4 measurements, the conclusion is two-sided:
- **Architecturally implementable, YES.** LLaDA2.1-mini's within-block bidirectional
  hidden states are a richer conditioning signal than causal AR states, and a DFlare-style
  layer-wise fusion head can consume them.
- **Speed payoff on this hardware: UNPROVEN and currently contraindicated.** Stage 4 showed
  that adding an AR-verification loop to LLaDA2.1-mini (S2D2 `ssd_policy`) made it *slower*
  (34–40 vs 61–65 tok/s for plain KV-cache), because Thor is forward-pass/MoE-GEMM-bound and
  the diffusion target already amortizes ~8 decoded tokens per forward. A *learned* draft
  head adds its own forward pass plus a target-verify forward per block; it only wins if the
  draft is much cheaper than the diffusion target's per-block cost AND acceptance is high.
  Neither is established here.

## Prior art (corrects the directive's premise)
The mature, high-value direction is the **inverse** of what the directive proposed, and it
is already this project's main line of work:
- **DFlash** (arXiv 2602.06036, Feb 2026): the **draft** is a block-diffusion model, the
  **target** is an autoregressive LLM. >2.5× over EAGLE-3. (This is exactly the DFlash head
  we run on Qwen3.x targets.)
- **DFlare** (arXiv 2606.02091, 2026-06-01): scales draft capacity via **layer-wise fusion**
  — each draft layer learns its own softmax-weighted combination of target hidden states,
  fₜ⁽ⁱ⁾ = RMSNorm(Σⱼ αⱼ⁽ⁱ⁾ hₜ⁽ʲ⁾), over **9** uniformly-selected target layers (DFlash used
  5). **7 draft layers, block size 16**, only D×T=63 extra scalar params. Reported **5.46×**
  e2e vs DFlash **5.05×** vs EAGLE-3 **2.35×** on Qwen3-8B (greedy). Trained on **800K→2.4M**
  samples (Nemotron/CodeAlpaca/Step-3.5-Flash-SFT), **~160 GPU-h/device on 32 GPUs**.
- **DART** (arXiv 2601.19278): diffusion-inspired draft predicting multiple masked future
  positions in parallel from target hidden states.
- For accelerating a **diffusion target** specifically, the established methods are
  **training-free self-speculation**: S2D2 (arXiv 2603.25702) and SimSD (arXiv 2606.02544).
  A *trained* draft head for a diffusion target is genuinely under-explored — but see TL;DR.

## If pursued anyway — architecture (DFlare-style, diffusion target)
Dims from LLaDA2.1-mini config: hidden 2048, 16 heads (4 KV, head_dim 128), 20 layers,
moe_intermediate 512, 8 active experts → draft FFN ≈ 8×512 = 4096. Block 16 (DFlare) or 32.
- **Fusion:** 7 draft layers, each a learned softmax combination of **9** target layers
  uniformly spaced in [2, 17] (of 20). The novel twist vs DFlash-on-AR: target hidden states
  here are **bidirectional within a block** — position i encodes all in-block positions, so
  the draft sees a globally-consistent view of what the block will become. Hypothesis: higher
  acceptance per draft forward. (Unverified — needs training + measurement.)
- **Verifier:** LLaDA2.1-mini itself at block_size→small (its native self-correction), OR a
  one-step denoise pass as the accept/reject critic (S2D2-style).
- Skeleton: `draft_head_architecture.py` (definition only; NOT trained).

## Training plan (out of scope this session)
- Data-gen: pairs (target bidirectional hidden states at the 9 fusion layers, committed
  block tokens) from full-denoise runs. ~500K–2.4M samples. At our measured ~40 tok/s
  static-baseline data-gen rate, 2.4M×16 tok ≈ 38M tok ≈ ~11 GPU-h just to generate — plus
  training (DFlare used ~160 GPU-h on Qwen3-8B). This is a multi-day dedicated job, **not**
  runnable in this session. Deferred.

## Recommendation
The evidence-backed, higher-EV path on Thor is the **existing** direction: use a
block-diffusion model (LLaDA2.x or the DFlash head) as the **drafter for a larger AR target**
(DFlash/DFlare, 5×). A trained draft head *for* the diffusion target should be gated on first
showing — in a cheap training-free probe — that any speculative verify can beat plain
KV-cache decode on this hardware. Stage 4 indicates it currently does not. Mark as research,
not a near-term speed lever.

## Sources
- https://arxiv.org/html/2602.06036v1 (DFlash) · https://arxiv.org/abs/2606.02091 (DFlare)
- https://arxiv.org/abs/2601.19278 (DART) · https://arxiv.org/abs/2603.25702 (S2D2)
- https://arxiv.org/html/2606.02544 (SimSD) · https://github.com/tencent/AngelSlim (DFLARE impl)
