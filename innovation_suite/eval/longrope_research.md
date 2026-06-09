# LongRoPE / LongRoPE2 research + plan (2026-06-09)

## Versions
- LongRoPE (arXiv:2402.13753, Microsoft): per-dimension non-uniform RoPE rescaling factors
  found by evolutionary search; extends context to 2M.
- LongRoPE2 (arXiv:2502.20082, ICML 2025): improves LongRoPE with (1) hypothesis that HIGH RoPE
  dims are undertrained -> OOD; (2) **evolutionary search guided by needle-driven perplexity**
  for the rescale factors; (3) **mixed-context-window TRAINING** (fine-tune weights, ~10B tokens)
  to adopt the rescaled RoPE while keeping short-context perf.

## Critical scope finding
LongRoPE2's quality comes from (2) a per-model SEARCH (no published Qwen3.6 factors) AND
(3) WEIGHT FINE-TUNING. Both are training-scale -> OUT OF SCOPE for this no-train program.
What is runnable INFERENCE-TIME (no search, no fine-tune) is an APPROXIMATION — analogous to
the inference-time DroPE we already tested (which did NOT help: 7.07@1M vs std-RoPE 5.03).

## vLLM support
vLLM HAS `rope_type="longrope"` (Phi3LongRoPEScaledRotaryEmbedding): fields
original_max_position_embeddings, short_factor[half], long_factor[half]. No implementation
needed — config only. YaRN also natively supported (rope_type="yarn").

## What I will run (faithful + feasible, inference-time, no train)
1. **YaRN** factor=4 (well-defined NTK-by-parts; the method LongRoPE improves on) — the clean
   inference-time-scaling baseline.
2. **LongRoPE step-function approximation** via rope_type=longrope: short_factor=[1.0]*128
   (within-native lossless); long_factor = step at the critical dimension. For Qwen3.6
   (theta=1e6, head_dim=256, native=262144), the critical dim where wavelength==native is i=99:
   dims 0..98 (well-trained high-freq) -> 1.0; dims 99..127 (undertrained low-freq) -> 4.0
   (interpolate to 1M target). This is the LongRoPE2 non-uniform idea WITHOUT the search/train.
   Config: longrope2_config.json. NOT the searched/finetuned LongRoPE2 — documented as approx.

## Comparison target (established baseline, do NOT re-run)
std-RoPE far-PPL: 5.07 @512k, 5.03 @1M (eval/results/longppl.md).

## Honest prior
Given std-RoPE already plateaus at ~5.0 (not broken) on this GDN hybrid — GDN recurrent layers
carry long context — inference-time scaling may not beat it (same architecture-self-sufficiency
that made DroPE fail). A clean negative is a real result. Searched+finetuned LongRoPE2 remains
the only predicted path to improvement (training, out of scope).
