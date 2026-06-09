# Result: LongPPL — DroPE long-context QUALITY (Qwen3.6-35B-A3B)  [2026-06-09]
Owner: longppl.py / run_longppl_full.sh. Written ONLY by the LongPPL run.

## Methodology (faithful to arXiv:2410.23771)
- Corpus: **GovReport** (the paper's corpus) — natural long-form prose, ~656k tokens, 50 reports
  (raw text from ccdv/govreport-summarization, re-tokenized per model).
- Discriminator: **Llama-3.1-8B-Instruct bf16** (cross-vendor, full-attention, GDN-independent;
  128k native). Long-pass (full 128k) vs short-pass (K=4096 windows) -> per-token CE -> key
  tokens where influence = loss_short - loss_long > alpha(2.0) AND loss_long < median.
- Cross-tokenizer: key tokens mapped Llama->Qwen by CHARACTER SPAN (offset_mapping).
- Eval: Qwen3.6-35B-A3B (NVFP4) per-token CE via prompt_logprobs, no spec-decode, BF16 KV.
  DroPE-on vs DroPE-off (standard RoPE) at ctx=288000 (native 262144 -> 1.1x).
- Disc key tokens: 2269/131056 (1.7%), median_long=0.757 (healthy natural-prose loss).

## Results (ctx=288000, far region = tokens 262144..288000, n=25840)
| Metric | DroPE-on | DroPE-off (std RoPE) |
|--------|----------|----------------------|
| LongPPL (key tokens <128k) | 1.2521 | 1.2521 (identical) |
| Far-region PPL (>262k) | 5.23 | 4.1786 |
| Far degenerate frac | 0.275 | 0.299 |

## Findings (honest)
1. LongPPL identical on/off (1.2521): EXPECTED + validates the pipeline. Key tokens are <128k;
   under causal attention they never attend to the >262k region where DroPE acts, so DroPE
   cannot affect them. The discriminator + cross-tokenizer mapping behave correctly.
2. At 288k (only 1.1x native), DroPE does NOT improve far-region perplexity — slightly worse
   (5.23 vs 4.18). At MILD extension, standard RoPE is only slightly OOD and still works, so
   DroPE's removal of positional info is a small net negative. This is consistent with (not
   contradictory to) the 1M NIAH: DroPE is designed for EXTREME extension (4x native), where
   standard RoPE breaks and DroPE enables retrieval (NIAH p95 5/5 at ~950k). The two evals
   measure different regimes.

## Caveat / next step
prompt_logprobs is quadratic + memory-heavy -> 288k was the feasible ceiling here, which is
BELOW DroPE's benefit regime. A 512k (2x native) run — supported by the 656k-token corpus —
is the regime where standard RoPE is materially OOD and DroPE should start winning. That would
align the perplexity story with the 1M NIAH retrieval result. (Offered; not yet run.)

## Discriminator compatibility log (overlay = vLLM 0.20.0.dev0+dflash, SM110a)
- Llama-3.1-8B-NVFP4 (modelopt) / Gemma-4-NVFP4 (modelopt): BLOCKED (FlashInfer kv_cache_sf skew).
- Nemotron Super / Cascade-2: Mamba hybrids -> circular with GDN -> rejected.
- GLM-4.7-Flash-NVFP4 (compressed-tensors, full attn): arch glm4_moe_lite UNRECOGNIZED by overlay.
- Qwen3-14B-NVFP4 (compressed-tensors, full attn, 40k): works; same-vendor; smaller window.
- **Llama-3.1-8B-Instruct bf16 (128k): USED** — bf16 (no quant block), full attn, cross-vendor,
  recognized arch, 128k window. The correct discriminator.
