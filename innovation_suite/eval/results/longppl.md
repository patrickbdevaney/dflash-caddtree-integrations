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

---
## UPDATE 2026-06-09: full 288k / 512k / 1M sweep — DroPE does NOT help (negative result)

GovReport corpus (~1.2M tok), Llama-3.1-8B bf16 disc @128k (reused key-mask), Qwen3.6 DroPE
on/off, prompt_logprobs with max_num_batched_tokens=8192 (logits-chunk cap -> no OOM at 1M).

| Context | DroPE-on far-PPL | DroPE-off (std RoPE) far-PPL | far tokens |
|---------|------------------|------------------------------|-----------|
| 288k (1.1x) | 5.23 | 4.18 | 25,840 |
| 512k (2x)   | 7.11 | 5.07 | 249,840 |
| 1M (4x)     | 7.07 | 5.03 | 646,580 |
LongPPL (key tokens <128k) = ~1.252 for all (on==off; causal-insensitive -> pipeline validated).

### Conclusion (honest, negative)
1. Inference-time DroPE (NO recalibration) WORSENS long-context perplexity at every scale; the
   gap is large and stable (DroPE-on ~7.1 vs std-RoPE ~5.0 from 512k on). Both PLATEAU.
2. Standard RoPE does NOT catastrophically break at 4x native on this GDN-hybrid (plateaus ~5.0,
   ~26% degenerate = normal). The GDN recurrent layers carry long context, so the attention
   layers' OOD RoPE at 1M is non-catastrophic.
3. IMPLICATION FOR THE 1M NIAH: it was DroPE-ON only; NO standard-RoPE baseline was run. This
   PPL evidence strongly suggests std RoPE would ALSO retrieve at 1M (better PPL, no collapse),
   so the NIAH DroPE result is likely NOT DroPE-specific.

### PR IMPACT (important)
The DroPE [Feature] PR (pr/drope-rope-type) value claim ("improves long-context") is NOT
supported by this evidence and is in fact contradicted on the perplexity dimension. DO NOT
submit the DroPE feature claim as a quality improvement. The rope_type implementation + the
20/20 bitwise graph-safe gate remain valid as INFRASTRUCTURE, but reframe as an optional
extension mechanism, NOT a quality win. Required before any DroPE quality claim:
  (a) standard-RoPE (DroPE-off) S-NIAH @1M baseline — does std RoPE retrieve too? (Tier 1)
  (b) RECALIBRATED DroPE (literature: DroPE needs continued-pretraining to help; we ran
      zero-shot inference-time = the baseline that the paper says underperforms).
This LongPPL eval did its job: it caught a result retrieval-only would have misrepresented.
