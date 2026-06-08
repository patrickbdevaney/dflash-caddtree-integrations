# [RFC] SnapKV-style KV compression on hybrid attention layers

AI-assisted (Claude); human reviews/defends. Not a PR — design + scope discussion.

## Discovery (C6, this run)
grep of upstream vLLM: NO general pluggable KV-compression framework
(snapkv/h2o/streaming_llm/token_budget) — only model-specific sparse attention
(deepseek_v4 sparse_attn_compress_cutedsl). So SnapKV on a GDN-hybrid must be PORTED,
~200-400 LOC, not configured.

## Design
- Observation-window importance scoring + eviction, applied to ATTENTION layers ONLY.
- Mandatory layer-type guard: `if not isinstance(layer, AttentionLayer): skip` — GDN/Mamba
  layers have no KV cache to compress (recurrent state).
- Hypothesis: hybrids tolerate higher compression (recurrent layers absorb background;
  attention layers are more uniformly retrieval-focused).

## Eval plan (deferred — needs GPU)
LongPPL sweep {0.25,0.5,0.75,1.0} at MAX_VIABLE_CONTEXT (Llama-3.1-8B discriminator) →
knee; NIAH at knee; memory headroom per ratio; hard gate = zero degenerate output at knee.

## Ask
Is layer-scoped KV compression for hybrids in scope for vLLM core, or a plugin? CUDA-graph
implications (dynamic token selection → variable shapes) need design input.
