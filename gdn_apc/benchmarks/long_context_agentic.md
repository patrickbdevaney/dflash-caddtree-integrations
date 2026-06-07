# Long-context agentic: GDN prefix cache correctness over ~11.8k tokens, 4 turns

Real Hermes shared system prompt (~8.3k tokens: SOUL.md + CLAUDE_hermes_max.md + skills
snapshot) + 4-turn agentic coding conversation. DFlash spec-decode + APC (align, block 128),
CUDA graphs, max_model_len 16384, T=0. COLD pass populates the cache; WARM pass re-runs the
identical conversation (cached prefixes → hits).

| turn | prompt_tok | cold (s) | warm (s) | speedup |
|---|---:|---:|---:|---:|
| 1 | 11122 | 7.68 | 4.28 | 1.79× |
| 2 | 11349 | 4.61 | 4.60 | 1.00× |
| 3 | 11571 | 4.78 | 4.27 | 1.12× |
| 4 | 11794 | 3.23 | 3.24 | 1.00× |

**cold == warm bitwise: TRUE (0/4 turns diverge).** The GDN prefix cache restores recurrent
state bitwise-identically across an ~11.8k-token, 4-turn conversation → **KV cache is stable
and not corrupt over long context**. This is the core production-correctness result.

Speedup note: turn 1 shows 1.79× (the 11k shared system prefix is cached). Turns 2–4 are
~1.0× *because the COLD pass already cached the growing prefix within itself* — i.e. within a
single agentic session, each turn already reuses prior turns' cached KV/state instead of
recomputing the full growing prefix (the real benefit). The APC-on-vs-APC-off e2e comparison
(where off recomputes the full prefix every turn) is in production_parity.md.
