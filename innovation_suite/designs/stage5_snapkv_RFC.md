# Stage 5 — SnapKV on attention layers only — RFC (>300 LOC, default-off flag)
Hypothesis: GDN/Mamba-2 hybrids tolerate higher KV compression than pure Transformers (the
recurrent layers already compress background; attention layers are more uniformly retrieval-
focused). Scope: 10 attention layers only (GDN has no KV). FIRST check if vLLM ships a SnapKV
path to configure; else port minimal SnapKV (observation-window scoring + eviction). Flag
HYBRID_SNAPKV=1. Eval: LongPPL vs ratio {0.25,0.5,0.75,1.0} → find the knee; NIAH at knee +
one past. lossy_gate (no bitwise): hard gate = ZERO degenerate output at the knee. STATUS:
RFC — >300 LOC fail-quiet + 262k–512k eval (heavy). Design ready; dedicated session needed.
