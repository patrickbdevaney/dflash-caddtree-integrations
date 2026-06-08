# Stage 4 — DroPE (262k→beyond) — RFC (>300 LOC, default-off flag)
Mechanism: flag `drope_beyond_native_context` (default False). seq_len>262144 on Qwen3.6 →
identity rotation on the 10 attention layers only; 30 GDN layers untouched (they carry
positional info recurrently). Insertion: the `_apply_rope` path / rope_scaling in qwen3_next
attention. Lossless gate: within 262k, DroPE on==off bitwise (identity fires only >262k).
Eval: LongPPL (PKU-ML/LongPPL, arXiv:2410.23771) on a 512k doc, rank DroPE vs YaRN variants;
confirm top-2 with RULER NIAH at {262k,512k}. Zero memory overhead (rotation computed, not
stored). STATUS: RFC — implementation is >300 LOC fail-quiet AND eval needs 262k–512k forward
passes (memory/time-heavy, multi-hour, OOM-risk on a single 35B at 512k). Not blind-built in
the overnight window. Design ready; needs a dedicated session with a 512k-capable config.
