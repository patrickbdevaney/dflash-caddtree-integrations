# Eval result isolation convention (NIAH / LongPPL / RULER never overwrite)

Each eval OWNS a disjoint set of paths. No eval ever writes another eval's paths.

| Eval | Result doc (this dir) | Attestation prefix | Status log | Overlay /out prefix |
|---|---|---|---|---|
| S-NIAH 1M | results/niah_1m.md | scores/sniah_* | niah_status.log | (stdout only) |
| LongPPL | results/longppl.md | scores/longppl_* | longppl_status.log | /out/longppl_* |
| RULER 1M | results/ruler_1m.md | scores/ruler_* | ruler_status.log | /out/ruler_* |

Rules:
- A run writes ONLY its own result doc + its own attestation prefix + its own status log + its
  own /out prefix. Never another eval's.
- Shared narrative docs (SUITE_NOTES.md, INNOVATION_SUITE_SUMMARY.md, EVAL_TODO.md) are
  APPEND/PREPEND-only — never overwrite a prior eval's section; add a new dated section.
- Result docs here are per-eval and frozen on completion; a re-run writes a NEW dated file
  (e.g. niah_1m_n20.md), it does NOT overwrite the prior result.
