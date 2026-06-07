# Correct gate: cold==warm WITHIN align mode

run-twice within APC-on align (block 128, graphs, DFlash spec k=12):
  run1 = cold (cache miss, computes + populates GDN state cache)
  run2 = warm (exact-repeat prompts -> prefix cache HIT -> GDN state restored)

**TWICE_RESULT: BYTE_IDENTICAL (run1 == run2, 20/20 prompts).**

⇒ the prefix cache restores GDN recurrent state bitwise-correctly in align mode.
The earlier P1 "14/20 divergence" was the align-vs-none MODE switch (benign FP-order
difference), NOT a cache-state bug. CUDA graphs captured (FULL_AND_PIECEWISE). No crash.
