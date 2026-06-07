# P1 — GDN-APC launch + bitwise T=0 gate

## Launch: PASS (no #39809 crash)
align-mode + `--enable-prefix-caching` + `mamba_block_size=128` + DFlash spec-decode +
CUDA graphs **launched and ran to completion** on `vllm-dflash-thor:fa-native`
(v0.20.0.dev0+dflash). vLLM auto-selected align mode when APC is enabled. So the launch
half of Tier-A works with **config only** (no #39809 crash on this base).

## Bitwise T=0 gate: FAIL with default SSM cache dtype
APC-on vs APC-off (baseline) at T=0, 20-prompt seed set: **14/20 prompts diverge** (first
diffs near block boundaries — prompt0 pos72, prompt1 pos125, prompt2 pos90). This is the
wrong-state hazard: align-mode block caching save/restores GDN recurrent state lossily.

Remedy under test (directive P1c): set `mamba_ssm_cache_dtype=bfloat16` to match the FLA
chunk_gated_delta_rule h dtype (a fp32 cache diverges from the bf16 compute path). Result in
the next entry; if still divergent → wrong-state bug → STOP + report, do not ship.

## bf16 retest: dtype is NOT the cause → wrong-state bug → STOP

`mamba_cache_dtype` and `mamba_ssm_cache_dtype` both default to **`"auto"` = model dtype =
bf16** (MambaDType = Literal["auto","float32","float16"]; "bfloat16" is rejected by the
validator). So the original diverging P1 run **already used a bf16 SSM cache** — the
divergence is **not** a dtype mismatch. (Forcing float32 would diverge *more*, per the
research note.)

**Conclusion (directive P1c): genuine wrong-state divergence on this base.** This
`v0.20.0.dev0+dflash` image predates the nightly #39809 fixes (Bug 1 confirmed ABSENT), and
align-mode GDN-state save/restore under DFlash spec-decode is not bitwise-correct here
(14/20 prompts diverge at T=0, first diffs at block boundaries pos 72/90/125).

**STOPPING per the sacred bitwise gate — divergent output must NOT ship.** Tier-A is NOT
reached on the prebuilt overlay base. Path forward (deeper, fail-quiet, future session):
port the nightly #39809 Bug-1/Bug-2 .py fixes + the correct align-mode GDN-state restore
into the overlay, then re-run this exact bitwise gate. Recovery point clean: APC was
config-only via the gated `DFLASH_APC` harness flag (default off); no image/code shipped.
