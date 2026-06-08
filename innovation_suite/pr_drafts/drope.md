# [Feature] DroPE: drop RoPE beyond native context for GDN-hybrid models

Co-authored-by: Claude
Signed-off-by: Patrick Devaney <patrickbdevaney@gmail.com>

## Problem
Extending a GDN-hybrid (Qwen3.6-35B-A3B: 30 GDN + 10 attention layers) beyond its 262k
native context via YaRN applies out-of-distribution RoPE rotations on the attention layers,
degrading long-range recall. The 30 GDN layers carry positional information recurrently, so
the attention layers need less RoPE help at extreme range.

## Design
New flag `drope_beyond_native_context` (env `DFLASH_DROPE`, default OFF). In the attention
forward, rotate normally for positions < native; apply **identity rotation** (drop RoPE) for
positions >= native. GDN layers untouched. ~20 logic LOC in qwen3_next attention:
clone pre-rotation q/k, rotate, then `torch.where(positions>=native, pre, rotated)`. Below
native the `where` never fires → **bitwise-identical to baseline** (lossless within native).

## Alternatives considered
- YaRN (rope interpolation): works but OOD rotations beyond training context degrade recall.
- NoPE everywhere: discards positional info the attention layers DO use within native.
DroPE keeps native RoPE intact and only drops it where it's already OOD.

## Test plan
1. Lossless gate (correctness): DroPE on vs off within native (≤262k) — bitwise T=0 identical.
2. Recall (benefit): LongPPL proxy + RULER NIAH at 262k / MAX_VIABLE_CONTEXT, DroPE vs YaRN
   variants. Discriminator = nvidia/Llama-3.1-8B-Instruct-NVFP4 (pure Transformer, full GQA,
   architecturally independent from the GDN hybrid; paper-validated, arXiv:2410.23771).
   LongPPL = exp(mean CE over KEY_MASK positions); same KEY_MASK for all configs. Top-2
   confirmed via NIAH; Spearman(LongPPL,NIAH) reported.

## Affected models
GDN/Mamba hybrids extended beyond native context (Qwen3.5/3.6 GDN family). Nemotron is NoPE
by design → not applicable.

## LOC estimate
~20 logic LOC (excl. eval harness/tests). Under the RFC threshold → [Feature] PR, no RFC.

## Status (this session)
IMPLEMENTED (default-off). Within-native lossless gate: running. Beyond-native recall eval
(LongPPL+NIAH at 384k-512k, needs context-extension config) = dedicated session; MAX_VIABLE_
CONTEXT probe confirmed 262k is reachable on Thor (200k-tok prefill 135s, 1480 tok/s).
