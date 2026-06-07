# GDN prefix-caching + DFlash spec-decode — finding chain (newest first)

## Bugfix session — diagnosis: mode-switch, not a cache bug (2026-06-07)

P1 divergence ROOT CAUSE: APC toggles the GDN compute MODE (vLLM forces 'none' w/o APC,
'align' with APC). The gate "APC-on vs APC-off" = align vs none = two different FP recurrence
paths -> greedy near-tie flips (16 vs 17), block-size-INVARIANT (64/128/2048 identical) ->
Candidates A & B (cache/block bugs) BOTH ELIMINATED. The bitwise gate is confounded; the real
check is cold==warm WITHIN align mode, which needs decoupling mode from the APC flag (next
session). No fix shipped (no candidate confirmed). See GDN_APC_BUGFIX.md.

## P1 — RESULT: launches, but bitwise gate FAILS → STOP (2026-06-07)

align+APC+spec **launches** (no #39809 crash, graphs capture). But T=0 APC-on vs off
**diverges 14/20 prompts** (block-boundary positions). Cache dtype is already bf16 ("auto"),
so NOT a dtype fix -> **wrong-state bug** on this pre-nightly-fix base (Bug 1 absent).
Per directive P1c: STOPPED, do not ship divergent output. Tier-A NOT reached on the overlay
base; needs the nightly #39809 .py port (future). Recovery point clean (config-only, gated).

## P0 — base-state investigation (2026-06-07)

**Workflow reality:** our base is the prebuilt overlay image `vllm-dflash-thor:fa-native`
= **vLLM 0.20.0.dev0+dflash** (custom DFlash patch). We do NOT compile vLLM; we COPY .py
into an overlay. The directive's "fork from nightly" is replaced by: verify what THIS base
has, and port nightly .py/Triton fixes into the overlay where needed (compiled .so can't be
changed, but the mamba state kernels are Triton/JIT, so they ARE overlay-portable).

**Verified against the actual image:**
- **#40738 accept-offset: PRESENT** — `worker/mamba_utils.py:postprocess_mamba` has the
  `num_accepted_tokens` offset logic (L214/247/251/272). (Scope is ngram upstream; the
  DFlash-path port is still P2.)
- **#39809 Bug 1: ABSENT** — `attention/backends/mamba_attn.py:110-121` sizes the all-mode
  buffer with a comment **"Speculative decoding not supported with prefix caching"** — i.e.
  this base predates the nightly Bug-1 fix. (All-mode path; align-mode is a different branch.)
- **suffix decoding (`method="suffix"`) UNUSABLE** — needs `arctic_inference`, **not installed**
  in the image. (vLLM's own `ngram_proposer.py` is present and needs no extra dep.)
- TP=1 confirmed (single-GPU Thor).

**Implication:**
- **Tier-A uses ALIGN mode**, which the directive says sidesteps Bug 3 and (per its text)
  the cudagraph Bugs 1/2 are nightly-fixed. The cheapest correct test is to **try the
  align-mode + `--enable-prefix-caching` + `--mamba-block-size 128` launch on THIS image**
  and run the **bitwise T=0 APC-on==off gate**. If it launches + bitwise-identical, Tier-A
  is reached with config only. If it crashes (#39809), port the specific nightly .py fix.
- **Stage B (CPU suffix drafter) blocker:** no native two-drafter ("pick better of DFlash
  vs suffix") path in vLLM, and the suffix method needs an absent package. A true hybrid is
  a from-scratch proposer integration (large, fail-loud). Documented; see designs/.
