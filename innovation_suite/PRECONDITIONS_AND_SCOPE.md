# CORRECTION (model located by user)
Nemotron IS present: `~/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (120B, 17 shards) + serve
script `~/serve-35b/serve-nemotron-120b.sh`. Verified config: max_position_embeddings=**262144
(262k, NOT 1M)**; rope_theta=10000 & rope_scaling=None (config carries RoPE params — the
"NoPE everywhere" claim is NOT confirmed by config; model code may not apply it — must verify,
do not assume); GQA 32Q/2KV head_dim128; arch NemotronHForCausalLM (nemotron_h hybrid). Serve
image = `vllm-thor:qwen35-latest` (different from the dflash overlay); serve script **omits
spec-decode** ("MTP omitted for stability") -> MTP smoke stages may be unstable. 120B memory
89-117GB -> standalone only, in-container memory gate. Stages at 512k-1M EXCEED the model's
262k config.

# Innovation Suite V2 — preconditions & honest scope (verified against the box)

## Precondition findings (verified, not assumed)
- **Reference model ABSENT.** No `Nemotron-3-Super-120B-A12B-NVFP4` and no `serve-nemotron.sh`
  on disk. The only Nemotron present is **`Nemotron-Cascade-2-30B-A3B-NVFP4`** — a *different*
  model (30B, not 120B). The directive's "Nemotron-3-Super" architecture facts (1M native,
  NoPE everywhere, 32Q/2KV, MTP) **cannot be assumed** for Cascade-2-30B. ⇒ **all Nemotron
  smoke stages are blocked as specified** (would need the actual model + its verified arch).
- **Host `nvidia-smi` query format unsupported** on Thor tegra (memory.used/total → N/A);
  GPU works in-container. Memory gate must be done in-container.
- **Extreme-context stages (4 DroPE, 5 SnapKV, 6 InfLLM at 262k–1M):** each needle-in-haystack
  sweep at 262k–1M on a 35B model is a many-hour, memory-heavy run; the four-length × 50-needle
  sweeps are a multi-week research program, not a single session. These are scoped as
  design-doc + (where feasible) a single bounded measurement, not full sweeps.

## What this session executes (tractable, high-value, shippable)
- **Stage 0:** correctness harness (already have gen.py + diffids + cold==warm + recall
  scaffolding) + Qwen baseline (baseline.json from prior work).
- **Stage 1: FP8 KV `--calculate-kv-scales` fix** — the "fastest merge" real bug. Reproduce
  the hybrid corruption, validate the scale=1.0 fix bitwise vs bf16, patch the hybrid guard.
  This is the one clearly-scoped, gateable, shippable deliverable.

## Honestly deferred (with reasons)
- Stages 2/3 Nemotron smoke: blocked (model absent). Qwen-side of 2/3 overlaps prior gdn-apc +
  accept-offset work (already done/diagnosed) — see gdn_apc/.
- Stages 4/5/6: large novel research at 262k–1M; design-docs + single bounded probes only.
- All "Nemotron-3-Super" claims: not verifiable here (wrong model on disk).

No fabricated results. Every claim is gated by a real run or marked "not measured / blocked".
