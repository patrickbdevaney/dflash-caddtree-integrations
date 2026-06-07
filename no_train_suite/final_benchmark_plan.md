# Final production benchmark (run ONCE at end of the entire suite)

After all optimization stages land, validate cumulative speed with the **real
vLLM serve + HTTP bench** (not the in-process micro-harness), per model.

## Models

| model | serve script | model dir | draft dir | notes |
|-------|--------------|-----------|-----------|-------|
| Qwen3.6-35B-A3B-NVFP4 | serve-35b.sh | ~/Qwen3.6-35B-A3B-NVFP4 | ~/Qwen3.6-35B-A3B-DFlash | NUM_SPEC=12 |
| Qwen3.5-122B-A10B-NVFP4 | serve-122b.sh | ~/Qwen3.5-122B-A10B-NVFP4/resharded | ~/models/Qwen3.5-122B-A10B-DFlash | cutlass MoE + TRITON_ATTN + gpu_util 0.78 |
| Qwen3.6-27B-NVFP4 | serve-27b.sh | ~/models/Qwen3.6-27B-NVFP4 | ~/models/Qwen3.6-27B-DFlash | NUM_SPEC=15 (dense, rewards max spec) |
| (bonus) Qwen3.5-4B-NVFP4 | — | ~/models/Qwen3.5-4B-NVFP4 | ~/models/Qwen3.5-4B-DFlash | optional extra datapoint |

**All three target models are present** (27B + 122B draft found in `~/models/`).
Serve scripts already reference the correct paths.

## Method

For each model: serve with the **overlay image** `IMAGE=vllm-dflash-thor:ddtree`
(so the optimizations are present) + the relevant `DFLASH_*` flags, wait for ready,
run `bench.py`, record tok/s, tear down, drop_caches, next model.

## Temperature caveat (important for interpretation)

`bench.py` runs at **T=0**. The main *currently-landed* optimization (typical
acceptance) is a **T>0** effect — it reduces to exact greedy at T=0, so a T=0
bench shows it as a no-op. Therefore the final bench will be run at **both T=0
(validates any C_verify-reducing / lossless stages) and T>0 (validates typical
acceptance)**, so the table reflects where each win actually applies. C_verify-
reducing no-train stages (F/C/A+E), if landed, help at all temperatures incl T=0.
