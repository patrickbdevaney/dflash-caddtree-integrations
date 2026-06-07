# Final production benchmark (run ONCE at end of the entire suite)

After all optimization stages land, validate cumulative speed with the **real
vLLM serve + HTTP bench** (not the in-process micro-harness), per model.

## Models

| model | dir present? | serve script | notes |
|-------|--------------|--------------|-------|
| Qwen3.6-35B-A3B-NVFP4 (+DFlash draft) | ✅ | serve-35b.sh | primary |
| Qwen3.5-122B-A10B-NVFP4 | ✅ | serve-122b.sh | cutlass MoE + TRITON_ATTN + gpu_util 0.78 (marlin/flashinfer crash) |
| Qwen3.6-27B | ❌ **MISSING** | serve-27b.sh exists | **no model dir on disk** — cannot bench until provided |

**27B cannot be benchmarked** — only `~/Qwen3.6-35B-A3B-*` and
`~/Qwen3.5-122B-A10B-NVFP4` exist on disk. Will bench the two available and
report 27B as unavailable (needs the model directory).

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
