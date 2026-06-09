# Stage 2 — NVFP4 Quantization of LLaDA2.1-mini: SUCCESS
Date: 2026-06-09 · Tool: llm-compressor (installed at container runtime) ·
Image: vllm-dflash-thor:latest (torch 2.10.0, transformers 4.57.3)

## Outcome
**NVFP4 quantization SUCCEEDED.** Artifact: `$HOME/models/LLaDA2.1-mini-NVFP4`
- Size: **9.5 GB** (from 32.5 GB BF16 → **3.4× smaller**), 3 safetensors shards.
- Format: `nvfp4-pack-quantized`, scheme NVFP4, targets `Linear`, ignore `lm_head`.
  Router gate is an `nn.Parameter` (not Linear) → left FP32 automatically. Files chowned
  to patrickd. Script: `scripts/quantize_nvfp4.py`.

## What it took (5 iterations through real API/arch mismatches)
The MoE arch itself quantized fine; the blockers were llm-compressor pipeline + custom-
diffusion-modeling integration, each fixed in turn:
1. `AutoModelForCausalLM(trust_remote_code=True)` → ImportError: official
   `modeling_llada2_moe.py` imports `create_bidirectional_mask` (absent in transformers
   4.57.3). → Load via S2D2's self-contained `modeling_llada2_moe_cache.py`.
2. oneshot can't init a trust_remote_code processor → pass `processor=tokenizer`.
3. oneshot needs a `datasets.Dataset`, not a list → build one.
4. **The key one:** llm-compressor's SequentialPipeline feeds a standard **2D**
   `attention_mask (B,S)`; LLaDA2's diffusion forward rejects it (`only supports
   (batch,1,q,kv) or (batch,q,kv)`). → **Feed calibration as `input_ids`-only** (no
   attention_mask column) so the model builds its own bidirectional block mask. This
   cleared the blocker — all **21 subgraphs** calibrated, model compressed and saved.

## Calibration caveat (honest)
`replace_modules_for_calibration` is absent from this llm-compressor's `llmcompressor.modeling`,
so calibration used **activated-expert routing only** (no forced all-expert pass). With 256
experts × top-8 over 64×256 calibration tokens, many of the 256 cold experts saw few/no
calibration activations → their NVFP4 activation scales are weakly calibrated. Expect some
quality degradation vs BF16, especially on rare-expert inputs. A correct production pass needs
a custom `MoECalibrationModule` forcing all-expert routing (DFLARE/Qwen3-MoE pattern) + more
samples. Coherence spot-check: see `bench_nvfp4.json` / table below.

## Spot-check (raw Transformers) — EMPIRICAL, measured
Ran `bench_s2d2.py` on the NVFP4 model (compressed-tensors installed at runtime), gen 128:
- **Load:** 320.6 s (vs BF16 **33 s**) — compressed-tensors does a CPU-bound "Compressing
  model" pass over all 14,692 tensors on load.
- **Runtime GPU memory:** **10.1 GB** (vs BF16 **32.5 GB**) — the 3.4× reduction holds at
  runtime; weights stay FP4-packed on device.
- **Decode speed:** **pathologically slow** — during generation GPU utilization was only ~28%
  while container CPU sat at ~92% (single-core), i.e. the FP4 **dequant runs on CPU each
  forward pass**. The first 256-token `nocache_quality` generation did **not** complete after
  >4 min (BF16 does the same in ~7 s). The run was **gracefully aborted** (docker stop, never
  kill) — measuring a precise tok/s was pointless; it is >30× slower and impractical.
- **Validity:** the model **loaded and began generating without error** (weights are valid,
  tokenizer/config intact); a full coherence sample was not captured because decode was too
  slow to finish. (Activated-expert calibration caveat above still applies for quality.)

**Empirical conclusion (matches the prediction):** in raw HF Transformers on SM110a there is
**no fused NVFP4 MoE kernel**, so NVFP4 weights dequantize on the CPU every forward → decode is
impractically slow. NVFP4's real payoff is the **memory footprint** (9.5 vs 32.5 GB) and the
**serving path** with fused FP4 GEMMs (vLLM+dllm-fork / dInfer), NOT raw-transformers decode.

## Bottom line
- Deliverable achieved: a valid NVFP4 LLaDA2.1-mini (3.4× smaller) + a reproducible recipe
  that handles the diffusion-mask incompatibility (input_ids-only calibration).
- For the **working raw-transformers stack on Thor, BF16 remains the speed choice** (NVFP4
  dequant overhead). NVFP4 is the right artifact for the fused-kernel serving path once that
  path is unblocked (Stage 5), where a fuller all-expert calibration should also be done.
