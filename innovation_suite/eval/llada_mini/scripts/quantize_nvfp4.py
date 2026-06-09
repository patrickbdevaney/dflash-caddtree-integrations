"""
NVFP4 quantization of LLaDA2.1-mini (custom llada2_moe MoE) via llm-compressor.

Honest caveat baked in: llada2_moe has NO built-in MoECalibrationModule in
llm-compressor, so without all-expert routing many of the 256 experts get few/no
calibration tokens. We attempt `replace_modules_for_calibration` (generic path);
if unavailable for this arch it no-ops and we proceed with activated-expert
calibration (documented as a quality risk). Router gate is an nn.Parameter, so
targets="Linear" leaves it FP32 automatically. Falls back cleanly on any error.
"""
import os, sys, getpass, traceback
import torch

assert getpass.getuser() in ("patrickd", "root")
MODEL = os.environ.get("SRC_MODEL", "/models/LLaDA2.1-mini")
SAVE_DIR = os.environ.get("DST_MODEL", "/models/LLaDA2.1-mini-NVFP4")
assert SAVE_DIR.startswith("/models"), SAVE_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print(f"[quant] loading {MODEL}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# all-expert calibration if llm-compressor exposes a generic linearizer
try:
    from llmcompressor.modeling import replace_modules_for_calibration
    replace_modules_for_calibration(model)
    print("[quant] replace_modules_for_calibration applied", flush=True)
except Exception as e:
    print(f"[quant] replace_modules_for_calibration unavailable/failed: {type(e).__name__}: {e}", flush=True)
    print("[quant] proceeding with activated-expert calibration (quality risk on cold experts)", flush=True)

print("[quant] building calibration set (wikitext-2, 64x256)", flush=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
samples = []
for item in ds:
    t = item["text"].strip()
    if len(t) > 80:
        samples.append(tokenizer(t, return_tensors="pt", max_length=256, truncation=True))
        if len(samples) >= 64:
            break
print(f"[quant] {len(samples)} calibration samples", flush=True)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
print("[quant] running oneshot NVFP4 ...", flush=True)
oneshot(model=model, dataset=samples, recipe=recipe,
        max_seq_length=256, num_calibration_samples=len(samples))

print(f"[quant] saving -> {SAVE_DIR}", flush=True)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
# custom modeling files are needed for trust_remote_code load of the quantized model
import shutil
for f in os.listdir(MODEL):
    if f.endswith(".py"):
        shutil.copy(os.path.join(MODEL, f), os.path.join(SAVE_DIR, f))
print("[quant] DONE", flush=True)
