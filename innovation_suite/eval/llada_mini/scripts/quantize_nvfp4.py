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

from transformers import AutoTokenizer
from datasets import load_dataset

# The official modeling_llada2_moe.py imports create_bidirectional_mask from
# transformers.masking_utils, absent in this image's transformers 4.57.3. Load via
# S2D2's self-contained modeling (proven to load these weights in the benchmark).
print(f"[quant] loading {MODEL} via S2D2 modeling", flush=True)
sys.path.insert(0, "/work/S2D2/LLaDA2")
from configuration_llada2_moe import LLaDA2MoeConfig
from modeling_llada2_moe_cache import LLaDA2MoeModelLM
cfg = LLaDA2MoeConfig.from_pretrained(MODEL)
model = LLaDA2MoeModelLM.from_pretrained(MODEL, config=cfg, torch_dtype="auto", device_map="cuda")
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
from datasets import Dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [it["text"].strip() for it in ds if len(it["text"].strip()) > 80][:64]
# pre-tokenize to input_ids ONLY (no attention_mask): LLaDA2's diffusion forward rejects
# the standard 2D (B,S) mask and builds its own bidirectional block mask when mask is None.
enc = [tokenizer(t, max_length=256, truncation=True)["input_ids"] for t in texts]
samples = Dataset.from_dict({"input_ids": enc})
print(f"[quant] {len(samples)} calibration samples (input_ids only, no attn mask)", flush=True)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
print("[quant] running oneshot NVFP4 ...", flush=True)
# pass processor=tokenizer directly: oneshot can't re-init a trust_remote_code processor
oneshot(model=model, dataset=samples, recipe=recipe, processor=tokenizer,
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
