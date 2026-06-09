"""
Unified LLaDA2.1-mini benchmark — loads model ONCE, sweeps all decode configs.
Covers Stage 3 (baseline) + Stage 4 (S2D2/Fast-dLLM stack) in one model load.

generate_fn:
  nocache    -> static block-diffusion baseline (no KV cache)
  cached     -> KV-cache decode (Fast-dLLM-style prefix+block cache)
  ssd_policy -> S2D2 training-free self-speculation (uses cache + AR verify)

Run inside vllm-dflash-thor image with S2D2/LLaDA2 on sys.path and the model mounted.
"""
import os, sys, time, json, getpass, argparse
import torch

assert getpass.getuser() in ("patrickd", "root"), f"user={getpass.getuser()}"

S2D2_LLADA = os.environ.get("S2D2_LLADA", "/work/S2D2/LLaDA2")
sys.path.insert(0, S2D2_LLADA)
from generate_utils import (generate, generate_cached, generate_ssd_policy,
                            load_model_and_tokenizer)  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL_TO_USE", "/models/LLaDA2.1-mini"))
ap.add_argument("--gen_length", type=int, default=256)
ap.add_argument("--block_length", type=int, default=32)
ap.add_argument("--out", default="/out/bench_results.json")
ap.add_argument("--smoke", action="store_true", help="one tiny config only")
args = ap.parse_args()

print(f"[bench] model={args.model} gen_length={args.gen_length} block={args.block_length}", flush=True)
t0 = time.time()
model, tok = load_model_and_tokenizer(args.model, dtype_str="bfloat16", device_map="auto")
print(f"[bench] loaded in {time.time()-t0:.1f}s", flush=True)
try:
    print(f"[bench] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
except Exception:
    pass

PROMPTS = [
    "Write a Python function that finds the longest palindromic substring.",
    "Implement a binary search tree in Python with insert, search, and delete methods.",
    "Write a Python function to solve the coin change problem using dynamic programming.",
]

# (label, fn, kwargs)  — threshold/editing_threshold per LLaDA2.1 quality/speed modes
COMMON = dict(block_length=args.block_length, gen_length=args.gen_length,
              temperature=0.0, top_p=None, top_k=None, eos_early_stop=True,
              eos_id=156892, mask_id=156895, num_to_transfer=1, return_stats=True)

CONFIGS = [
    ("nocache_quality",  generate,            dict(threshold=0.7, editing_threshold=0.5, max_post_steps=16)),
    ("nocache_speed",    generate,            dict(threshold=0.5, editing_threshold=0.0, max_post_steps=0)),
    ("cached_quality",   generate_cached,     dict(threshold=0.7, editing_threshold=0.5, max_post_steps=16)),
    ("cached_speed",     generate_cached,     dict(threshold=0.5, editing_threshold=0.0, max_post_steps=0)),
    ("ssd_quality",      generate_ssd_policy, dict(threshold=0.7, editing_threshold=0.5, do_verify_policy="mask_span_length")),
    ("ssd_speed",        generate_ssd_policy, dict(threshold=0.5, editing_threshold=0.0, do_verify_policy="mask_span_length")),
    # paper's conservative S2D2 setting (reported 4.4x vs static baseline, slightly higher acc)
    ("ssd_conservative", generate_ssd_policy, dict(threshold=0.9, editing_threshold=0.5, do_verify_policy="mask_span_length", min_ssd_span_length=1)),
]
if args.smoke:
    CONFIGS = [("cached_smoke", generate_cached, dict(threshold=0.7, editing_threshold=0.5, max_post_steps=4))]
    COMMON["gen_length"] = 64

results = {}
DFLASH_REF = 137.0
for label, fn, extra in CONFIGS:
    tot_tok = 0; tot_t = 0.0; tot_nfe = 0; previews = []
    print(f"\n--- {label} ({fn.__name__}) {extra} ---", flush=True)
    ok = True
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
        torch.cuda.synchronize()
        ts = time.time()
        try:
            out, stats = fn(model=model, input_ids=ids, **COMMON, **extra)
            torch.cuda.synchronize()
            el = time.time() - ts
            n = out.shape[1] - ids.shape[1]
            tot_tok += n; tot_t += el; tot_nfe += int(stats.get("nfe", 0))
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            previews.append(txt[:100])
            print(f"  {n}t {el:.2f}s = {n/el:.1f} tok/s  nfe={stats.get('nfe')}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)
            ok = False
            break
    if ok and tot_t > 0:
        avg = tot_tok / tot_t
        results[label] = dict(avg_tok_s=round(avg, 2), total_tokens=tot_tok,
                              total_time_s=round(tot_t, 2), total_nfe=tot_nfe,
                              tok_per_nfe=round(tot_tok / max(tot_nfe, 1), 3),
                              vs_dflash137=round(avg / DFLASH_REF, 3),
                              fn=fn.__name__, config=extra, sample_outputs=previews)
        print(f"  AVG: {avg:.1f} tok/s ({avg/DFLASH_REF:.2f}x vs DFlash 137)", flush=True)
    else:
        results[label] = dict(status="FAILED", fn=fn.__name__, config=extra)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=2)
print(f"\n[bench] wrote {args.out}", flush=True)
print("\n=== SUMMARY (tok/s) ===", flush=True)
for k, v in results.items():
    if "avg_tok_s" in v:
        print(f"  {k:20s} {v['avg_tok_s']:7.1f} tok/s  tok/NFE={v['tok_per_nfe']}", flush=True)
    else:
        print(f"  {k:20s} FAILED", flush=True)
