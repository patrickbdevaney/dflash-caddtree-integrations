"""Host-side SGLang benchmark client (stdlib only). Measures wall-clock tok/s on the
OpenAI-compatible endpoint for the standard coding prompts. Usage:
  python3 sglang_bench_client.py --port 8002 --label sglang_bf16 --out out.json
"""
import argparse, json, time, urllib.request, urllib.error, sys

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--model", default=None)
args = ap.parse_args()
BASE = f"http://localhost:{args.port}"

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# resolve model id
model = args.model
info = get("/v1/models")
if not model:
    try:
        model = info["data"][0]["id"]
    except Exception:
        model = "default"
print(f"[client] model id: {model}", flush=True)

PROMPTS = [
    "Write a Python function that finds the longest palindromic substring.",
    "Implement a binary search tree in Python with insert, search, and delete methods.",
    "Write a Python function to solve the coin change problem using dynamic programming.",
]

def chat(prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = resp.get("usage", {})
    ntok = usage.get("completion_tokens", 0)
    text = resp["choices"][0]["message"]["content"]
    return ntok, dt, text

tot_tok = 0; tot_t = 0.0; samples = []
print(f"--- {args.label} ---", flush=True)
for p in PROMPTS:
    try:
        n, dt, text = chat(p)
        tot_tok += n; tot_t += dt
        samples.append(text[:120])
        print(f"  {n}t {dt:.2f}s = {n/dt:.1f} tok/s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        samples.append(f"ERROR: {e}")

res = {"label": args.label, "model": model}
if tot_t > 0 and tot_tok > 0:
    avg = tot_tok / tot_t
    res.update(avg_tok_s=round(avg, 2), total_tokens=tot_tok, total_time_s=round(tot_t, 2),
               vs_hf_bf16_649=round(avg / 64.9, 3), vs_dflash_137=round(avg / 137.0, 3),
               samples=samples)
    print(f"  AVG: {avg:.1f} tok/s ({avg/64.9:.2f}x vs HF BF16 64.9, {avg/137.0:.2f}x vs DFlash 137)", flush=True)
else:
    res["status"] = "FAILED"; res["samples"] = samples
json.dump(res, open(args.out, "w"), indent=2)
print(f"[client] wrote {args.out}", flush=True)
