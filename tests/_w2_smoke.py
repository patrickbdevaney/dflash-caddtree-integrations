import os, sys, json, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
sys.path.insert(0,"/tests"); import harness as H
try:
    llm = H.build_llm(tree_width=int(os.environ.get("DFLASH_TREE_WIDTH","2")))
    print("ENGINE_UP", flush=True)
    toks = H.greedy(llm, ["def fibonacci(n):"], max_tokens=40, seed=0)[0]
    print("OUT_TOKENS", json.dumps(toks), flush=True)
    print("N_TOKENS", len(toks), flush=True)
    print("W2_SMOKE_OK", flush=True)
except Exception as e:
    print("W2_SMOKE_FAIL", repr(e), flush=True); traceback.print_exc(); sys.exit(1)
