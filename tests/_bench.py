import os,sys,time,json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; sys.path.insert(0,"/tests"); import harness as H
W=int(os.environ["DFLASH_TREE_WIDTH"])
llm=H.build_llm(tree_width=W)
from vllm import SamplingParams
prompts=["def fibonacci(n):","def quicksort(arr):","def binary_search(arr, target):",
         "Write a function to reverse a linked list:","# Compute the nth Catalan number\ndef catalan(n):"]
sp=SamplingParams(temperature=0.0,max_tokens=128,seed=0)
# warmup
llm.generate(["hi"],SamplingParams(temperature=0,max_tokens=8))
t0=time.time(); outs=llm.generate(prompts,sp); dt=time.time()-t0
ntok=sum(len(o.outputs[0].token_ids) for o in outs)
print(f"BENCH W={W} total_tokens={ntok} wall={dt:.2f}s tok_s={ntok/dt:.2f}", flush=True)
