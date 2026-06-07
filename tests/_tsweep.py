import os, sys, json, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; sys.path.insert(0,"/tests"); import harness as H
from vllm import SamplingParams
W=int(os.environ["DFLASH_TREE_WIDTH"]); llm=H.build_llm(tree_width=W)
import vllm.v1.sample.rejection_sampler as rs
prompts=["def fibonacci(n):","def quicksort(arr):","def binary_search(arr,target):","# reverse a linked list\n","def catalan(n):"]
llm.generate(["hi"],SamplingParams(temperature=0,max_tokens=4))
def run(T,eps):
    os.environ["DFLASH_ACCEPT_EPS"]=str(eps); rs._NSTEPS['n']=0
    sp=SamplingParams(temperature=T,max_tokens=128,seed=0)
    t0=time.time(); outs=llm.generate(prompts,sp); dt=time.time()-t0
    ntok=sum(len(o.outputs[0].token_ids) for o in outs); steps=rs._NSTEPS['n']
    return {"W":W,"T":T,"eps":eps,"tau":round(ntok/max(1,steps),2),"tok_s":round(ntok/dt,2)}
grid=[(0.0,0.0),(0.3,0.0),(0.3,0.09),(0.5,0.09)]  # both W: exact + typical
for (T,eps) in grid:
    print("TSWEEP", json.dumps(run(T,eps)), flush=True)
