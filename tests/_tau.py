import os,sys,time,json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; sys.path.insert(0,"/tests"); import harness as H
from vllm import SamplingParams
W=int(os.environ["DFLASH_TREE_WIDTH"]); llm=H.build_llm(tree_width=W)
import vllm.v1.sample.rejection_sampler as rs
llm.generate(["hi"],SamplingParams(temperature=0,max_tokens=8)); rs._NSTEPS['n']=0
prompts=["def fibonacci(n):","def quicksort(arr):","def binary_search(arr,target):","# reverse a linked list\n","def catalan(n):"]
t0=time.time(); outs=llm.generate(prompts,SamplingParams(temperature=0.0,max_tokens=128,seed=0)); dt=time.time()-t0
ntok=sum(len(o.outputs[0].token_ids) for o in outs); steps=rs._NSTEPS['n']
print("TAU_RESULT", json.dumps({"W":W,"tokens":ntok,"steps":steps,"tau":round(ntok/max(1,steps),2),"tok_s":round(ntok/dt,2)}),flush=True)
