import os, sys, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; os.environ["DFLASH_PROBE"]="1"
sys.path.insert(0,"/tests"); import harness as H
from vllm import SamplingParams
llm=H.build_llm(tree_width=1)  # linear run; capture draft top-2 + target greedy
import vllm.v1.sample.rejection_sampler as rs
llm.generate(["hi"],SamplingParams(temperature=0,max_tokens=8)); rs._PROBE_RECORDS.clear()
prompts=["def fibonacci(n):","def quicksort(arr):","def binary_search(arr,target):","# reverse a linked list\n","def catalan(n):"]
llm.generate(prompts,SamplingParams(temperature=0.0,max_tokens=128,seed=0))
recs=rs._PROBE_RECORDS
lin=[]; up=[]; low=[]; catch=0; n=0
for r in recs:
    tg,d1,t2=r["tg"],r["d1"],r["t2"]; K=len(tg)
    # linear accepted prefix: d1[i]==tg[i]
    la=0
    while la<K and d1[la]==tg[la]: la+=1
    # tree W=2 accepted prefix (path-independent approx): top1 or top2 == tg
    ta=0
    while ta<K and (t2[ta][0]==tg[ta] or (len(t2[ta])>1 and t2[ta][1]==tg[ta])): ta+=1
    # rigorous lower bound: linear + 1 if the rejection position's top-2 matches
    bc = 1 if (la<K and len(t2[la])>1 and t2[la][1]==tg[la]) else 0
    lin.append(la); up.append(ta); low.append(la+bc); catch+=bc; n+=1
import statistics as st
print("PROBE_RESULT", json.dumps({
  "steps":n,
  "tau_linear": round(st.mean(lin)+1,2),          # +1 bonus
  "tau_tree_lower": round(st.mean(low)+1,2),       # rigorous: branch catches rejection
  "tau_tree_upper": round(st.mean(up)+1,2),        # optimistic: top1|top2 along chain
  "branch_catch_rate": round(catch/max(1,n),3),    # frac steps a W=2 branch helps
  "mean_lin_acc": round(st.mean(lin),2),
  "mean_tree_acc_upper": round(st.mean(up),2),
}), flush=True)
