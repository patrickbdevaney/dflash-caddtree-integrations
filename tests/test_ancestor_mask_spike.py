"""4d Part A spike: FlexAttention ancestor mask over a 22-node W=2 tree on Thor.
Gates all 4d integration: if this fails, the tree-verify attention path is blocked.
"""
import torch, sys
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

dev = "cuda"
B = 22  # total tree nodes

# W=2 tree: root(0) + two branches of depth ~11 each.
# parent[i]: node 0 is root; nodes 1..11 chain on branch A; 12..21 chain on branch B off root.
parent = [-1] * B
# branch A: 1<-0, 2<-1, ... 11<-10
for i in range(1, 12):
    parent[i] = i - 1
# branch B: 12<-0, 13<-12, ... 21<-20
parent[12] = 0
for i in range(13, 22):
    parent[i] = i - 1

anc = torch.zeros(B, B, dtype=torch.bool, device=dev)
for i in range(B):
    j = i
    while j != -1:
        anc[i, j] = True
        j = parent[j]

def ancestor_mask_mod(b, h, q_idx, kv_idx):
    return anc[q_idx, kv_idx]

try:
    block_mask = create_block_mask(ancestor_mask_mod, B=None, H=None,
                                   Q_LEN=B, KV_LEN=B, device=dev)
    q = torch.randn(1, 1, B, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 1, B, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 1, B, 128, device=dev, dtype=torch.bfloat16)
    cfa = torch.compile(flex_attention)
    out = cfa(q, k, v, block_mask=block_mask)
    torch.cuda.synchronize()
    finite = torch.isfinite(out).all().item()
    print("ANCESTOR_MASK_SPIKE shape", tuple(out.shape), "finite", finite)
    print("ANCESTOR_MASK_SPIKE", "PASS" if finite else "FAIL")
    sys.exit(0 if finite else 1)
except Exception as e:
    import traceback; traceback.print_exc()
    print("ANCESTOR_MASK_SPIKE FAIL", repr(e))
    sys.exit(1)
