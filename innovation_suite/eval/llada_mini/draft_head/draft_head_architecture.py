"""
DFlare-style draft-head SKELETON for a LLaDA2.1-mini (diffusion) TARGET.
ARCHITECTURE DEFINITION ONLY — not trained, not runnable for inference.
See draft_head_feasibility.md: speed payoff is UNPROVEN/contraindicated on Thor
(Stage 4: adding speculative verify to LLaDA2.1-mini was slower than plain KV-cache).
This file exists to make the proposed design concrete, not to claim a speedup.

Dims from LLaDA2.1-mini config.json:
  hidden_size=2048, num_attention_heads=16, num_hidden_layers=20,
  moe_intermediate_size=512, num_experts_per_tok=8 -> draft FFN ~= 8*512 = 4096,
  vocab_size=157184.
"""
import torch
import torch.nn as nn

HIDDEN_SIZE = 2048
NUM_HEADS = 16
NUM_TARGET_LAYERS = 20
VOCAB_SIZE = 157184
D_EXPERT = 512
NUM_ACTIVE = 8
DRAFT_FFN_DIM = NUM_ACTIVE * D_EXPERT          # 4096
BLOCK_SIZE = 16                                # DFlare default
NUM_DRAFT_LAYERS = 7                           # DFlare
# 9 target layers uniformly in [2, num_layers-3] = [2, 17]
FUSION_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 17]
NUM_FUSION = len(FUSION_LAYERS)                # 9


class LayerWiseFusion(nn.Module):
    """DFlare fusion: f_t^(i) = RMSNorm( sum_j softmax(alpha^(i))_j * h_t^(j) ).
    D x T = 7 x 9 = 63 scalar params total. Each draft layer i gets a distinct input."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(NUM_DRAFT_LAYERS, NUM_FUSION))
        self.norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-6)

    def forward(self, target_hidden, draft_layer_idx):
        # target_hidden: [B, S, NUM_FUSION, HIDDEN]  (gathered at FUSION_LAYERS)
        w = torch.softmax(self.alpha[draft_layer_idx], dim=-1)        # [NUM_FUSION]
        fused = torch.einsum("bsfh,f->bsh", target_hidden, w)
        return self.norm(fused)


class LLaDA2DraftHead(nn.Module):
    """7-layer draft transformer over fused BIDIRECTIONAL target hidden states.
    Novelty vs DFlash-on-AR: target states are within-block bidirectional, so each
    position's conditioning encodes the whole block -> hypothesised higher acceptance.
    Block-parallel output (no autoregression within the draft block)."""
    def __init__(self):
        super().__init__()
        self.fusion = LayerWiseFusion()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=NUM_HEADS,
                                       dim_feedforward=DRAFT_FFN_DIM,
                                       batch_first=True, norm_first=True)
            for _ in range(NUM_DRAFT_LAYERS)
        ])
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, target_hidden):
        # target_hidden: [B, S, NUM_FUSION, HIDDEN] from a target forward at FUSION_LAYERS
        x = None
        for i, layer in enumerate(self.layers):
            cond = self.fusion(target_hidden, i)      # distinct fused input per layer
            x = cond if x is None else x + cond
            x = layer(x)                              # bidirectional (no causal mask)
        return self.lm_head(x[:, -BLOCK_SIZE:, :])    # [B, BLOCK_SIZE, VOCAB]


if __name__ == "__main__":
    h = LLaDA2DraftHead()
    n = sum(p.numel() for p in h.parameters())
    print(f"draft-head params: {n/1e6:.1f}M  (fusion scalars: {h.fusion.alpha.numel()})")
    print("NOTE: definition only — requires data-gen + training (see feasibility doc).")
