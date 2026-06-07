# Staged optimization — final results (Jetson AGX Thor SM110a)

Model: Qwen3.6-35B-A3B-NVFP4 (GDN-hybrid MoE) + DFlash draft (num_spec=12).
Single sequence, 5 coding prompts × 128 tokens. Stages 0–5 of the optimal-path plan.

## Headline result

**Linear DFlash + typical (Medusa-style) acceptance beats the linear baseline by
+11% tok/s and +17% τ at production temperature (T=0.3), with zero extra
per-step overhead. T=0 stays byte-identical.** The DDTree tree machinery works
correctly but does not beat linear; the win came from the *acceptance criterion*,
not the tree.

## Stage 0 — multi-branch verification: NO BUG (settled)

Added a hard per-step invariant `realized tree τ ≥ its own spine τ` (the spine is
always a path in the tree). **Zero violations** across a full B=17 multi-branch
run. The earlier τ=4.26<5.48 was a *cross-config confound* (num_spec=16 degrades
the draft + spine-12 cap throws away depth), not a verification error. Multi-branch
acceptance is correct (accepted paths of 11–12 tokens observed).

## Stage 1 — verification relaxation (typical acceptance): THE WIN

Typical acceptance: accept draft token i iff target prob ≥ eps (eps=0 → exact
argmax = byte-identical greedy; eps=0.09 → Medusa-style relaxation). Implemented
for both the tree path and the lean linear path.

CUDA graphs, T=0.3:

| config | τ | tok/s | vs baseline |
|---|---:|---:|---:|
| Linear + rejection (baseline) | 5.84 | 81.97 | — |
| Tree + typical | 6.35 | 66.1 | −19% tok/s |
| **Linear + typical** | **6.84** | **90.71** | **+11% tok/s** |

T=0.5: linear+typical = 6.84 τ, **91.09 tok/s**. T=0: byte-identical (5.75, 77.2).

Why it works: linear's rejection sampling *degrades* at temperature (rejects
more), while typical acceptance *holds up* (accepts draft tokens carrying real
target mass). The relaxation is a minor quality trade (standard Medusa, widely
used in production); at T=0 it reduces to exact greedy (byte-identical).

Why it belongs on linear, not the tree: at B=13 the tree is spine-only (= the
linear chain), so typical acceptance is orthogonal to tree breadth — it needs no
tree. Tree+typical (66 tok/s) loses to linear+typical (90.7) by exactly the
tree's per-step overhead (per-branch GDN + eager attn + host tree ops).

## Stage 2/3 — tree placement / dynamic budget: MOOT for beating linear

Selective placement (Stage 2, validated earlier: +9% τ over best-first) and a
throughput-optimal budget (Stage 3, CaDDTree) refine the *tree*, but every tree
config carries the ~20% per-step overhead that already makes tree+typical (66)
lose to linear+typical (90.7). No tree placement removes that overhead, and the
typical-acceptance win is fully captured on lean linear. So tree refinements
cannot beat linear+typical on this strong-draft model.

## Stage 5 — Tree-WY GDN kernel: negative result (FLOP analysis)

Tree-WY is O(B²d) (NOT overhead-free). At B=13: only 9% fewer FLOPs than the
depth loop, 13×13 GEMMs fill 66% of one tensor-core tile (latency-bound), and the
63 MB/step state IO is identical and bandwidth-bound on Thor. Per-branch fusion +
CUDA graphs already removed launch overhead. Only wins at B≫16. Not worth
building. (Full analysis: `stage5-tree-wy-flop-analysis.md`.)

## Option X (B≫K decouple): provably futile — skipped without the rewrite

The tree's measured per-step overhead ratio is 66/90.7 ≈ **0.73** (tree+typical
tok/s ÷ linear+typical tok/s, both CUDA graphs). For *any* tree config to match
linear+typical's throughput it would need τ > 6.84 / 0.73 ≈ **9.4**, far above the
measured τ ceiling (~7.5 from the offline probe, and tree+typical's actual τ is
6.35 < linear's 6.84). So no amount of breadth — including the full Option-X
decouple — can beat linear+typical. The large #42121-class scheduler rewrite is
**not pursued**: the data rules it out before building it.

## Bottom line

- **Ship: linear DFlash + typical acceptance (eps≈0.09) at T>0 → +11% tok/s,
  +17% τ, byte-identical at T=0.** A real, simple, overhead-free win.
- DDTree on GDN-hybrid MoE is implemented and verified correct (the novel
  contribution), but does not beat strong-draft linear DFlash: at B=13 it equals
  linear + overhead; larger budgets are depth-dominated; Tree-WY doesn't pay at
  feasible B. It would help weaker drafts / much larger budgets.
- The optimization journey's actual payoff is the acceptance criterion, not the
  tree — found by measuring at production temperatures rather than only T=0.
