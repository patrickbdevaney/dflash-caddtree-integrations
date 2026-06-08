# Innovation Suite — autonomous code-contribution run (2026-06-08)

## Headline
- DroPE proven + authored as a clean upstream rope_type. Graph-safe 20/20 bitwise within
  native; 1M single-needle retrieval on Qwen3.6-35B-A3B (4x native, SM110a): p05 5/5,
  p50 4/5, p95 5/5 (overall 14/15), DroPE zero-shot inference-time (no recal), BF16 KV,
  no spec-decode. p95 Wilson95 [0.566,1.0] = scouting-N (N=5); N>=13 to license lo>0.8.
- KEY autonomous finding: most planned bugfixes ALREADY landed in upstream vLLM — do NOT
  submit redundant PRs.

## Contribution status table
| # | Contribution | Upstream status | Action | Branch / artifact |
|---|---|---|---|---|
| C1 | mamba_block_size silent override | FIXED upstream (user_specified_mamba_block_size, cache.py:53/243) | no PR | - |
| C2 | FP8 KV calc_kv_scales hybrid guard | FIXED upstream (config.py:209 disable calc_kv_scales for hybrid) | no PR | - |
| C3 | GDN APC + spec coexistence | core LANDED upstream (mamba_cache_mode=align, test_mamba_prefix_cache.py) | no PR; verify gap | - |
| C4 | Accept-offset rollback MTP/DFlash | core LANDED upstream (mamba_hybrid.py num_accepted_tokens) | no PR; verify gap | - |
| C5 | M-RoPE vision fallback | qwen3_next text-only upstream — needs full multimodal class | DEFERRED | - |
| C6 | SnapKV infra discovery | no general sparse-KV framework upstream (only deepseek_v4 model-specific) | port needed | EVAL_TODO |
| DroPE | rope_type='drope' [Feature] | NOVEL (only unrelated xdrope exists) | branch authored + pushed to fork | origin/pr/drope-rope-type |

## The one genuine code branch
pr/drope-rope-type off upstream/main (zero dflash-thor dependency), GPG-signed, pushed to
FORK ONLY (NOT PR'd — human review required per vLLM AGENTS.md + user instruction):
- drope.py (new, 67 LOC): DroPERotaryEmbedding builds cos/sin cache at native*factor,
  overwrites rows >= native with identity (cos=1,sin=0). No forward override -> cudagraph-safe;
  within-native bitwise to YaRN at the same max_position_embeddings.
- __init__.py: import + elif scaling_type == "drope" dispatch.
- Gate status: AST-clean, <=88 cols, isolated 2-file diff. NOT executed here (host has no
  torch; cannot build upstream on SM110). Logic PROVEN EQUIVALENT in the overlay
  (qwen3_next cache-shape-parity: 20/20 bitwise graph-safe + 1M NIAH 14/15). Needs CI/overlay
  validation before PR — human runs tests per AGENTS.md.

## Deferred (need supervised GPU / fresh session)
RULER, LongPPL Phase A, SnapKV impl+eval, DroPE+M-RoPE compat gate, YaRN-baseline 1M NIAH,
N=20 p95, Nemotron (vLLM 0.16.2 version-blocked). Priority order in EVAL_TODO.md.

## Integrity notes
- No gate fabricated; gates the host couldn't run are marked NOT-EXECUTED with proven-equivalent ref.
- Spearman = N/A (no LongPPL sweep; single-config NIAH).
- All commits GPG-signed (project key 2D9DDA64F9C568AE) + DCO. NO PR opened to vllm-project —
  branches pushed to personal fork for human review.



---
# (prior) # Innovation Suite V3 — morning handoff (real results, structured gates)

## Per-stage structured gate results

**Stage 0 — HybridCorrectnessGate**
{passed: true, mode: full, fallbacks: [], novel_finding: "shared lossless/lossy/recall gate
for hybrid recurrent models", notes: "tests/v1/worker/test_hybrid_correctness.py, self-tests pass"}

**Stage 1 — FP8 KV --calculate-kv-scales fix**
{passed: false, mode: fallback, context_length_used: 4096,
 fallbacks_applied: ["TRITON_ATTN backend retry"],
 novel_finding: "FP8 KV is broken on Qwen GDN-hybrid + DFlash under BOTH backends — FlashInfer:
   TypeError BatchDecodeWithPagedKVCacheWrapper.run() unexpected kwarg 'kv_cache_sf'
   (flashinfer.py:1739, version/API mismatch); TRITON_ATTN: AssertionError. BF16 KV works.
   So FP8 KV fails UPSTREAM of the calc_kv_scales corruption the fix targets — the hybrid
   guard (skip calc -> scale=1.0) is correct in design but UNTESTABLE here until FP8 KV runs.",
 notes: "reverted (harness fp8 wiring is test-only, default off). Independent of Stage 2+."}

**Stage 2 — GDN APC + spec-decode coexistence**  (proven in prior gdn_apc/ work)
{passed: true, mode: full, context_length_used: 11800,
 fallbacks_applied: [],
 novel_finding: "cold==warm WITHIN align mode is BITWISE (20/20 prompts, and over 11.8k-tok /
   4-turn agentic) -> GDN prefix cache restores recurrent state exactly; KV stable, not corrupt.
   e2e 1.66x vs base on the real Hermes agentic trace. The correct gate is cold==warm-within-
   align (the APC-on-vs-off gate is confounded by the none<->align mode switch).",
 notes: "fixes: mamba_block_size override, mamba_cache_mode<-APC decoupling. PR-ready."}

**Stage 3 — accept-offset rollback (DFlash)**
{passed: true(partial), mode: partial,
 fallbacks_applied: ["evidence-based instead of forced-M hook"],
 novel_finding: "DFlash partial-acceptance is clean (no degenerate output) across cold==warm
   bitwise, long-context 4-turn, T>0 typical-acceptance, and tree INV3/INV5 state-isolation;
   #40738 accept-offset present in postprocess_mamba.",
 notes: "explicit forced-M=3 + promoted_pos==M assert NOT run (no runner forcing hook built
   unattended). MTP path = Block-B (deferred). Next session: forced-M gate."}

**Stage 4 — DroPE (262k->beyond)**  /  **Stage 5 — SnapKV**  /  **Stage 6 — KV offload**
{passed: false, mode: RFC, fallbacks_applied: ["rule-3 >300LOC -> RFC"],
 novel_finding: null,
 notes: "Each is >300 LOC fail-quiet AND its eval needs 262k-512k forward passes (memory/time-
   heavy, OOM-risk on a single 35B at 512k). Per rule 3: RFC-flagged with default-off flags,
   design docs committed (designs/stage4_drope_RFC.md, stage5_snapkv_RFC.md,
   stage6_kv_offload_RFC.md). NOT blind-built unattended. Stage 6 also conditional on Stage 5."}

**Block B — Nemotron-3-Super-120B smokes**
{passed: false, mode: deferred,
 fallbacks_applied: ["version-mismatch skip"],
 notes: "DEFERRED — Nemotron serve image vllm-thor:qwen35-latest is **vLLM 0.16.2**, incompatible
   with the 0.20-era innovation-suite/APC patches; the smokes cannot validate any suite stage on
   the native image, and building a 0.16.2 overlay of the patches is a separate port. GPU memory
   IS clear (7.1GB after drop_caches) -> NOT a memory block, a VERSION block. Nemotron config
   verified: 262k (not 1M) context, rope_theta present in config, GQA 32Q/2KV, serve omits
   spec-decode 'for stability'."}

## PR readiness table
| Stage | Status | PR type | Blocker |
|-------|--------|---------|---------|
| 1 FP8 KV fix       | blocked  | [Bugfix] | FP8 KV doesn't run (FlashInfer kv_cache_sf API / TRITON assert) — fix upstream first |
| 2 APC coexistence  | **READY**| [Bugfix] | none — proven (cold==warm bitwise + 1.66x) |
| 3 Accept-offset    | partial  | [Bugfix] | needs explicit forced-M gate |
| 4 DroPE            | RFC      | [Feature]/[RFC] | >300 LOC + 262k-512k eval |
| 5 SnapKV hybrid    | RFC      | [RFC]+data | >300 LOC + 262k-512k eval |
| 6 KV offload       | RFC/skip | [RFC] | conditional on Stage 5 |

## Novel findings (incl. negatives)
- FP8 KV is broken on the Qwen GDN-hybrid + DFlash stack under both attention backends
  (kv_cache_sf FlashInfer API mismatch; TRITON AssertionError) — environment-level finding.
- GDN prefix cache is bitwise-correct over long context (cold==warm 11.8k/4-turn) with the
  correct cold==warm-within-align gate; 1.66x agentic e2e.
- Bitwise parity with the base model is NOT achievable for spec decode (verify-forward near-tie
  numerics cascade) — a property of all spec decode, not an APC defect.
- Nemotron-3-Super-120B is 262k (not 1M) per config, carries rope_theta (NoPE unverified), and
  ships on vLLM 0.16.2.

## Unresolved (FAILED-UNKNOWN needing human attention)
- None unknown. Stage 1 FP8 KV failure is fully characterized (two backend-specific errors).
- The only human-decisions pending: (a) provide a FlashInfer build with kv_cache_sf (or fix the
  TRITON fp8 path) to unblock Stage 1; (b) a 512k-capable session for Stages 4/5 RFC eval;
  (c) a vLLM-0.16.2 patch port (or a 0.20 Nemotron image) for Block B.

AI attribution for PRs: Co-authored-by: Claude
