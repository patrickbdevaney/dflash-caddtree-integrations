═══════════════════════════════════════════
EVAL TODO — Future sessions when available
═══════════════════════════════════════════

COMPLETED (valid, sufficient for PR submissions):
  - S-NIAH 1M: DroPE, 3 depths, 5 instances each,
    Wilson CI reported. Result:
      p05 (~50k):  5/5  Wilson95 [0.566, 1.0]
      p50 (~500k): 4/5  Wilson95 [0.376, 0.964]  (one mid-window miss, expected)
      p95 (~950k): 5/5  Wilson95 [0.566, 1.0]    (deep-tail, headline cell)
      overall 14/15. DroPE factor=4 zero-shot inference-time (no recal), BF16 KV,
      no spec-decode, SM110a. p95 5/5 but N=5 -> Wilson lo=0.566; N>=13 needed for
      lo>0.8 headline claim (re-run p95 at N=20 = Tier 1).
  - DroPE graph-safe gate: 20/20 bitwise under
    cudagraphs (commit b5858d6)
  - APC coexistence: cold==warm 20/20 + 1.66x e2e
  - Tree speculation: spec_state_indices_tensor
    routing, Tree-WY O(B²d) finding
  - FP8 KV: root cause characterized, error
    signatures documented
  - HybridCorrectnessGate: lossless/lossy/recall
    gates built and self-tested

TIER 1 — Next session, highest priority:
  YaRN baseline S-NIAH at 1M (comparison condition)
    15 instances, 3 depths, same protocol as DroPE
    ~2-3 hours GPU time
    Makes DroPE result comparative not just absolute
    Required for honest "DroPE vs YaRN" claim

  S-NIAH with distractors at p95
    3 distractors, ask for specific token
    Same 15 instances
    Rules out easiest-task criticism
    ~2 hours additional

  N=20 at p95 for DroPE
    Tighten Wilson CI lower bound to >0.83
    20 instances at 950k only
    ~4 hours
    Upgrades from scouting to headline claim

TIER 2 — Follow-on session, research paper quality:
  LongPPL Phase A
    5 configs × forward pass at 512k
    Llama-3.1-8B discriminator via HF transformers
    GovReport corpus (download first)
    ~90 min setup + 65 min compute
    Gives config rankings and generation quality
    metric independent of retrieval

  MK-NIAH (multi-key) DroPE + YaRN
    2 keys per instance, 15 instances, 3 depths
    Tests multi-fact retrieval not just single needle
    ~6 hours

  Spearman rank correlation
    Requires LongPPL Phase A AND matching NIAH
    for all 5 configs — ~14 hours compute
    Methodology validity check
    Deferred until both prerequisites complete

TIER 3 — Full research paper, whenever compute available:
  RULER NIAH subset (4 task variants)
    262k and 1M, 20 samples per cell
    Standard RULER pipeline at pinned commit
    ~18 hours total
    Upgrades claim from "S-NIAH retrieval"
    to "RULER NIAH subset validated"

  RULER non-NIAH tasks
    Variable tracking, aggregation, QA
    Tests context management beyond retrieval
    ~10 additional hours at 1M
    Required for full paper claim

  Recalibrated DroPE checkpoint
    LoRA fine-tuning on Thor (~days)
    Full recalibration requires datacenter hardware
    Expected to improve multi-key and mid-window
    results significantly
    Highest leverage follow-on action

IMPLEMENTATION TODO (non-eval, code work):
  S3: Accept-offset rollback MTP+DFlash
      Port #40738 to EagleProposer path
      ~1 coding session, no long GPU time

  S5: SnapKV on GDN hybrid attention layers
      Search vLLM for existing infra first
      LongPPL + NIAH sweep after implementation
      ~1 coding session + eval overnight

  S6: KV offload OffloadingConnector fix (#36463)
      Conditional on S5 results
      Small targeted fix

  Block B: Nemotron-H unblock
      vLLM 0.16.2 → 0.20 port or new image
      Then full innovation suite on Nemotron-H

PR SUBMISSIONS (ready now, no eval blocking):
  PR 1: GDN APC coexistence [Bugfix] — SUBMIT FIRST
  PR 2: DroPE rope_type [Feature] — SUBMIT SECOND
  PR 3: Accept-offset rollback [Bugfix] — after S3
  PR 4: FP8 KV hybrid guard [Bugfix] — standalone
  PR 5: Tree speculation [RFC issue] — file now
═══════════════════════════════════════════════════

---

## VISION / MULTIMODAL STACK — M-RoPE + APC + DFlash

### Background

Qwen3.6-35B-A3B is NATIVELY MULTIMODAL — the vision encoder is built into the model;
there is NO separate VLM variant to download (the on-disk NVFP4 weights already include
vision; confirm via `vision_config` in config.json). Enabling vision requires M-RoPE
(Multimodal RoPE) — the positional encoding extension that handles vision tokens, text
tokens, and their 3D spatial relationships (time T, height H, width W dimensions)
simultaneously. Without M-RoPE, image inputs either fail or receive incorrect positional
encodings.

vLLM HEAD does not implement get_mrope_input_positions for Qwen3.6 — confirmed by
AEON-7/Qwen3.6-NVFP4-DFlash (2026). A small inline fallback patch is required.
The AMD Strix Halo repo (hec-ovi/vllm-awq4-qwen) confirmed DFlash + vision + tool calling
+ APC work together on 128GB unified memory hardware — the same architecture as Thor.

### IMPLEMENTATION TODO

TIER 1 — Required patches (small, targeted):

  M-ROPE FALLBACK PATCH [Bugfix, ~20 LOC]
    File: vllm/model_executor/models/qwen3_next.py
      or equivalent VLM model class
    What: implement get_mrope_input_positions with
      inline fallback for canonical text-only
      positions (T=H=W=arange) when vision inputs
      are absent, and correct 3D M-RoPE position
      IDs (T, H, W) when vision inputs are present.
    Reference: AEON-7/Qwen3.6-NVFP4-DFlash patch set
      commit history — find the M-RoPE fallback
      commit specifically.
    Gate: serve Qwen3.6 VLM variant, send a single
      image + text prompt, confirm non-null response
      with no position encoding errors in logs.
    PR type: [Bugfix] — missing implementation, not
      a new feature.

  VLM WEIGHTS VERIFICATION
    Qwen3.6-35B-A3B is natively multimodal — vision is ALREADY in the on-disk NVFP4
    weights; there is NO separate -VL variant to download.
    Check: grep -i vision ~/models/Qwen3.6-35B-A3B-NVFP4/config.json
    Confirm config.json contains "vision_config" (and a visual/ tensor namespace in the
    weights). No download step required.
    Note: with the vision tower active, confirm the full model still fits within the
    memory envelope alongside the 1M KV cache budget (~43GB).

TIER 2 — Correctness gates (run after patch applied):

  APC + VISION CORRECTNESS GATE
    Goal: verify APC prefix caching works correctly
      when the prefix contains image tokens.
    Protocol:
      1. Serve Qwen3.6 VLM with --enable-prefix-caching
         --mamba-cache-mode align and
         --mm-processor-cache-type shm
      2. Send the same image + system prompt + question
         twice (cold then warm)
      3. Verify: warm response bitwise matches cold
         response (same as the text-only cold==warm gate)
      4. Verify: prefix cache hit rate > 0% on warm
         request (image tokens ARE being cached)
    Why this matters: image tokens in the prefix must
      be handled correctly by the APC block alignment
      logic. The mamba_block_size alignment constraint
      applies to all prefix tokens including image
      tokens. A misalignment would cause 0% cache
      hits on the image-bearing prefix.
    Document: whether image tokens participate in
      APC prefix blocks or are excluded (vision
      encoder output may bypass the KV cache
      entirely depending on architecture).

  DFLASH + VISION ACCEPTANCE RATE MEASUREMENT
    Goal: characterize whether DFlash acceptance
      rate degrades with interleaved vision tokens.
    Protocol:
      1. Baseline: DFlash on text-only prompt, 20
         prompts, measure acceptance rate and tok/s.
      2. Vision: DFlash on image + text prompt of
         similar text length, 20 prompts, same
         measurement.
      3. Compare acceptance rates and tok/s.
    Expected finding: the DFlash drafter was trained
      on text. Vision tokens in the sequence may
      reduce acceptance rate because the drafter
      has not seen image token patterns. Document
      the actual degradation (if any). Even partial
      acceptance rate is still net positive.
    Note: the vision PREFILL is not speculated —
      only the text generation portion benefits
      from DFlash. The measurement should reflect
      this by measuring tok/s on the generation
      portion only, not including vision prefill.

  MROPE + DROPE COMPATIBILITY GATE
    Goal: verify DroPE (rope_type='drope') works
      correctly when M-RoPE is active for vision
      inputs.
    The concern: DroPE applies identity rotation
      beyond native context on the standard RoPE
      dimensions. M-RoPE adds T/H/W dimensions on
      top of the standard position dimension. The
      DroPE cache overwrite (cos=1.0, sin=0.0
      beyond native) must not corrupt the M-RoPE
      dimensional structure.
    Protocol:
      1. Serve with rope_type='drope' + VLM enabled
      2. Send image + text prompt with context
         length > 262k (beyond native threshold)
      3. Verify: coherent response, no positional
         encoding artifacts, no attention pattern
         corruption visible in generation quality
    If DroPE corrupts M-RoPE: the fix is to scope
      the identity rotation to the standard 1D
      position dimension only, leaving the H/W/T
      dimensions of M-RoPE untouched. DroPE should
      only affect the sequential position encoding,
      not the spatial encodings of image patches.
    This gate is required before claiming DroPE
      and vision are simultaneously usable.

TIER 3 — Production serve config (after gates pass):

  FULL VISION STACK SERVE COMMAND
    The target production serve configuration:

    docker run [standard flags] \
      vllm-dflash-thor:ddtree \
      vllm serve [VLM model path] \
      --rope-scaling \
        '{"rope_type":"drope",
          "factor":4.0,
          "original_max_position_embeddings":
          262144}' \
      --max-model-len 1010000 \
      --kv-cache-dtype auto \
      --enable-prefix-caching \
      --mamba-cache-mode align \
      --mm-processor-cache-type shm \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --speculative-config \
        '{"method":"dflash",
          "num_speculative_tokens":15}' \
      --port 8000

    This is the complete Hermes executor stack:
      DroPE 1M context extension
      APC prefix caching (text + image tokens)
      Vision encoder with M-RoPE
      Tool calling
      DFlash speculative decoding
      OpenAI-compatible API surface

  APC EFFICIENCY WITH REPEATED IMAGES
    For agentic workloads where the same image
    (UI screenshot, diagram, codebase visualization)
    appears in multiple turns, the image tokens
    should be prefix-cached after the first turn.
    Measure: cache hit rate on image token blocks
    across a 10-turn conversation with a static
    image in the system prompt.
    Expected: near-100% cache hit rate on image
    blocks after turn 1, same as text system prompt.
    This is the production efficiency claim for
    multimodal agentic workloads.

### PR IMPLICATIONS

  M-RoPE fallback patch:
    Small [Bugfix] PR to vLLM, ~20 LOC.
    Affects all Qwen3.5/3.6 VLM users in vLLM.
    Currently undocumented gap — no open issue found
    as of session date. File the issue first, then
    the PR, same discipline as the other bugfixes.

  APC + vision correctness:
    If the cold==warm gate fails for image-bearing
    prefixes, the fix is an extension of the
    mamba_cache_mode decoupling work already done
    for text. The same HybridCorrectnessGate
    methodology applies — add a vision-input variant
    of the cold==warm test to
    tests/v1/worker/test_hybrid_correctness.py.
    This extends the existing PR rather than
    creating a new one.

  DroPE + M-RoPE compatibility:
    If the compatibility gate reveals a conflict,
    the fix scopes the identity rotation to the
    sequential position dimension only. This is
    a one-function change to the DroPE cache
    builder in rotary_embedding/ — add a guard
    that only overwrites the standard 1D position
    rows, not any M-RoPE dimensional extensions.
    Update the DroPE PR description to document
    M-RoPE compatibility.

### ATTESTATION NOTE

All vision gates must follow the same attestation
protocol as the text-only gates:
  - Hardware record in attestation/hardware/
  - Config JSON with VLM model path and all flags
  - Per-instance raw outputs (image hash + response)
    in attestation/raw_outputs/
  - GPG-signed commits for all gate results

The image hash (sha256 of the image file) should be
recorded in each raw output record so the exact
image used for each gate run is reproducible.
