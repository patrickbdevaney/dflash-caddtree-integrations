
## UPDATE 2026-06-09 — SGLang on Thor: BLOCKED (both prebuilt + source-build), characterized
Attempted to actually stand up SGLang dLLM (JointThreshold) for LLaDA2.1-mini. Two independent walls:

### Wall 1 — prebuilt wheels: aarch64 sgl-kernel ↔ torch ABI corridor
- sgl-kernel cu130 aarch64 wheel only exists at **0.3.21** (github sgl-project/whl), built against
  **torch 2.9.1** (per sglang 0.5.9 requires_dist). The DFlash image ships **torch 2.10.0**.
  Loading sgl-kernel 0.3.21 against torch 2.10.0 → `undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementation...` (C10 ABI mismatch).
- Version matrix (PyPI requires_dist): sglang 0.5.9→torch 2.9.1+sgl-kernel 0.3.21; 0.5.10→2.9.1+
  sglang-kernel 0.4.1; 0.5.11/0.5.12→torch **2.11.0**+sglang-kernel 0.4.2. **No sglang pins torch
  2.10.0** (the image's version). JointThreshold dLLM landed in **0.5.9**.
- Downgrading the image to stock torch 2.9.1 is blocked: vLLM (and torchvision/torchaudio/torchcodec)
  pin torch==2.10.0 → ResolutionImpossible; and removing them, the aarch64 cu130 torchvision/torchaudio
  for 2.9.1 aren't cleanly available (jetson mirror has them only for the image's 2.10.0 set). The
  coherent aarch64 stack on this box is jetson-mirror torch 2.10.0; the prebuilt sgl-kernel targets
  stock torch 2.9.1/2.11.0 — no overlap.
- (Layered install DID work to the point of `--dllm-algorithm` present: sgl-kernel 0.3.21 direct wheel
  + sglang 0.5.1.post2 deps + `--no-deps` bump to 0.5.12 + orjson, flashinfer 0.6.6, torch 2.10.0 kept.
  But the serve crashes at import: sgl-kernel sm100 .so undefined-symbol vs torch 2.10.0 ABI.)

### Wall 2 — source-build sgl-kernel vs torch 2.10.0: impractical compile
- Built sgl-kernel from source (sglang v0.5.9 tag) against the image's torch 2.10.0. CMakeLists hardcodes
  a **7-arch gencode set** for CUDA 13 aarch64 (sm_87, sm_90a, sm_100a, sm_103a, sm_110a, sm_120a,
  sm_121a) — every kernel compiled 7×. First attempt ran **40+ min** at 8 cores, no end (compiling
  `common_ops_sm90_build.dir`), capped at 48 GiB (cgroup; host safe).
- Patched CMakeLists to **sm_110a-only + FA3 OFF** (FA3 is Hopper sm90 and crashes on sm_110 per
  dflash-thor-build memory). Still **18+ min, no end**: sgl-kernel vendors and compiles **FlashAttention
  (sparse, sm80/sm90), bundled FlashInfer, and CUTLASS marlin-MoE from source** as `_deps/` subprojects
  whose own arch configs the top-level gencode patch does NOT reach. Estimated 60–90 min+; not a
  practical bring-up. Stopped (graceful), CMakeLists reverted.

### Conclusion
SGLang dLLM serving for LLaDA2.1-mini is **not practically achievable on this Thor right now**:
prebuilt sgl-kernel is ABI-locked to torch versions not coherently installable on aarch64 here, and a
from-source sgl-kernel is a 60–90 min+ vendored-dep (FA+FlashInfer+CUTLASS) compile. A future path:
build the FULL all-stock-torch-2.9.1 aarch64 env from a clean base (not layered on the DFlash image) +
prebuilt sgl-kernel 0.3.21 — OR a CI-built sgl-kernel wheel for torch 2.10.0 aarch64. Deferred.
**The working/closest diffusion-serving path on this box is vLLM via dllm-plugin (see
vllm_diffusion_result.md), pursued next.** No SGLang tok/s reported (never served). No fabrication.
