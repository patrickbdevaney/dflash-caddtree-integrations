# Stage E — MoE expert prefetch: LIKELY MOOT on Thor (design analysis)

Stage F profiling shows the MoE expert FP4 GEMM (`cutlass_fp4_group_mm`) dominates
C_verify (~34% of CUDA time). Stage E proposes routing-ahead + prefetching experts.

**Why it is likely moot on Jetson Thor specifically:**
- Thor has **128 GB unified LPDDR5x** (~273 GB/s). The 35B-A3B-NVFP4 weights (~20 GB)
  are **fully resident** — there is no host→device or disk→device expert fetch to hide.
- The `cutlass_fp4_group_mm` cost is the GEMM itself **reading resident weights over the
  273 GB/s bus** — that read is intrinsic to the matmul and bandwidth-bound. Prefetch does
  not reduce bytes moved; it only hides *latency* of a separate fetch, which doesn't exist
  here (weights are already in unified memory, and the working set — 256 experts — vastly
  exceeds L2, so prefetch-to-L2 won't help).
- Therefore expert prefetch cannot reduce the dominant term on this device. It would help
  on a discrete-GPU + host-offload setup (experts in host RAM), not Thor unified memory.

**Recommendation:** do NOT implement on Thor. The real MoE lever here would be reducing
*bytes read* (e.g. lower-precision experts, fewer active experts) — out of scope (no-train,
no model change). **Status: documented moot; needs runtime confirmation only if a host-
offload config is ever used.**
