# Stage 6 — LRU KV offload to unified memory — RFC (conditional, default-off)
Run only if Stage 5 shows memory pressure unsolved by SnapKV. Thor 128GB unified LPDDR5x →
evicted attention-KV blocks live on the CPU-side of the unified pool, fetched at bus bandwidth
(~273 GB/s). LRU eviction; NEVER evict APC-cached prefix blocks (explicit guard). Flag
HYBRID_KV_OFFLOAD=1. Lossless gate: offloaded-KV attention == all-in-GPU, bitwise (eviction is
storage, not lossy). STATUS: RFC/skeleton — conditional on Stage 5; >300 LOC. Deferred.
