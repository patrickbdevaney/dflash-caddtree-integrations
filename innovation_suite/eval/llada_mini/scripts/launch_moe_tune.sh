#!/bin/bash
# V2: autotune the LLaDA2 fused-MoE GEMM for Thor sm_110a -> writes E=256,N=512,device_name=NVIDIA_Thor.json
# (the serve warned this was missing -> default MoE config -> "perf sub-optimal"). ALONE + capped.
source $HOME/dflash-dev/gpu_run.sh
NAME=moe_tune; LOG=/tmp/moe_tune.log
ALIVE=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'sglk_build|sglang_|vllm_dllm_|moe_tune' | grep -v "^${NAME}$" || true)
[ -n "$ALIVE" ] && { echo "ABORT: heavy container alive: $ALIVE"; exit 1; }
mkdir -p $HOME/dflash-dev/llada_mini/out/moe_configs
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 8g --memory 48g --memory-swap 48g \
  -v $HOME/models:/models -v $HOME/dflash-dev:/work \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e TORCH_CUDA_ARCH_LIST=11.0a -e PYTHONUNBUFFERED=1 \
  -w /work/llada_mini/out/moe_configs \
  vllm-dflash-thor:dllm \
  bash -lc "/opt/venv/bin/pip install -q ray 2>&1 | tail -2; \
    BM=/build/vllm/benchmarks/kernels/benchmark_moe.py; \
    sed -i 's/        \"Qwen3NextForCausalLM\",/        \"Qwen3NextForCausalLM\",\n        \"LLaDA2MoeModelLM\",/' \$BM && echo '[patch] added LLaDA2MoeModelLM to get_model_params'; \
    /opt/venv/bin/python \$BM --model /models/LLaDA2.1-mini --trust-remote-code --tune --tp-size 1 --dtype auto ${MOE_BATCHES:+--batch-size ${MOE_BATCHES}}"
rc=$?; echo "[moe_tune] gpu_run rc=$rc"
if [ $rc -eq 0 ]; then ec=$(gpu_wait "$NAME"); echo "[moe_tune] exit=$ec"; gpu_stop "$NAME"; fi
echo "=== produced configs ==="; ls -la $HOME/dflash-dev/llada_mini/out/moe_configs/ 2>/dev/null | grep -i json | head
