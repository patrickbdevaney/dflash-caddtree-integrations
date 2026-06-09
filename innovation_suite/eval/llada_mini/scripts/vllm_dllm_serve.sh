#!/bin/bash
# vllm_dllm_serve.sh NAME MODEL PORT MEMUTIL
# Attempt vLLM block-diffusion serving of LLaDA2.1-mini via the dllm-plugin (real LLaDA2ForCausalLM,
# default) on our 0.20.0.dev0+dflash fork. Installs the plugin (deps=[], non-disruptive) at runtime.
source $HOME/dflash-dev/gpu_run.sh
NAME="$1"; MODEL="$2"; PORT="${3:-8011}"; MEMUTIL="${4:-0.4}"
LOG="/tmp/${NAME}.log"
IMG="${LLADA_IMG:-vllm-dflash-thor:ddtree}"
# strict serialization: refuse if a build or another serve is alive
ALIVE=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'sglk_build|sglang_|vllm_dllm_' | grep -v "^${NAME}$" || true)
if [ -n "$ALIVE" ]; then echo "[vllm_dllm_serve] ABORT: other memory-heavy containers alive: $ALIVE"; exit 1; fi
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 8g --memory 88g --memory-swap 88g -p ${PORT}:${PORT} \
  -v $HOME/models:/models -v $HOME/dflash-dev:/work \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e VLLM_PLUGINS=dllm -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} \
  -w /work \
  "$IMG" \
  bash -lc "git config --global --add safe.directory '*'; \
    /opt/venv/bin/pip install /work/dllm-plugin 2>&1 | tail -3; \
    FI=/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py; \
    sed -i 's/kv_cache_sf=kv_cache_sf,/**({\"kv_cache_sf\": kv_cache_sf} if kv_cache_sf is not None else {}),/g' \$FI && echo '[patch] flashinfer kv_cache_sf made conditional'; \
    /opt/venv/bin/vllm serve $MODEL \
      --trust-remote-code --max-model-len 2048 --max-num-seqs 4 \
      --attention-backend ${ATTN_BACKEND:-TRITON_ATTN} \
      --gpu-memory-utilization $MEMUTIL --enforce-eager --no-async-scheduling \
      --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
      --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker \
      --host 0.0.0.0 --port $PORT"
echo "[vllm_dllm_serve] gpu_run rc=$? (log: $LOG)"
