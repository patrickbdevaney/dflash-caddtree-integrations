#!/bin/bash
# sglang_serve.sh NAME MODEL PORT MEMFRAC BACKEND [extra dllm-algorithm-config json]
# Launches an SGLang dLLM server (JointThreshold) for LLaDA2.1-mini via the gpu_run guard,
# publishing PORT to host. Returns once the container is running; caller waits for the ready
# marker in the log, benchmarks, then calls gpu_stop NAME.
source $HOME/dflash-dev/gpu_run.sh
NAME="$1"; MODEL="$2"; PORT="$3"; MEMFRAC="${4:-0.6}"; BACKEND="${5:-flashinfer}"; ALGCFG="$6"
LOG="/tmp/${NAME}.log"
IMG="${LLADA_IMG:-vllm-dflash-thor:sglang}"
ALG_ARG=""; [ -n "$ALGCFG" ] && ALG_ARG="--dllm-algorithm-config $ALGCFG"
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 16g -p ${PORT}:${PORT} \
  -v $HOME/models:/models \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e TORCH_CUDA_ARCH_LIST=11.0a -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  "$IMG" \
  /opt/venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --dllm-algorithm JointThreshold $ALG_ARG \
    --tp-size 1 --trust-remote-code \
    --mem-fraction-static "$MEMFRAC" \
    --attention-backend "$BACKEND" \
    --max-running-requests 1 \
    --host 0.0.0.0 --port "$PORT"
echo "[sglang_serve] gpu_run rc=$? (log: $LOG)"
