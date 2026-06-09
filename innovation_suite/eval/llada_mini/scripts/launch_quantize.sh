#!/bin/bash
# launch_quantize.sh — NVFP4 quantize LLaDA2.1-mini inside vllm-dflash-thor.
# Installs llmcompressor at container runtime (ephemeral container; image untouched).
source $HOME/dflash-dev/gpu_run.sh
NAME=llada_quantize; LOG=/tmp/llada_quantize.log
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 16g \
  -v $HOME/models:/models \
  -v $HOME/dflash-dev:/work \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_OFFLINE=0 -e PYTHONUNBUFFERED=1 \
  -e SRC_MODEL=/models/LLaDA2.1-mini \
  -e DST_MODEL=/models/LLaDA2.1-mini-NVFP4 \
  -w /work/llada_mini \
  vllm-dflash-thor:latest \
  bash -lc "pip install -q --no-cache-dir 'llmcompressor>=0.8' compressed-tensors datasets 2>&1 | tail -3; /opt/venv/bin/python /work/llada_mini/quantize_nvfp4.py"
rc=$?
echo "[launch_quantize] gpu_run rc=$rc"
if [ $rc -eq 0 ]; then
  ec=$(gpu_wait "$NAME"); echo "[launch_quantize] $NAME exit=$ec"; gpu_stop "$NAME"
fi
