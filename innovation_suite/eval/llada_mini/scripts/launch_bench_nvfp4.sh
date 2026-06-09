#!/bin/bash
# NVFP4 spot-check: install compressed-tensors at runtime, run bench on the NVFP4 model.
source $HOME/dflash-dev/gpu_run.sh
NAME=llada_nvfp4_bench; LOG=/tmp/llada_nvfp4_bench.log
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 16g \
  -v $HOME/models:/models -v $HOME/dflash-dev:/work \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e S2D2_LLADA=/work/S2D2/LLaDA2 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 -w /work/S2D2/LLaDA2 \
  vllm-dflash-thor:latest \
  bash -lc "pip install -q --no-cache-dir 'compressed-tensors' 2>&1 | tail -1; /opt/venv/bin/python /work/llada_mini/bench_s2d2.py $*"
rc=$?
echo "[launch_bench_nvfp4] gpu_run rc=$rc"
if [ $rc -eq 0 ]; then ec=$(gpu_wait "$NAME"); echo "[launch_bench_nvfp4] $NAME exit=$ec"; gpu_stop "$NAME"; fi
