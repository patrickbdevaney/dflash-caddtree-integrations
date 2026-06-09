#!/bin/bash
# launch_bench.sh NAME LOGFILE -- <python args...>
# Wraps gpu_run: runs bench_s2d2.py inside vllm-dflash-thor with model+repo mounts.
source $HOME/dflash-dev/gpu_run.sh
NAME="$1"; LOG="$2"; shift 2; [ "$1" = "--" ] && shift
PYARGS="$@"
IMG="${LLADA_IMG:-vllm-dflash-thor:latest}"
gpu_run "$NAME" "$LOG" -- \
  --runtime nvidia --ipc host --shm-size 16g \
  -v $HOME/models:/models \
  -v $HOME/dflash-dev:/work \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e S2D2_LLADA=/work/S2D2/LLaDA2 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -w /work/S2D2/LLaDA2 \
  "$IMG" \
  /opt/venv/bin/python /work/llada_mini/bench_s2d2.py $PYARGS
rc=$?
echo "[launch_bench] gpu_run rc=$rc for $NAME"
if [ $rc -eq 0 ]; then
  echo "[launch_bench] waiting for $NAME to finish..."
  ec=$(gpu_wait "$NAME")
  echo "[launch_bench] $NAME exit=$ec"
  gpu_stop "$NAME"
fi
