#!/bin/bash
source /home/patrickd/dflash-dev/gpu_run.sh
exec >> /home/patrickd/dflash-dev/longppl_status.log 2>&1
echo "================ LONGPPL2_START $(date) MODE=${RUN_MODE:-smoke} ================"
DISC_CTX=${DISC_CTX:-4096}; EVAL_CTX=${EVAL_CTX:-8192}; NATIVE=${NATIVE:-4096}; SHORTK=${SHORTK:-2048}
COMMON_V="-v /home/patrickd/dflash-dev/tests:/tests -v /home/patrickd/dflash-dev/baseline_outputs:/out \
 -v /home/patrickd/thor-vllm-cache/vllm-dflash:/root/.cache/vllm -v /home/patrickd/thor-vllm-cache/flashinfer:/root/.cache/flashinfer"
COMMON_E="-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 -e HF_HUB_DISABLE_XET=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CUDA_DEVICE_MAX_CONNECTIONS=1"
GPU="--runtime nvidia --gpus all --ipc=host --shm-size=16g"
IMG="--entrypoint /opt/venv/bin/python vllm-dflash-thor:ddtree /tests/longppl.py"

echo "--- Phase A: Llama-8B discriminator (ctx=$DISC_CTX) ---"
gpu_run lppl_disc /tmp/lppl_disc.log -- $GPU $COMMON_E $COMMON_V \
  -e LPPL_MODE=disc -e LPPL_DISC_CTX=$DISC_CTX -e LPPL_SHORT_K=$SHORTK \
  -v /home/patrickd/models/Llama-3.1-8B-Instruct-NVFP4:/disc:ro $IMG
rc=$?
if [ $rc -eq 2 ]; then echo "ABORT: disc WEDGED (reboot needed)"; echo "LONGPPL2_DONE $(date)"; exit 2; fi
if [ $rc -ne 0 ]; then echo "ABORT: disc launch failed rc=$rc"; echo "LONGPPL2_DONE $(date)"; exit 1; fi
echo "disc exit=$(gpu_wait lppl_disc)"; grep -E "DISC|Error|Traceback|modelopt" /tmp/lppl_disc.log 2>/dev/null | grep -vE "non-default" | tail -6
gpu_stop lppl_disc

echo "--- Phase B: Qwen3.6 DroPE eval (ctx=$EVAL_CTX native=$NATIVE) ---"
gpu_run lppl_eval /tmp/lppl_eval.log -- $GPU $COMMON_E $COMMON_V \
  -e LPPL_MODE=eval -e LPPL_EVAL_CTX=$EVAL_CTX -e LPPL_NATIVE=$NATIVE -e DFLASH_DROPE=1 -e DFLASH_DROPE_NATIVE=$NATIVE -e DFLASH_DROPE_MAX=$EVAL_CTX -e LPPL_TAG=drope \
  -v /home/patrickd/Qwen3.6-35B-A3B-NVFP4:/model:ro -v /home/patrickd/Qwen3.6-35B-A3B-DFlash:/drafter:ro $IMG
rc=$?
if [ $rc -ne 0 ]; then echo "ABORT: eval launch rc=$rc"; echo "LONGPPL2_DONE $(date)"; exit $rc; fi
echo "eval exit=$(gpu_wait lppl_eval)"; grep -E "EVAL|LONGPPL_RESULT|Error|Traceback" /tmp/lppl_eval.log 2>/dev/null | grep -vE "non-default" | tail -6
gpu_stop lppl_eval
echo "================ LONGPPL2_DONE $(date) ================"
