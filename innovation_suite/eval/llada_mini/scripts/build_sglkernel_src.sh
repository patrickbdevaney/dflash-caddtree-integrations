#!/bin/bash
# Build sgl-kernel from source against the image's torch 2.10.0 (sm_110a), then install sglang
# 0.5.9 --no-deps. Commits to vllm-dflash-thor:sglang2. CPU compile (no GPU needed for nvcc).
set -e
NAME=sglk_build
docker rm -f "$NAME" >/dev/null 2>&1 || true
# run build in a container with the v0.5.9 source mounted; keep container, then commit.
docker run --name "$NAME" \
  -v $HOME/dflash-dev/sglang-src:/src \
  -e TORCH_CUDA_ARCH_LIST=11.0a \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e CMAKE_BUILD_PARALLEL_LEVEL=8 \
  -e MAX_JOBS=8 \
  vllm-dflash-thor:ddtree \
  bash -lc '
set -x
P=/opt/venv/bin/pip
$P install --no-cache-dir cmake ninja scikit-build-core 2>&1 | tail -2
cd /src/sgl-kernel
echo "=== building sgl-kernel from source (sm_110a, torch 2.10.0) ==="
$P install . --no-build-isolation --no-cache-dir 2>&1 | tail -40
echo "=== install sglang 0.5.9 --no-deps + orjson ==="
$P install --no-cache-dir "sglang==0.5.9" --no-deps --extra-index-url https://pypi.org/simple/ 2>&1 | tail -4
$P install --no-cache-dir orjson 2>&1 | tail -2
$P show torch sgl-kernel sglang 2>/dev/null | grep -E "^Name|^Version"
echo "SGLK_BUILD_DONE"
'
rc=$?
echo "[build_sglkernel] container exit=$rc"
if [ $rc -eq 0 ]; then
  docker commit "$NAME" vllm-dflash-thor:sglang2 >/dev/null && echo "committed vllm-dflash-thor:sglang2"
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
