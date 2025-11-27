#!/usr/bin/env bash
set -euo pipefail

# Default NGC PyTorch image.
# 25.10 is a good NVFP4-era baseline and still Blackwell-optimized.
# Override with:
#   NGC_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3 ./run_ngc.sh
IMAGE_DEFAULT="nvcr.io/nvidia/pytorch:25.10-py3"
IMAGE="${NGC_IMAGE:-$IMAGE_DEFAULT}"

# Resolve repo root (directory containing this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Starting NGC PyTorch container"
echo "    Image:  ${IMAGE}"
echo "    Mount:  ${REPO_DIR} -> /workspace/k5"
echo "    Extra:  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
echo

docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${REPO_DIR}":/workspace/k5 \
  -w /workspace/k5 \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  bash

