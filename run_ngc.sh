#!/usr/bin/env bash
set -e

# Default NGC image; user can override via NGC_IMAGE=...
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.10-py3}"

# Resolve repo root on the host
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${REPO_ROOT}":/workspace/k5 \
  -w /workspace/k5 \
  "${NGC_IMAGE}" \
  bash -lc "
    set -e
    echo '>>> Inside NGC container:' \$(python -c 'import torch; print(torch.__version__)')
    echo '>>> Installing Python dependencies from requirements-core.txt (if present)...'
    if [ -f requirements-core.txt ]; then
      python -m pip install --upgrade pip
      pip install -r requirements-core.txt
    else
      echo '    requirements-core.txt not found, skipping.'
    fi

    echo '>>> Ensuring TorchAO is NOT present (Transformers+TorchAO mismatch workaround)...'
    pip uninstall -y torchao >/dev/null 2>&1 || true
    echo \">>> This is the preferred workaround for torchao since we're not using quantization...\"

    echo '>>> Environment ready. Dropping into interactive shell in /workspace/k5'
    exec bash
  "

