#!/bin/sh
set -eu
# CTranslate2, Paddle and Torch use the same pip-installed CUDA runtime libraries.
CUDA_LIBRARY_DIRS=$(python -c 'import sysconfig,pathlib; p=pathlib.Path(sysconfig.get_path("purelib"))/"nvidia"; print(":".join(str(x) for x in p.glob("*/lib")))')
export LD_LIBRARY_PATH="${CUDA_LIBRARY_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME=/workspace/runtime/cache
export PADDLE_PDX_CACHE_HOME=/workspace/runtime/cache/paddlex
export HF_HOME=/workspace/runtime/cache/huggingface
export JUPYTER_CONFIG_DIR=/workspace/runtime/jupyter/config
export JUPYTER_DATA_DIR=/workspace/runtime/jupyter/data
export JUPYTER_RUNTIME_DIR=/workspace/runtime/jupyter/runtime
export IPYTHONDIR=/workspace/runtime/ipython
mkdir -p "$XDG_CACHE_HOME"
exec "$@"
