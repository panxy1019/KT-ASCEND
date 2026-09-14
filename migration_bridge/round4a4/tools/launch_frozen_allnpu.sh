#!/usr/bin/env bash
set -eo pipefail

run_root=${1:?usage: launch_frozen_allnpu.sh RUN_ROOT [PORT]}
port=${2:-31000}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/build/r3-python:/workspace/kt-src/third_party/sglang/python:/workspace/kt-src/third_party/llama.cpp/gguf-py:${PYTHONPATH:-}
mkdir -p "${run_root}/tensors"

exec python -m sglang.launch_server \
  --model-path /workspace/models/DeepSeek-V2-Lite-604d5664 \
  --host 127.0.0.1 --port "${port}" --device npu --tp-size 1 \
  --dtype bfloat16 --context-length 512 --max-total-tokens 512 \
  --chunked-prefill-size 512 --max-prefill-tokens 512 \
  --mem-fraction-static 0.55 --max-running-requests 1 --random-seed 0 \
  --disable-cuda-graph --disable-custom-all-reduce --skip-server-warmup \
  --weight-loader-disable-mmap --attention-backend ascend --sampling-backend pytorch \
  --debug-tensor-dump-output-folder "${run_root}/tensors" \
  --debug-tensor-dump-layers 999
