#!/usr/bin/env bash
set -eo pipefail

profile=${1:?usage: launch_quality_placement.sh PROFILE GGUF RUN_ROOT [PORT]}
gguf=${2:?usage: launch_quality_placement.sh PROFILE GGUF RUN_ROOT [PORT]}
run_root=${3:?usage: launch_quality_placement.sh PROFILE GGUF RUN_ROOT [PORT]}
port=${4:-31000}
shift $(( $# >= 4 ? 4 : $# ))
case "${profile}" in
  p1) ratio=0.9975961538461539 ;;
  p2) ratio=0.9903846153846154 ;;
  p3) ratio=0.9807692307692307 ;;
  *) echo "unsupported frozen profile: ${profile}" >&2; exit 2 ;;
esac
# Default to the frozen profile.  Root-cause experiments may override both
# values together through the environment; this is diagnostic-only and keeps
# every normal P1/P2/P3 invocation unchanged.
placement=${KT_PLACEMENT_PATH:-/workspace/kt-src/migration_bridge/round4a/placements/placement_${profile}.pt}
ratio=${KT_GPU_EXPERTS_RATIO:-${ratio}}
cpuinfer_threads=${KT_CPUINFER_THREADS:-16}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/build/r3-python:/workspace/kt-src/third_party/sglang/python:/workspace/kt-src/third_party/llama.cpp/gguf-py:${PYTHONPATH:-}
mkdir -p "${run_root}"

exec python -m sglang.launch_server \
  --model-path /workspace/models/DeepSeek-V2-Lite-604d5664 \
  --host 127.0.0.1 --port "${port}" --device npu --tp-size 1 \
  --dtype bfloat16 --context-length 512 --max-total-tokens 512 \
  --chunked-prefill-size 512 --max-prefill-tokens 512 \
  --mem-fraction-static 0.55 --max-running-requests 1 --random-seed 0 \
  --disable-cuda-graph --disable-custom-all-reduce --skip-server-warmup \
  --weight-loader-disable-mmap --attention-backend ascend --sampling-backend pytorch \
  --kt-weight-path "${gguf}" --kt-method LLAMAFILE --kt-cpuinfer "${cpuinfer_threads}" \
  --kt-threadpool-count 1 --kt-numa-nodes 0 \
  --kt-gpu-experts-ratio "${ratio}" --kt-max-deferred-experts-per-token 0 \
  --kt-expert-placement-strategy frequency --init-expert-location "${placement}" \
  "$@"
