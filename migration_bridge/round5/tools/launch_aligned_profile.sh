#!/usr/bin/env bash
set -euo pipefail

profile=${1:?usage: launch_aligned_profile.sh P1|P2|P3 RUN_ROOT [PORT]}
run_root=${2:?usage: launch_aligned_profile.sh P1|P2|P3 RUN_ROOT [PORT]}
port=${3:-31000}

case "${profile}" in
  P1)
    placement=/workspace/results/placement_p1.pt
    gpu_ratio=0.9975961538461539
    dump_layers=17
    ;;
  P2)
    placement=/workspace/results/placement_p2.pt
    gpu_ratio=0.9903846153846154
    dump_layers=1,9,17,26
    ;;
  P3)
    placement=/workspace/results/placement_p3.pt
    gpu_ratio=0.9807692307692307
    dump_layers=1,5,8,12,17,19,22,26
    ;;
  *)
    echo "profile must be P1, P2, or P3; got ${profile}" >&2
    exit 2
    ;;
esac

model=${KT_R5_MODEL_PATH:-/workspace/models/DeepSeek-V2-Lite-604d5664}
gguf=${KT_R5_GGUF_PATH:-/workspace/round4a5-reference/deepseek-v2-lite-p3-f32.gguf}
mkdir -p "${run_root}" "${run_root}/tensors" "${run_root}/routes"

# Vendor setup scripts probe optional shell variables (for example
# ``ZSH_VERSION``) without guarding them.  Temporarily relax nounset while
# sourcing, then restore this launcher's strict mode.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:/Megatron-Bridge/src:${PYTHONPATH:-}"

extra=()
if [[ ${KT_R5_ENABLE_TENSOR_DUMP:-0} == 1 ]]; then
  extra+=(
    --debug-tensor-dump-output-folder "${run_root}/tensors"
    --debug-tensor-dump-layers 999
  )
fi

if [[ ${KT_R5_ENABLE_ROUTE_DUMP:-0} == 1 ]]; then
  export SGLANG_KT_NUMERICAL_DUMP_DIR="${run_root}/routes"
  export SGLANG_KT_NUMERICAL_DUMP_LAYERS="${dump_layers}"
  export SGLANG_KT_NUMERICAL_DUMP_MAX_PASSES="${SGLANG_KT_NUMERICAL_DUMP_MAX_PASSES:-4096}"
  export SGLANG_KT_NUMERICAL_DUMP_ARM_FILE="${run_root}/dump.arm"
fi

exec python -m sglang.launch_server \
  --model-path "${model}" \
  --host 127.0.0.1 --port "${port}" --device npu --tp-size 1 \
  --dtype bfloat16 --context-length 512 --max-total-tokens 512 \
  --chunked-prefill-size 512 --max-prefill-tokens 512 \
  --mem-fraction-static 0.55 --max-running-requests 1 --random-seed 0 \
  --disable-cuda-graph --disable-custom-all-reduce --skip-server-warmup \
  --weight-loader-disable-mmap --attention-backend ascend --sampling-backend pytorch \
  --kt-weight-path "${gguf}" --kt-method LLAMAFILE \
  --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-numa-nodes 0 \
  --kt-gpu-experts-ratio "${gpu_ratio}" --kt-max-deferred-experts-per-token 0 \
  --kt-expert-placement-strategy frequency --init-expert-location "${placement}" \
  "${extra[@]}"
