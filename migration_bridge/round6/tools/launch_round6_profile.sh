#!/usr/bin/env bash
set -euo pipefail

profile=${1:?usage: launch_round6_profile.sh ALLNPU|D1|D2|D3 RUN_ROOT [PORT]}
run_root=${2:?usage: launch_round6_profile.sh ALLNPU|D1|D2|D3 RUN_ROOT [PORT]}
port=${3:-31000}

case "${profile}" in
  ALLNPU)
    placement=
    gpu_ratio=
    max_deferred=0
    ;;
  D1)
    placement=${KT_R6_D1_PLACEMENT:-/workspace/results/placements/placement_d1.pt}
    gpu_ratio=0.9951923076923077
    max_deferred=1
    ;;
  D2)
    placement=${KT_R6_D2_PLACEMENT:-/workspace/results/placements/placement_d2.pt}
    gpu_ratio=0.9903846153846154
    max_deferred=1
    ;;
  D3)
    placement=${KT_R6_D3_PLACEMENT:-/workspace/results/placements/placement_d3.pt}
    gpu_ratio=0.9807692307692307
    max_deferred=1
    ;;
  *)
    echo "profile must be ALLNPU, D1, D2, or D3; got ${profile}" >&2
    exit 2
    ;;
esac

python_bin=${KT_R6_PYTHON:-/workspace/results/venv-r6/bin/python}
model=${KT_R6_MODEL_PATH:-/workspace/models/DeepSeek-V2-Lite-604d5664}
gguf=${KT_R6_GGUF_PATH:-/workspace/round5-reference/deepseek-v2-lite-p3-f32.gguf}
mkdir -p "${run_root}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"

common=(
  --model-path "${model}"
  --host 127.0.0.1 --port "${port}" --device npu --tp-size 1
  --dtype bfloat16 --context-length 512 --max-total-tokens 512
  --chunked-prefill-size 512 --max-prefill-tokens 512
  --mem-fraction-static 0.55 --max-running-requests 1 --random-seed 0
  --disable-cuda-graph --disable-custom-all-reduce --skip-server-warmup
  --weight-loader-disable-mmap --attention-backend ascend --sampling-backend pytorch
)

if [[ ${KT_R6_ENABLE_ROUTE_RECORDER:-0} == 1 ]]; then
  common+=(
    --expert-distribution-recorder-mode stat
    --expert-distribution-recorder-buffer-size -1
    --enable-return-routed-experts
    --disable-radix-cache
  )
fi

if [[ ${KT_R6_ENABLE_TENSOR_DUMP:-0} == 1 ]]; then
  mkdir -p "${run_root}/tensors"
  common+=(
    --debug-tensor-dump-output-folder "${run_root}/tensors"
    --debug-tensor-dump-layers 999
  )
fi

hybrid=()
if [[ ${profile} != ALLNPU ]]; then
  [[ -f ${placement} ]] || { echo "missing placement: ${placement}" >&2; exit 3; }
  [[ -f ${gguf} ]] || { echo "missing GGUF: ${gguf}" >&2; exit 3; }
  hybrid=(
    --kt-weight-path "${gguf}" --kt-method LLAMAFILE
    --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-numa-nodes 0
    --kt-gpu-experts-ratio "${gpu_ratio}"
    --kt-max-deferred-experts-per-token "${max_deferred}"
    --kt-expert-placement-strategy frequency
    --init-expert-location "${placement}"
  )
fi

exec "${python_bin}" -m sglang.launch_server "${common[@]}" "${hybrid[@]}"
