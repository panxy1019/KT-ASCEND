#!/usr/bin/env bash
set -euo pipefail
root=${1:?run root}
profile=${2:?profile json}
workers=${3:-32}
[[ $root == /workspace/results/r65-* ]] || exit 2
[[ $workers == 16 || $workers == 32 ]] || exit 2
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/results/venv-r6/bin/python /workspace/r65-tools/migration_bridge/round6_5/tools/static_command.py "$root" "$profile" "$workers"
