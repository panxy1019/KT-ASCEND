#!/usr/bin/env bash
set -euo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_HEALTH_CHECK_GENERATION=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/results/venv-r6/bin/python "$(dirname "$0")/serial_command.py" "$@"
