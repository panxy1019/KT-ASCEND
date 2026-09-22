#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_APPLY_CONFIG_BACKUP=none
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/results/venv-r6/bin/python /workspace/r65a-tools/migration_bridge/round6_5a/tools/diagnostic_command.py "$@"
