#!/usr/bin/env bash
set -euo pipefail
entry=${1:?tool filename}
shift
case "$entry" in
  run_batch_capture.py|run_quality_dev.py|run_d1_lifecycle.py) ;;
  *) echo 'Unsupported capacity test'; exit 2 ;;
esac
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
exec taskset -c 0-15 /workspace/results/venv-r6/bin/python "/workspace/results/r64-capacity-tools-v1/$entry" "$@"
