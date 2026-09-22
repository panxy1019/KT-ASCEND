#!/usr/bin/env bash
# Run inside the dedicated container. No host writes or process-name killing.
set -euo pipefail
run_root=${1:?fresh /workspace/results/r64-* directory}
[[ "$run_root" == /workspace/results/r64-* ]] || exit 2
mkdir "$run_root"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
export KT_R6_ENABLE_ROUTE_RECORDER=0 KT_R6_ENABLE_TENSOR_DUMP=0
export KT_R62_COMPLETE_FORWARD_DUMP=0
unset KT_R61_EXPLICIT_REPLAY KT_R63_FORCE_MANIFEST KT_R63_FORCE_ARM_FILE KT_R63_FORCE_TRACE
py=/workspace/results/venv-r6/bin/python
legacy=/workspace/results/round6_2-prequalification/tools
taskset -c 0-15 "$py" "$legacy/prepare_run_metadata.py" \
  --mode defer1 --output "$run_root/provenance.json" \
  --artifact "$0" \
  --artifact /workspace/kt-src/kt-kernel/test/round5/test_ascend_framework_transfer.py \
  --artifact /workspace/kt-src/kt-kernel/test/round6/test_deferred_selection.py \
  --artifact /workspace/kt-src/third_party/sglang/test/registered/ascend/test_kt_ep_alignment_contracts.py \
  --artifact /workspace/results/placements/placement_d1.pt \
  --artifact /workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf \
  --artifact /workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json
exec "$py" "$legacy/run_owned_process.py" --output "$run_root/guard" --timeout-seconds 1800 -- \
  taskset -c 0-15 "$py" -m pytest -q \
  /workspace/kt-src/kt-kernel/test/round5/test_ascend_framework_transfer.py \
  /workspace/kt-src/kt-kernel/test/round6/test_deferred_selection.py \
  /workspace/kt-src/third_party/sglang/test/registered/ascend/test_kt_ep_alignment_contracts.py \
  --junitxml="$run_root/junit.xml"
