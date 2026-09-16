#!/usr/bin/env bash
# Retained real Layer17/Expert8 regression; no production edits.
set -euo pipefail
run_root=${1:?fresh /workspace/results/r64-* directory}
[[ "$run_root" == /workspace/results/r64-* ]] || exit 2
mkdir "$run_root"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
export KT_R6_ENABLE_ROUTE_RECORDER=0 KT_R6_ENABLE_TENSOR_DUMP=0
unset KT_R61_EXPLICIT_REPLAY KT_R63_FORCE_MANIFEST KT_R63_FORCE_ARM_FILE KT_R63_FORCE_TRACE
py=/workspace/results/venv-r6/bin/python
old=/workspace/results/round6_2-prequalification/tools
tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test_file=/workspace/round5-reference/tools/round3/verify_real_single_layer_hybrid.py
gguf=/workspace/round5-reference/deepseek-v2-lite-p2-f32.gguf
taskset -c 0-15 "$py" "$old/prepare_run_metadata.py" --mode defer0 \
  --output "$run_root/provenance.json" --artifact "$0" --artifact "$test_file" --artifact "$tool_dir/run_owned_process.py" \
  --artifact "$gguf" --artifact /workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json
exec "$py" "$tool_dir/run_owned_process.py" --output "$run_root/guard" --timeout-seconds 1800 -- \
  taskset -c 0-15 "$py" "$test_file" \
  --model-dir /workspace/models/DeepSeek-V2-Lite-604d5664 \
  --gguf "$gguf" --layer 17 --cpu-expert 8 --output "$run_root/round3.json"
