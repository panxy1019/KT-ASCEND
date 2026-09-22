#!/usr/bin/env bash
set -euo pipefail
run_root=${1:?fresh /workspace/results/r64-* root}
[[ "$run_root" == /workspace/results/r64-* ]] || exit 2
mkdir "$run_root"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
py=/workspace/results/venv-r6/bin/python
old=/workspace/results/round6_2-prequalification/tools
here=/workspace/results/r64-wp1-tools
taskset -c 0-15 "$py" "$old/prepare_run_metadata.py" --mode defer0 \
  --output "$run_root/provenance.json" --artifact "$0" \
  --artifact "$here/run_real_expert_safety.py" --artifact "$here/verify_real_expert_identity.py" \
  --artifact /workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf \
  --artifact /workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json
exec "$py" "$old/run_owned_process.py" --output "$run_root/guard" --timeout-seconds 10800 -- \
  taskset -c 0-15 "$py" "$here/run_real_expert_safety.py" \
  --probe "$here/verify_real_expert_identity.py" --output "$run_root/cases"
