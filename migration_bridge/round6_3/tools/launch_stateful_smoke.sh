#!/usr/bin/env bash
set -euo pipefail
profile=${1:?D1 or D1_NODEFER}
run_root=${2:?fresh absolute run root}
force_manifest=${3:-}
case "$profile" in
  D1) mode=defer1 ;;
  D1_NODEFER) mode=defer0 ;;
  *) exit 2 ;;
esac
[[ "$run_root" == /workspace/results/r63-* ]] || exit 3
mkdir "$run_root"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export KT_R6_ENABLE_ROUTE_RECORDER=1 KT_R6_ENABLE_TENSOR_DUMP=1
export KT_R62_COMPLETE_FORWARD_DUMP=1
export KT_R63_CAPTURE_INDEX="$run_root/index.jsonl"
export KT_R62_RUN_METADATA="$run_root/provenance.json"
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:/workspace/results/round6_2-prequalification/tools
unset KT_R62_REFERENCE_LOGITS_ONLY KT_R63_FORCE_MANIFEST KT_R63_FORCE_ARM_FILE KT_R63_FORCE_TRACE KT_R63_FORCE_MANIFEST_SHA256
case "${KT_R63_OBSERVER_PROFILE:-full}" in
  full) ;;
  reference)
    export KT_R62_COMPLETE_FORWARD_DUMP=0 KT_R62_REFERENCE_LOGITS_ONLY=1
    export KT_R63_COMPACT_CAPTURE=0
    unset KT_R63_CAPTURE_INDEX
    ;;
  off)
    [[ -z "$force_manifest" ]] || { echo 'OFF cannot force history'; exit 4; }
    export KT_R6_ENABLE_TENSOR_DUMP=0 KT_R62_COMPLETE_FORWARD_DUMP=0
    unset KT_R63_CAPTURE_INDEX KT_R63_COMPACT_CAPTURE
    ;;
  *) echo 'Invalid observer profile'; exit 4 ;;
esac
if [[ -n "$force_manifest" ]]; then
  export KT_R63_FORCE_MANIFEST="$force_manifest"
  KT_R63_FORCE_MANIFEST_SHA256=$(sha256sum "$force_manifest")
  export KT_R63_FORCE_MANIFEST_SHA256=${KT_R63_FORCE_MANIFEST_SHA256%% *}
  export KT_R63_FORCE_ARM_FILE="$run_root/armed.sha256"
  export KT_R63_FORCE_TRACE="$run_root/forcing.jsonl"
fi
py=/workspace/results/venv-r6/bin/python
taskset -c 0-15 "$py" /workspace/results/round6_2-prequalification/tools/prepare_run_metadata.py \
  --mode "$mode" --output "$KT_R62_RUN_METADATA" \
  --artifact /workspace/kt-src/third_party/sglang/python/sglang/srt/model_executor/model_runner.py \
  --artifact /workspace/kt-src/third_party/sglang/python/sglang/srt/debug_utils/round63_teacher_force.py \
  --artifact /workspace/results/placements/placement_d1.pt \
  --artifact /workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf \
  --artifact /workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json \
  --artifact /workspace/results/round6_3-prequalification/tools/launch_stateful_smoke.sh \
  --artifact /workspace/results/round6_3-prequalification/tools/run_stateful_probe.py
"$py" -c 'import json,sys; p=json.load(open(sys.argv[1]))["file_sha256"]; expected={"/workspace/results/placements/placement_d1.pt":"23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c", "/workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf":"4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f", "/workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json":"41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"}; assert all(p[k]==v for k,v in expected.items()), "R63_ARTIFACT_DRIFT"' "$KT_R62_RUN_METADATA"
exec "$py" /workspace/results/round6_2-prequalification/tools/run_owned_process.py \
  --output "$run_root/guard" --timeout-seconds 3600 -- \
  taskset -c 0-15 bash /workspace/kt-src/migration_bridge/round6/tools/launch_round6_profile.sh \
  "$profile" "$run_root" 31000
