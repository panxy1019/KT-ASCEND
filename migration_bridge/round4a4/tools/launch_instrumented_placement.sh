#!/usr/bin/env bash
set -eo pipefail

profile=${1:?usage: launch_instrumented_placement.sh PROFILE GGUF LAYERS RUN_ROOT [PORT]}
gguf=${2:?usage: launch_instrumented_placement.sh PROFILE GGUF LAYERS RUN_ROOT [PORT]}
layers=${3:?usage: launch_instrumented_placement.sh PROFILE GGUF LAYERS RUN_ROOT [PORT]}
run_root=${4:?usage: launch_instrumented_placement.sh PROFILE GGUF LAYERS RUN_ROOT [PORT]}
port=${5:-31000}
export SGLANG_KT_NUMERICAL_DUMP_DIR="${run_root}/routes"
export SGLANG_KT_NUMERICAL_DUMP_LAYERS="${layers}"
export SGLANG_KT_NUMERICAL_DUMP_MAX_PASSES=${SGLANG_KT_NUMERICAL_DUMP_MAX_PASSES:-4096}
export SGLANG_KT_NUMERICAL_DUMP_ARM_FILE="${run_root}/dump.arm"
mkdir -p "${run_root}/routes" "${run_root}/tensors"

script_dir=$(cd "$(dirname "$0")" && pwd)
"${script_dir}/launch_quality_placement.sh" "${profile}" "${gguf}" "${run_root}" "${port}" \
  --debug-tensor-dump-output-folder "${run_root}/tensors" \
  --debug-tensor-dump-layers 999
