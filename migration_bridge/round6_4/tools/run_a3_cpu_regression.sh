#!/usr/bin/env bash
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
py=/workspace/results/venv-r6/bin/python
old=/workspace/results/round6_2-prequalification/tools
tests=/workspace/round5-reference/legacy-tests/kt-kernel/test/per_commit
fixture=/workspace/round5-reference/legacy-tests/kt-kernel/test/fixtures/tiny_moe/create_tiny_moe_gguf_fixture.py
taskset -c 0-15 "$py" "$old/prepare_run_metadata.py" --mode defer0 \
  --output "$run_root/provenance.json" --artifact "$0" --artifact "$fixture" \
  --artifact "$tests/test_cpu_detect_metadata.py" \
  --artifact "$tests/test_llamafile_gguf_e2e.py" \
  --artifact "$tests/test_llamafile_routed_correctness.py" \
  --artifact "$tests/test_llamafile_cpuinfer_lifecycle.py"
# Same three historical private-API exclusions as Round5, not new relaxation.
exec "$py" "$old/run_owned_process.py" --output "$run_root/guard" --timeout-seconds 1800 -- \
  taskset -c 0-15 "$py" -m pytest -q -s -p no:cacheprovider \
  "$tests/test_cpu_detect_metadata.py" "$tests/test_llamafile_gguf_e2e.py" \
  "$tests/test_llamafile_routed_correctness.py" "$tests/test_llamafile_cpuinfer_lifecycle.py" \
  -k 'not create_load_forward_destroy_and_loader_path_isolation and not threadpool_and_numa_mapping_are_observable' \
  --junitxml="$run_root/junit.xml"
