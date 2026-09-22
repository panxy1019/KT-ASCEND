#!/usr/bin/env bash
set -euo pipefail
run_root=${1:?fresh /workspace/results/r64-* root}
[[ "$run_root" == /workspace/results/r64-* ]] || exit 2
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/kt-src/third_party/sglang/python:/workspace/kt-src
py=/workspace/results/venv-r6/bin/python
here=/workspace/results/r64-runtime-tools-v1
# Do not launch the long matrix if either missing WP1 gate failed.
"$py" -c 'import json; from pathlib import Path; base=Path("/workspace/results"); assert json.loads((base/"r64-nohit-v1/whole-request-audit.json").read_text())["status"] == "WHOLE_REQUEST_NOHIT_PASS"; r=json.loads((base/"r64-expert-safety-v1/cases/report.json").read_text()); assert r["status"] == "EXPERT_SAFETY_PASS" and len(r["cases"]) == 16; assert json.loads((base/"r64-expert-safety-v1/guard/result.json").read_text())["reason"] == "EXIT_ZERO"'
contract_sha=$(sha256sum "$here/APPROXIMATE_ACCELERATION_CONTRACT.json")
exec taskset -c 0-15 "$py" "$here/run_d1_runtime.py" \
  --corpus "$here/s6.json" --contract "$here/APPROXIMATE_ACCELERATION_CONTRACT.json" \
  --contract-sha256 "${contract_sha%% *}" \
  --provenance /workspace/results/r64-preflight-v1/provenance.json --output "$run_root"
