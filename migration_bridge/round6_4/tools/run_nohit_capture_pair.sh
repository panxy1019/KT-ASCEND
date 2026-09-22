#!/usr/bin/env bash
set -euo pipefail
run_root=${1:?fresh /workspace/results/r64-* root}
[[ "$run_root" == /workspace/results/r64-* ]] || exit 2
mkdir "$run_root"
py=/workspace/results/venv-r6/bin/python
tools=/workspace/results/round6_3-prequalification/tools
fixture=/workspace/results/r64-wp1-tools/whole_request_nohit.json
sha256sum "$fixture" "$0" "$tools/run_stateful_session.py" > "$run_root/input-sha256.txt"
# The retained capture launcher requires an r63-* prefix. These are new Round6.4
# runtime controls, not a rerun or promotion of the rejected Round6.3 contract.
for profile in D1_NODEFER D1; do
  "$py" "$tools/run_stateful_session.py" --profile "$profile" \
    --server-root "/workspace/results/r63-${run_root##*/}-$profile-server" \
    --output "$run_root/$profile" --requests "$fixture"
done
