#!/usr/bin/env bash
set -euo pipefail

# One invocation owns one model process and has no automatic retry path.
if [[ $# -ne 4 ]]; then
  echo "usage: $0 VERSION BLOCK SNAPSHOT ADMISSION" >&2
  exit 2
fi

VERSION=$1
BLOCK=$2
SNAP=$3
ADMISSION=$4
ROOT=/mnt/data1/kt-round9
LOWER=$(tr '[:upper:]' '[:lower:]' <<<"$BLOCK" | tr -d '\n')
NAME="round17-profile-${LOWER}-v${VERSION}"
EVID="$ROOT/evidence/20260929-$NAME"
TRAINING=0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef
container_id=""

cleanup() {
  if [[ -n "$container_id" ]] && docker inspect "$container_id" >/dev/null 2>&1; then
    docker stop -t 10 "$container_id" >/dev/null 2>&1 || true
    docker inspect "$container_id" > "$EVID/container-final-inspect.json" 2>/dev/null || true
    docker rm "$container_id" > "$EVID/container-remove.txt" 2>&1 || true
  fi
}
trap cleanup EXIT

[[ "$(basename "$SNAP")" == "$NAME" ]]
[[ ! -e "$EVID" ]]
[[ ! -e "$SNAP/host-controller.json" ]]
[[ ! -e "$SNAP/execution-budget.lock" ]]
! docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"

ADMISSION_COMPAT="$(dirname "$ADMISSION")/result-compatible.json"
python3 - "$ADMISSION" "$ADMISSION_COMPAT" <<'PY'
import hashlib
import json
import pathlib
import sys
import time

source_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
source = json.loads(source_path.read_text())
expected_target = {
    "physical_id": 12,
    "card_id": 6,
    "chip_id": 0,
    "pci_bdf": "0000:85:00.0",
    "numa_nodes": [2, 3, 4, 7],
}
if source.get("status") != "PASS" or source.get("target") != expected_target:
    raise RuntimeError("reviewed admission result is not a physical12 PASS")
raw = source.get("raw_sha256")
if not isinstance(raw, dict) or set(raw) != {"round1.json", "round2.json"}:
    raise RuntimeError("reviewed admission lacks the exact two raw identities")
for name, expected in raw.items():
    actual = hashlib.sha256((source_path.parent / name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"reviewed admission raw identity changed: {name}")
compatible = dict(source)
compatible.update(
    admission_duration_contract="2x10s",
    rounds=["round1.json", "round2.json"],
    validated_at_epoch=time.time(),
)
with open(output_path, "x", encoding="utf-8") as handle:
    json.dump(compatible, handle, indent=2)
    handle.write("\n")
PY
ADMISSION="$ADMISSION_COMPAT"
ADMISSION_SHA=$(sha256sum "$ADMISSION" | awk '{print $1}')
PYTHONPATH="$SNAP" \
ROUND16_A1_PROFILE_ENABLE=1 \
ROUND16_A1_ADMISSION_SHA256="$ADMISSION_SHA" \
python3 "$SNAP/source/tools/launch_hybrid.py" \
  --version "$VERSION" --storage MXFP4 --placement static --resource-mode shared \
  --shared-training-container "$TRAINING" \
  --bank-receipt "$SNAP/rebound/bank-receipt.json" \
  --device-gate "$ROOT/evidence/20260920-c128-expert-v19/result.json" \
  --overlay-python "$SNAP/overlay-python" \
  --framework-profile "$SNAP/framework-profile.json" \
  --context-target 4096 \
  --context-reference-profile "$ROOT/evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json" \
  --streaming-profile "$SNAP/streaming-profile.json" \
  --cancel-trace --cancel-closure --cost-mode OFF \
  --cost-manifest "$SNAP/cost-manifest.json" \
  --cost-budget "$SNAP/execution-budget.json" \
  --cost-admission "$ADMISSION"

container_id=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["container_id"])' "$EVID/model-start.json")
deadline=$((SECONDS + 21600))
while (( SECONDS < deadline )); do
  docker inspect -f '{{.State.Running}}' "$container_id" | grep -Fxq true || {
    echo server-exited >&2
    exit 4
  }
  if [[ -f "$EVID/server.log" ]] && tail -c 131072 "$EVID/server.log" \
      | grep -Eq 'Uvicorn running on http://127.0.0.1:30091|The server is fired up and ready to roll!'; then
    break
  fi
  sleep 5
done
(( SECONDS < deadline )) || { echo readiness-timeout >&2; exit 5; }

docker stats --no-stream --format '{{json .}}' "$container_id" \
  > "$EVID/pre-driver-docker-stats.json"
docker exec "$container_id" bash -lc \
  "PYTHONPATH=/round9/$NAME:\$PYTHONPATH python /round9/$NAME/run_profile_driver.py --snapshot /round9/$NAME --output /out/kt-root-profile-result.json --block $BLOCK" \
  > "$EVID/kt-root-profile-driver.log" 2>&1

python3 - "$EVID/kt-root-profile-result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "PASS_KT_ROOT_CAUSE_PROFILE_REQUESTS"
assert result["generation_attempts"] == 6
print(result["status"], result["ended_at"] - result["started_at"])
PY

# The service writes raw evidence as root inside the bind-mounted /out.  Make
# it Host-readable before the owned container exits so archival hashing does
# not require a privileged or partial read path.
docker exec "$container_id" chmod -R a+rX /out
docker stop -t 10 "$container_id" >/dev/null
sleep 2
docker inspect "$container_id" > "$EVID/container-final-inspect.json"
docker rm "$container_id" > "$EVID/container-remove.txt"
container_id=""
trap - EXIT
npu-smi info -t proc-mem -i 6 -c 0 > "$EVID/post-proc-mem.txt"
find "$EVID" -type f -print0 | sort -z | xargs -0 sha256sum > "$EVID/SHA256SUMS"
echo "PASS_ROUND17_PROFILE_$BLOCK"
