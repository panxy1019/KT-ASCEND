#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/data1/kt-round9"
readonly SNAPSHOT="$ROOT/round16-s0-perf-a-v403"
readonly EVIDENCE="$ROOT/evidence/20260929-round16-s0-perf-a-v403"
readonly NAME="kt-round16-s0-perf-a-v403"
readonly TRAINING_ID="0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
readonly ADMISSION="$1"

test -f "$ADMISSION"
test ! -e "$EVIDENCE"
test ! -e "$SNAPSHOT/host-controller.json"
test ! -e "$SNAPSHOT/execution-budget.lock"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "refusing to replace existing container $NAME" >&2
  exit 3
fi

container_id=""
cleanup() {
  if [[ -n "$container_id" ]] && docker inspect "$container_id" >/dev/null 2>&1; then
    docker stop -t 10 "$container_id" >/dev/null 2>&1 || true
    docker inspect "$container_id" > "$EVIDENCE/round16-container-final-inspect.json" 2>/dev/null || true
    docker rm "$container_id" > "$EVIDENCE/round16-container-remove.txt" 2>&1 || true
  fi
}
trap cleanup EXIT

python3 "$SNAPSHOT/source/tools/launch_hybrid.py" \
  --version 403 \
  --storage MXFP4 \
  --placement static \
  --resource-mode shared \
  --shared-training-container "$TRAINING_ID" \
  --bank-receipt "$SNAPSHOT/rebound/bank-receipt.json" \
  --device-gate "$ROOT/evidence/20260920-c128-expert-v19/result.json" \
  --overlay-python "$SNAPSHOT/overlay-python" \
  --framework-profile "$SNAPSHOT/framework-profile.json" \
  --context-target 4096 \
  --context-reference-profile "$ROOT/evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json" \
  --streaming-profile "$SNAPSHOT/streaming-profile.json" \
  --cancel-trace \
  --cancel-closure \
  --cost-mode OFF \
  --cost-manifest "$SNAPSHOT/cost-manifest.json" \
  --cost-budget "$SNAPSHOT/execution-budget.json" \
  --cost-admission "$ADMISSION"

container_id="$(python3 -c 'import json; print(json.load(open("'"$EVIDENCE"'/model-start.json"))["container_id"])')"
deadline=$((SECONDS + 21600))
while (( SECONDS < deadline )); do
  if ! docker inspect -f '{{.State.Running}}' "$container_id" | grep -Fxq true; then
    echo "S0 server exited before readiness" >&2
    exit 4
  fi
  if [[ -f "$EVIDENCE/server.log" ]] && tail -c 131072 "$EVIDENCE/server.log" | grep -Eq 'Uvicorn running on http://127.0.0.1:30091|The server is fired up and ready to roll!'; then
    break
  fi
  sleep 5
done
if (( SECONDS >= deadline )); then
  echo "S0 server readiness timeout" >&2
  exit 5
fi

docker stats --no-stream --format '{{json .}}' "$container_id" > "$EVIDENCE/round16-pre-driver-docker-stats.json"
docker exec "$container_id" bash -lc \
  'PYTHONPATH=/round9/round16-s0-perf-a-v403:$PYTHONPATH python /round9/round16-s0-perf-a-v403/round16_integration/s0_perf_http_driver.py --fixture /round9/round16-s0-perf-a-v403/round10-t1-fixture.json --block S0-A --out /out' \
  > "$EVIDENCE/round16-s0-perf-driver.log" 2>&1

python3 - <<'PY'
import json
p=json.load(open('/mnt/data1/kt-round9/evidence/20260929-round16-s0-perf-a-v403/round16-s0-perf-result.json'))
assert p['status']=='PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK'
assert p['generation_attempts']==10
assert set(p['summaries'])=={'W1','W2','W4'}
print(json.dumps(p['summaries'],sort_keys=True))
PY

docker stop -t 10 "$container_id" >/dev/null
docker inspect "$container_id" > "$EVIDENCE/round16-container-final-inspect.json"
docker rm "$container_id" > "$EVIDENCE/round16-container-remove.txt"
container_id=""
trap - EXIT
