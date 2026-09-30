#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE="quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler"
readonly NAME="round18-native-expert-pool-fixed-slot-v017"
readonly SNAPSHOT="/mnt/data1/kt-round9/artifacts/round18/native-expert-pool-v017"
readonly SNAPSHOT_SHA256="e78482746a11e8d4da11dea65baa3c44668eb726aab5dfcddd1125b4aff77ac0"
readonly EVIDENCE="/mnt/data1/kt-round9/evidence/20260930-round18-native-expert-pool-fixed-slot-v017"
readonly MODEL="/home/admin/models/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540"
readonly KT_RUNTIME="/mnt/data1/kt-round9/round16-b-kt-runtime-py312-v1"
readonly KT_DEPS="/mnt/data1/kt-round9/round16-b-kt-native-py312-v1/deps"
readonly ADMISSION_TOOL="/mnt/data1/kt-round9/kt-root-cause-admission-tool-v2/tools/round16_admission.py"
readonly POLICY="/mnt/data1/kt-round9/round17-admission-v14/policy.json"
readonly CPUSET="160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190"

test -d "$SNAPSHOT"
(cd "$SNAPSHOT" && sha256sum -c SHA256SUMS.txt >/dev/null)
test "$(cat "$SNAPSHOT/SNAPSHOT_SHA256")" = "$SNAPSHOT_SHA256"
test -f "$SNAPSHOT/v2-receipt.json"
test -d "$MODEL"
test -d "$KT_RUNTIME"
test -f "$KT_DEPS/lib/libhwloc.so.15.5.2"
test ! -e "$EVIDENCE"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "refusing to replace existing container $NAME" >&2
  exit 3
fi

mkdir -p "$EVIDENCE"
python3 "$ADMISSION_TOOL" precheck \
  --policy "$POLICY" --output "$EVIDENCE/admission-precheck.json"
python3 "$ADMISSION_TOOL" sample \
  --policy "$POLICY" --output "$EVIDENCE/admission-round1.json"
python3 "$ADMISSION_TOOL" sample \
  --policy "$POLICY" --output "$EVIDENCE/admission-round2.json"
python3 "$ADMISSION_TOOL" validate \
  --policy "$POLICY" \
  --round1 "$EVIDENCE/admission-round1.json" \
  --round2 "$EVIDENCE/admission-round2.json" \
  --output "$EVIDENCE/admission.json"
python3 - "$EVIDENCE/admission.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("status") != "PASS":
    raise SystemExit("admission did not pass")
PY

python3 - "$EVIDENCE/launch-contract.json" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": 1,
    "created_at": time.time(),
    "container": "round18-native-expert-pool-fixed-slot-v017",
    "physical_device": 12,
    "container_device": 0,
    "numa_nodes": [2, 3, 4, 7],
    "cpus": 16,
    "memory_bytes": 64 * 1024**3,
    "memory_swap_bytes": 64 * 1024**3,
    "shm_bytes": 16 * 1024**3,
    "network": "none",
    "snapshot": "/work",
    "snapshot_sha256": "e78482746a11e8d4da11dea65baa3c44668eb726aab5dfcddd1125b4aff77ac0",
    "model": "/model",
    "probe": "round18_integration.native_fixed_slot_write_probe",
    "scope": "single layer0 Physical-Q32 owner; no full model; no CPU bank",
    "mutation_rule": "fail closed on unsupported post-load format; no parent rebuild",
}, indent=2, sort_keys=True) + "\n")
PY

set +e
timeout --signal=TERM --kill-after=180s 1800s docker run \
  --name "$NAME" --network none --cap-add SYS_NICE \
  --cpuset-cpus "$CPUSET" --cpuset-mems 2,3,4,7 \
  --memory 64g --memory-swap 64g --shm-size 16g \
  --device /dev/davinci12:/dev/davinci12 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$SNAPSHOT:/work:ro" -v "$EVIDENCE:/out:rw" \
  -v "$MODEL:/model:ro" -v "$KT_RUNTIME:/round16-kt:ro" \
  -v "$KT_DEPS:/opt/round16-hwloc:ro" \
  -e PYTHONPATH=/work:/round16-kt \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e PYTHONUNBUFFERED=1 \
  -e ROUND18_ENABLE_FIXED_SLOT_WRITE=1 \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e MALLOC_ARENA_MAX=16 \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e LD_LIBRARY_PATH=/opt/round16-hwloc/lib \
  "$IMAGE" \
  /bin/bash -c 'source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} && python -m unittest /work/tests/round18/test_fixed_slot_write.py /work/tests/round18/test_fixed_slot_write_probe.py /work/tests/round18/test_native_fixed_slot_write_probe.py && python -m round18_integration.native_fixed_slot_write_probe --enable-device --source-token /work/source-token.json --snapshot-manifest /work/snapshot.json --v2-receipt /work/v2-receipt.json --model-root /model --output /out/fixed-slot-receipt.json --layer 0 --physical-slot 0 --replacement-expert 32' \
  2>&1 | tee "$EVIDENCE/container.log"
run_rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "$run_rc" > "$EVIDENCE/container-exit-code.txt"
docker inspect "$NAME" > "$EVIDENCE/container-inspect.json" || true
find "$EVIDENCE" -maxdepth 1 -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$EVIDENCE/evidence-sha256-pre-remove.txt"
docker rm "$NAME" > "$EVIDENCE/container-remove.txt" 2>&1 || true
find "$EVIDENCE" -maxdepth 1 -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$EVIDENCE/evidence-sha256-final.txt"
exit "$run_rc"
