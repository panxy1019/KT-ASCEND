#!/usr/bin/env bash
set -euo pipefail
readonly IMAGE="quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler"
readonly NAME="round16-b-qsa-multihead-v001"
readonly SNAPSHOT="/mnt/data1/kt-round9/artifacts/round16-b/q32-v5-live-v023"
readonly EVIDENCE="/mnt/data1/kt-round9/evidence/20260929-round16-b-qsa-multihead-v001"
readonly IMAGE_LD="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:"
test -d "$SNAPSHOT"
test -f "$SNAPSHOT/snapshot.json"
test ! -e "$EVIDENCE"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "refusing to replace existing container $NAME" >&2
  exit 3
fi
mkdir -p "$EVIDENCE"
python3 - "$EVIDENCE/launch-contract.json" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": 1,
    "created_at": time.time(),
    "container": "round16-b-qsa-multihead-v001",
    "physical_device": 12,
    "container_device": 0,
    "numa_nodes": [2, 3, 4, 7],
    "snapshot": "/mnt/data1/kt-round9/artifacts/round16-b/q32-v5-live-v023",
    "probe": "round16_integration.b_qsa_multihead_device_probe",
    "network": "none",
    "memory_bytes": 32 * 1024**3,
}, indent=2, sort_keys=True) + "\n")
PY
set +e
timeout --signal=TERM --kill-after=60s 1800s docker run \
  --name "$NAME" \
  --network none \
  --cpuset-mems 2,3,4,7 \
  --memory 32g \
  --memory-swap 32g \
  --shm-size 4g \
  --device /dev/davinci12:/dev/davinci0 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$SNAPSHOT:/work:ro" \
  -v "$EVIDENCE:/evidence:rw" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/work \
  -e "LD_LIBRARY_PATH=$IMAGE_LD" \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  "$IMAGE" \
  /bin/bash -c 'python -m round16_integration.b_qsa_multihead_device_probe --output /evidence/result.json' \
  2>&1 | tee "$EVIDENCE/container.log"
run_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_rc" > "$EVIDENCE/container-exit-code.txt"
docker inspect "$NAME" > "$EVIDENCE/container-inspect.json" || true
docker rm "$NAME" > "$EVIDENCE/container-remove.txt" 2>&1 || true
exit "$run_rc"
