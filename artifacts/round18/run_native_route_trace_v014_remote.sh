#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE="quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler"
readonly NAME="round18-native-route-trace-v014"
readonly SNAPSHOT="/mnt/data1/kt-round9/artifacts/round18/native-hybrid-full-v014"
readonly SNAPSHOT_SHA256="db343d82b54c5a08d072630b90c3459a99945e62f36913457ac00e49df5730bf"
readonly EVIDENCE="/mnt/data1/kt-round9/evidence/20260930-round18-native-route-trace-v014"
readonly MODEL="/home/admin/models/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540"
readonly BANK_ROOT="/mnt/data1/kt-round9/banks"
readonly KT_RUNTIME="/mnt/data1/kt-round9/round16-b-kt-runtime-py312-v1"
readonly KT_DEPS="/mnt/data1/kt-round9/round16-b-kt-native-py312-v1/deps"
readonly ADMISSION_TOOL="/mnt/data1/kt-round9/kt-root-cause-admission-tool-v2/tools/round16_admission.py"
readonly POLICY="/mnt/data1/kt-round9/round17-admission-v14/policy.json"
readonly CPUSET="160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638"
readonly IMAGE_LD="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:"

test -d "$SNAPSHOT"
(cd "$SNAPSHOT" && sha256sum -c SHA256SUMS.txt >/dev/null)
test "$(cat "$SNAPSHOT/SNAPSHOT_SHA256")" = "$SNAPSHOT_SHA256"
test -d "$MODEL"
test -d "$BANK_ROOT/candidate-mxfp4-all48-v1"
test -d "$KT_RUNTIME"
test -f "$KT_DEPS/lib/libhwloc.so.15.5.2"
test ! -e "$EVIDENCE"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "refusing to replace existing container $NAME" >&2
  exit 3
fi
mkdir -p "$EVIDENCE"
python3 "$ADMISSION_TOOL" precheck --policy "$POLICY" --output "$EVIDENCE/admission-precheck.json"
python3 "$ADMISSION_TOOL" sample --policy "$POLICY" --output "$EVIDENCE/admission-round1.json"
python3 "$ADMISSION_TOOL" sample --policy "$POLICY" --output "$EVIDENCE/admission-round2.json"
python3 "$ADMISSION_TOOL" validate --policy "$POLICY" --round1 "$EVIDENCE/admission-round1.json" --round2 "$EVIDENCE/admission-round2.json" --output "$EVIDENCE/admission.json"
python3 - "$EVIDENCE/admission.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get('status') != 'PASS':
    raise SystemExit('admission did not pass')
PY
python3 - "$EVIDENCE/launch-contract.json" <<'PY'
import json,pathlib,sys,time
path=pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
 "schema":1,"created_at":time.time(),
 "container":"round18-native-route-trace-v014",
 "physical_device":12,"container_device":0,"numa_nodes":[2,3,4,7],
 "cpus":136,"memory_bytes":640*1024**3,"memory_swap_bytes":640*1024**3,
 "shm_bytes":16*1024**3,"network":"none","cap_add":["SYS_NICE"],
 "snapshot":"/work","snapshot_sha256":"db343d82b54c5a08d072630b90c3459a99945e62f36913457ac00e49df5730bf",
 "model":"/model","bank":"/banks/candidate-mxfp4-all48-v1",
 "probe":"round18_integration.native_route_trace_probe",
 "worker":"round18_integration.native_hybrid_live_worker.Round18NativeHybridWorker",
 "graph_mode":"NONE","requests":{"W1":1,"W2":1,"W4":1},"diagnostic":"D2H route capture; not performance"
},indent=2,sort_keys=True)+"\n")
PY

set +e
timeout --signal=TERM --kill-after=180s 21600s docker run \
  --name "$NAME" --network none --cap-add SYS_NICE \
  --cpuset-cpus "$CPUSET" --cpuset-mems 2,3,4,7 \
  --memory 640g --memory-swap 640g --shm-size 16g \
  --device /dev/davinci12:/dev/davinci0 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$SNAPSHOT:/work:ro" -v "$EVIDENCE:/evidence:rw" \
  -v "$MODEL:/model:ro" -v "$BANK_ROOT:/banks:ro" \
  -v "$KT_RUNTIME:/round16-kt:ro" -v "$KT_DEPS:/round16-deps:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/work:/round16-kt \
  -e "LD_LIBRARY_PATH=/round16-deps/lib:$IMAGE_LD" \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e MALLOC_ARENA_MAX=16 -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V1=1 \
  -e ROUND18_ENABLE_NATIVE_HYBRID=1 \
  -e ROUND18_REQUESTED_GRAPH_MODE=NONE \
  -e ROUND18_WORKER_PREFLIGHT=/work/B2_EXACT_WORKER_PREFLIGHT.json \
  -e ROUND18_SNAPSHOT_MANIFEST=/work/snapshot.json \
  -e ROUND18_SOURCE_TOKEN=/work/source-token.json \
  -e ROUND18_BANK_ROOT=/banks/candidate-mxfp4-all48-v1 \
  -e ROUND18_BINDING_RECEIPT_PATH=/evidence/binding-receipt.json \
  "$IMAGE" \
  /bin/bash -c 'python -m unittest discover -s /work/tests/round18 -p "test_*.py" && python -m round18_integration.native_route_trace_probe --model /model --fixtures /work/fixtures/threshold-request-fixtures.json --output /evidence/route-result.json --trace-parent /evidence/route --enable-parent /evidence/route-control' \
  2>&1 | tee "$EVIDENCE/container.log"
run_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_rc" > "$EVIDENCE/container-exit-code.txt"
docker inspect "$NAME" > "$EVIDENCE/container-inspect.json" || true
find "$EVIDENCE" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/evidence-sha256-pre-remove.txt"
docker rm "$NAME" > "$EVIDENCE/container-remove.txt" 2>&1 || true
find "$EVIDENCE" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/evidence-sha256-final.txt"
exit "$run_rc"
