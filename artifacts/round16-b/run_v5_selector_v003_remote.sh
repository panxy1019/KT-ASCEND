#!/usr/bin/env bash
set -euo pipefail
readonly IMAGE="quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler"
readonly NAME="round16-b-v5-selector-v003"
readonly ROOT="/mnt/data1/kt-round9/artifacts/round16-b/q32-v5-live-v015"
readonly EVIDENCE="/mnt/data1/kt-round9/evidence/20260929-round16-b-v5-selector-v003"
readonly IMAGE_LD="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64"
test -f "$ROOT/round16_integration/b_v5_ascend_selector_probe.py"
test ! -e "$EVIDENCE"
mkdir -p "$EVIDENCE"
set +e
timeout --signal=TERM --kill-after=30s 1800s docker run \
  --name "$NAME" --network none --cpuset-cpus 160,162,164,166 --cpuset-mems 2 \
  --memory 16g --memory-swap 16g --shm-size 1g \
  --device /dev/davinci12:/dev/davinci0 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$ROOT:/work:ro" -v "$EVIDENCE:/evidence:rw" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 -e PYTHONPATH=/work -e "LD_LIBRARY_PATH=$IMAGE_LD" \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  "$IMAGE" /bin/bash -c 'python -m round16_integration.b_v5_ascend_selector_probe --output /evidence/result.json' \
  2>&1 | tee "$EVIDENCE/container.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$EVIDENCE/container-exit-code.txt"
docker inspect "$NAME" > "$EVIDENCE/container-inspect.json" || true
docker rm "$NAME" > "$EVIDENCE/container-remove.txt" 2>&1 || true
cd "$EVIDENCE"; sha256sum container.log container-exit-code.txt container-inspect.json result.json > SHA256SUMS.txt 2>/dev/null || true
exit "$rc"
