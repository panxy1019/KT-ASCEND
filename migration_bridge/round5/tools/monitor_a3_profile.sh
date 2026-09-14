#!/usr/bin/env bash
set -euo pipefail

container=${1:?usage: monitor_a3_profile.sh CONTAINER MARKER OUTPUT [INTERVAL]}
marker=${2:?usage: monitor_a3_profile.sh CONTAINER MARKER OUTPUT [INTERVAL]}
output=${3:?usage: monitor_a3_profile.sh CONTAINER MARKER OUTPUT [INTERVAL]}
interval=${4:-10}

printf 'timestamp_utc,container_memory_usage,sglang_rss_kib,npu_process_memory_mib\n' >"${output}"
while [[ -e ${marker} ]]; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  container_memory=$(docker stats --no-stream --format '{{.MemUsage}}' "${container}" | cut -d/ -f1 | xargs)
  sglang_rss=$(docker exec "${container}" sh -c \
    "ps -eo rss=,cmd= | awk '/sglang::|python -m sglang.launch_server/ {sum += \$1} END {print sum + 0}'")
  npu_memory=$(npu-smi info | awk -F'|' \
    '/python|sglangschedul/ {gsub(/[[:space:]]/, "", $5); sum += $5} END {print sum + 0}')
  printf '%s,%s,%s,%s\n' "${timestamp}" "${container_memory}" "${sglang_rss}" "${npu_memory}" >>"${output}"
  sleep "${interval}"
done
