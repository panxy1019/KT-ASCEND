#!/bin/bash
set -euo pipefail
probe() {
  local name=$1
  shift
  timeout 240 python /round9/tools/native_unblocked_probe.py --model /model --output "/round9/evidence/${name}-final.json" "$@" > "/round9/evidence/${name}-final.log" 2>&1
}
for layer in 0 23 24 47; do
  for pools in 1 2; do probe "BF16-p${pools}-L${layer}" --format BF16 --pools "$pools" --layer "$layer"; done
done
for pools in 1 2; do
  probe "F32-p${pools}-L0" --format F32 --pools "$pools" --layer 0
  for fmt in BF16 F32; do probe "${fmt}-p${pools}-L0-compat" --format "$fmt" --pools "$pools" --layer 0 --compat; done
done
probe BF16-p2-L0-full512 --format BF16 --pools 2 --layer 0 --full-bank
