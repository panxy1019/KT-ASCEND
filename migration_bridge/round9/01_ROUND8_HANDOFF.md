# Round8 handoff — PENDING

Do not replace this with PASS until current evidence is independently audited.
Source of live execution is round8_prep/10_HANDOFF/CURRENT_EXECUTION.md.
Round8 Q64 is an engineering reference, not full All-NPU qualification.

Required handoff: final parent/child and actual imported Python/native/overlay
hashes; image/CANN/torch/torch_npu/driver; model/tokenizer/GGUF/Q32 mask/CMMLU
manifest; exact C64 topology; system/identity/quality/stability/capacity and
performance artifacts; known failures and scope reductions; rebuild commands.

Current Dynamic candidate uses256 threshold,512chunk; DEV1024 deliberately
skipped. Preserve old heldout-v2 watchdog failure and explicit v3 retry.
Use `tools/qwen_dynamic_retry_release.py` for actual heldout-v3/CMMLU-v2 audit,
then independent release review; `--performance` only after gate release and
isolation. Do not mix contended quality timings with formal performance.

Dynamic/Deferred are not required to deliver positive speedup. A truthful
rejection/OFF decision is valid; an unfinished authorized test isn't PASS.
Round9 heavy entry requires a locally committed Round8 final state, not merely
metadata availability or a server finishing its last request.
