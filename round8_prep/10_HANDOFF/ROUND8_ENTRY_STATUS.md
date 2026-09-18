# Round8 execution entry

Round7 reference: `30de6a234284bfa25d196037ed692d9584855ddf`.
Round7 closed with user-approved scope reduction. Do not resume Medium/Long
performance or claim full isolated performance qualification.

## Actual execution

- Host: `110.129.0.22`; writable root `/home/admin/kt-round8-qwen35`.
- Container: `kt-round8-qwen35-prep`.
- Container ID: `b41c6686790fcf6ebaf5566302854f56cf653e8139a2bc66c22af90c1ed89d52`.
- CPU set 128–191; memory nodes 4,5; physical NPU2; memory limit 192 GiB.
- Existing Round7 source/model/environment mounts are read-only.
- Independent venv created at `/workspace/round8/venv`; dependencies not yet installed.
- Config probe used the existing **read-only reference environment**, not the new
  empty venv. Therefore this is not an independent clean-build PASS.

The config-only, offline probe passed using local metadata and
`trust_remote_code=False`. It imported torch and resolved
`Qwen3_5MoeForConditionalGeneration`: 40 layers, 256 experts, top-k 8.
Raw result is preserved in `round8_entry_probe.json`; original remote result is
`/home/admin/kt-round8-qwen35/entry-probe.json`.

Runtime reported Python 3.11.15, torch package 2.9.0, torch-npu 2.9.0.post2,
transformers 5.3.0.dev0, sgl-kernel-npu 2026.6.1. SGLang distribution metadata was
not installed; this does not establish absence of source-based serving imports.
Torch import emitted `can not use command: npu-smi info` and
`DrvMngGetConsoleLogLevel failed.(ret=4)`. NPU tensor/context execution remains
unverified and these warnings must be investigated before runtime promotion.

## Ordered continuation

1. Verify new container NPU device visibility and CANN environment with an isolated
   tensor/context probe; preserve any failures. Do not disturb unrelated workers.
2. Freeze standalone dependency/build decision in the independent environment;
   do not mutate the Round7 reference environment.
3. Audit packed expert weight shapes and conversion feasibility against the frozen
   Qwen revision, then qualify sample conversion and CPU512/C64 local operator.
4. Validate GDN/model construction before static full-model Q64, then Q32.

Frozen model: `Qwen/Qwen3.5-35B-A3B` revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`.
Dynamic, Deferred, Graph, MTP and speculative decoding remain OFF. No Qwen model
weights have been loaded, no full-model inference or quality gate has passed.
Do not inherit DeepSeek qualification or claim Round8 complete.

No push, main merge, tag, or wholesale upstream upgrade.
