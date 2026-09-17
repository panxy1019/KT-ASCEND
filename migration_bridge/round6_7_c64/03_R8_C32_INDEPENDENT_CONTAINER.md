# R8-B1-C32 independent execution

User request: create a new container for R8-B1-C32. Date: 2026-09-17 UTC.

## Isolation

- Container: `kt-r67-r8-b1-c32-v2`.
- ID: `702a77a98374b6054fd4c09f9eb668af7fa1a5bcee508eb6a184647117a39ffa`.
- NPU6, PCI `0000:41:00.0`, physical NUMA2; CPU64–95.
- 32 workers, **one** subpool, batch1, Deferred/Dynamic/Graph OFF.
- Memory limit192GiB; swap forbidden; shm16GiB; isolated network namespace.
- SYS_NICE permits explicit NUMA memory policy. No business-container changes.
- Frozen model/GGUF/shared runtime mounts read-only. Only this study's output is
  writable, with container-local bind overrides for the candidate native binary
  and its two WorkerPool source files. No shared source/venv replacement.

Host admission: approximately961GiB available of1TiB, swap0. NPU6 had no running
process. R16 stays in its original container on NPU0/CPU0–31.

## Why there is a C32 runtime amendment

The old WorkerPool cannot safely use sparse physical map[2]. Only its translation
unit was replaced with the tested NUMA candidate. There is no 1408 TP2 change.
Re-linking all original objects and applying the original strip operation first
reproduced the original extension SHA exactly:
`0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895`.

After replacing only `worker_pool.cpp.o`, candidate SHA:
`a2bdcdd79a7d25263358f9a01fe3ca9197531f2dc328719694eb9f1b80a4d049`.
Expert arithmetic object files are unchanged. This is still a new runtime
identity, not a claim that the entire extension is byte-identical.

The worker freezes a replacement runtime manifest, permits changes only to
WorkerPool cpp/header/native-extension hashes, and rechecks all other frozen
assets. Baseline corpus, model, GGUF, placement and evaluation rules are unchanged.

## Automation and status

Study: `/workspace/results/r65-r67-r8-c32-v2/study`.
Host backing directory:
`/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-dev-v1/r8-c32-v2`.
Freeze SHA: `2f70c71744efed82dd8ce65f2b6daa5268fb9635a69e2e6217ba8cec9aceb6b2`.

Background flow: compact32 → restart → observer/mapping/routes → local expert
screen52 experts/156 operands → CMMLU256 → paired product report. Predecessor
failure prevents quality. Outer owned-process guard bounds runtime to24h; errors
do not target the original R16 process or unrelated containers.

At launch, status is SYSTEM RUNNING / QUALITY NOT_RUN. This document is not a
completed system or quality qualification. Results are written to progress.json
and summary.json. No concurrent measurements count as formal performance.

Original serial controller is untouched and may later schedule its own R8-C32.
That is a separate run, not this container's result; do not merge their raw files
or silently substitute one runtime identity for the other.

Initial container creation hit a missing nested mountpoint under the read-only
parent mount. The new empty target directory was created and the same container
started; no model process had started during that failure. Docker reordered its
bind list; resume verification compares the complete sorted bind set.
