# Round9 candidates implementation plan

> Execute inline with superpowers:executing-plans. User explicitly authorized continuous implementation on 2026-09-20; do not repeat approval.

Goal: host-tested candidate configuration/C128, CPU low-bit conversion and bounded Dynamic seams, with honest native/device eligibility.
Architecture: independent candidate modules alongside the immutable reference; reuse LLAMAFILE multi-pool runtime; no automatic device launch.
Tech: Python standard library, optional torch only at actual tensor adapter boundary, C++ partition helper compiler test.
Spec: migration_bridge/round9/35_DEVELOPMENT_FIRST_C128_LOWBIT_DYNAMIC.md

Constraints: preserve reference hash/v6 environment; Q64/TP1/batch1, original BF16 promotion; no heavy conversion, NPU/performance tests, timer restart or push.
Review focus: stale/SMT topology; quantized block bit order and under/overflow; partial publication failure; active request/generation race; unsupported kernel never silently enabled.

- [x] Candidate contract: runtime/candidate_config.py Profile.from_dict, partitions, validate_topology; tests/test_candidates.py rejects unknown fields, duplicate cores, incorrect quota and unsupported native. candidate overlay hook is opt-in and preserves default guard.
- [x] C128: compile existing native make_llamafile_tp_slices(640,4,true), assert offsets0/160/320/480 and lengths160; compare Python partitions. Produce four-pool args with explicit admitted NUMA IDs. Never allocate CPUs automatically.
- [x] Lowbit: runtime/lowbit.py explicit GGML Q8_0/MXFP4 codec; runtime/expert_source.py original BF16 streamed projection reader; tools/convert_candidate_bank.py no-clobber GGUF publication and source-bound manifest. Tests known bytes, signed/zero/extreme blocks, decoded oracle, bank dimensions and truncated source. Native lowbit320/160 remains explicit UNSUPPORTED until true implementation qualification.
- [x] Dynamic: runtime/dynamic.py deterministic plan, lock/state/generation/request lease and fail-closed commit; runtime/dynamic_adapter.py concrete original-source tensor copy and KTEP maps/masks; CPU tensor integration tests. No full GPU context. No request begins after partial failed mutation. Actual scheduler integration requires explicit quiescence evidence.
- [x] tools/candidate_workflow.py stage-selective host/native/tensor tests with hashes/results; documentation and candidate configs; review whole diff, fix issues, commit.

For each code batch: add failure-oriented tests, observe RED, implement, run GREEN. Record actual commands and limitations in36_CANDIDATE_DEVELOPMENT_DELIVERY.md.

Execution record: report36 and source-bound development-20260920 evidence. All framework tasks completed within explicit unsupported/device-pending boundaries; no candidate model qualification. Independent review gpt-5.6-sol/medium issues repaired.
