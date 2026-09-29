# Round16 Dual-Carrier DEV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce installable, CPU-only-tested SGLang/KT provider and vLLM-Ascend/KT Expert bridge packages that are ready to consume S15 after its terminal barrier.

**Architecture:** One runtime-neutral immutable contract is owned by the integration worktree. A and B consume that contract in independent sparse worktrees, with lazy runtime imports and explicit device-pending capability states. Device correctness, full-model integration, graph capture, and performance remain gated behind the Round15 terminal receipt.

**Tech Stack:** Python 3.11+, dataclasses/protocols, unittest, packaging metadata; SGLang/KT and vLLM-Ascend are optional runtime dependencies imported only through explicit registration.

**Spec:** User-provided documents 60, 61, and 62 in the 2026-09-28 Round16 request.

## Global Constraints

- Every new path is default OFF and import-side-effect free.
- DEV uses zero NPU, zero full-model loads, zero graph capture, zero performance tests, and zero full-bank reads.
- Round15 source, image, environment, and shared libraries are read-only.
- A/B use separate worktrees, caches, artifacts, ports, and build outputs.
- A/B consume `round16-contract-v2`; only the integration coordinator writes shared contracts.
- CPU tests use at most four exclusive physical cores, 16 GiB, two compile jobs, and one OpenMP/MKL thread per agent.
- No push, merge, tag, reset, clean, timer, cache retirement, or global runtime registration.

## Review Focus

- Unsupported shape/dtype/layout must choose fallback before any state write.
- A repeated or spawn-process import must not initialize NPU, CPUInfer, or worker pools.
- Router Top-K and weights must be frozen once and weights applied exactly once after CPU/NPU split.
- Cancellation and slot reuse must reject late completions by the full request/allocation/batch generation identity.
- Selective loading must preserve logical expert count and aliases without a full-expert temporary allocation plan.

---

### Task 1: Shared immutable contract and control receipts

**Files:** `round16/common/contracts.py`, `tests/round16/test_common_contracts.py`, `artifacts/round16/*.json`.

- [x] Write and observe failing common-contract tests.
- [x] Implement immutable tensor, route, lease, support, completion, result, and provider protocols.
- [x] Run the common tests and freeze `round16-contract-v2`.

### Task 2: A provider package

**Files:** `round16/a/provider_manifest.py`, `round16/a/sglang_adapter.py`, `round16/a/registration.py`, `tests/round16/test_a_*.py`.

- [x] Map the actual Round14a/SGLang insertion points and Round15-owned boundaries.
- [x] Test rejection, lazy registration, metadata conversion, fallback-before-write, completion, lease, alias, close, and packaging behavior.
- [x] Implement the minimum P0/P1 provider manifest and thin adapter; classify P2 dependencies without copying their runtime.
- [x] Record all device-only imports and numerical checks as `DEVICE_REQUIRED_NOT_RUN`.

### Task 3: B vLLM Expert bridge package

**Files:** `round16/b/model_plugin.py`, `routing_bridge.py`, `selective_loader.py`, `kt_expert_backend.py`, `lifecycle_bridge.py`, `graph_boundary.py`, and `tests/round16/test_b_*.py`.

- [x] Resolve local and official frozen-version extension points without importing a device runtime.
- [x] Test model-only idempotent registration, route split/merge, q=0/q=32/q=all, padding, duplicate experts, and one-time weight application.
- [x] Test selective Q32 allocation plans, fused slices, tied aliases, rollback, and absence of full-expert temporary plans.
- [x] Test cancel, late completion, generation mismatch, buffer reuse, close, and graph replay issuing a fresh CPU task.
- [x] Implement the minimum bridge around injected mock/native interfaces; leave device bindings explicit and pending.

### Task 4: Integration review and DEV freeze

**Files:** combined `round16/`, `tests/round16/`, `artifacts/round16/RESOLVED_CODE_MAP.md`, `DEV_READINESS.json`, hashes and Chinese report.

- [x] Import A/B in fresh child processes and prove no workers, device initialization, or global registration occur.
- [x] Build/install each package in an isolated temporary target and rerun its offline suite without network access.
- [x] Review A/B against contracts 60/61/62 and run the combined offline test suite.
- [x] Freeze source/build/test hashes, budgets, unresolved device items, S15 rebase procedure, and `ROUND16_DEV_READY_WAIT_ROUND15`.
