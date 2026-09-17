# KT-ASCEND Qualification Workbench

Status: `DESIGN_APPROVED__IMPLEMENTATION_NOT_STARTED`

This directory defines a small, AI-friendly qualification workbench for the
KTransformers × Kunpeng × Ascend integration.  The first version is a
diagnostic tool, not a mandatory CI platform and not an autonomous root-cause
engine.

## 1. Problem statement

The repository already contains strong test assets: CPU and transfer contract
tests, real-model replay, deterministic serving checks, frozen numerical
contracts, route coverage, downstream quality, and stability evidence.  The
remaining usability problem is orchestration:

- tools are distributed across `migration_bridge/round3` through `round5`;
- complete P1/P2/P3 execution still depends on documented manual commands;
- an AI agent has no single capability registry from which to choose tests;
- raw outputs use several related schemas without a common run envelope;
- environment, parent commit, SGLang child, model, GGUF, placement, and corpus
  provenance are not checked by one entry point;
- current CPU CI does not qualify the Ascend runtime path;
- correctness instrumentation and performance observation are not represented
  as separate execution modes.

The workbench addresses those gaps while preserving the existing, proven test
implementations.

## 2. Initial qualification boundary

The initial implementation supports the already-qualified Round 5 boundary:

| Dimension | Initial value |
|---|---|
| Model | DeepSeek-V2-Lite revision `604d5664dddd88a0433dbae533b7fe9472482de0` |
| Host | Kunpeng 920 / AArch64 |
| Accelerator | one Ascend NPU, NPU0 |
| Parallelism | TP1, batch 1 |
| Placement | static P1/P2/P3 |
| CPU plane | LLAMAFILE / CPUInfer |
| NPU plane | BF16 unquantized |
| Serving | SGLang KTEP |
| Disabled | Graph, Deferred, Dynamic, MTP, Speculative |

The workbench is developed from the moving repository `main`, but its initial
reference baseline is immutable:

```text
tag: round5-upstream-aligned-tp1-a3-verified
parent: be5b48621db32fb894402e3e886060dea263cdc9
SGLang: 412219b4b05a888808f0b88d2963a8b6164a0fda
```

The subject under test may be any working tree, commit, PR, wheel installation,
or future baseline.  A run records subject and baseline separately.

## 3. Design principles

### 3.1 AI plans; the runner executes

AI is external to the workbench.  The workbench does not embed an LLM or Agent
SDK and does not restrict an AI agent from using shell, SSH, or other tools.
It exposes discoverable, composable commands and machine-readable results.

```text
AI or engineer
  -> capability registry
  -> JSON run plan
  -> deterministic runner
  -> local/SSH/container executor
  -> raw artifacts + normalized evidence
  -> AI or engineer chooses the next probe
```

The AI may select, skip, reorder, or append tests, but the run plan records the
reason for those choices.  The runner does not change a frozen acceptance
contract.

### 3.2 Atomic operations before fixed pipelines

The primary interface is a set of atomic operations.  Suites are templates
that compose those operations, not hard-coded mandatory pipelines.

Expected interface:

```bash
python -m qualification list
python -m qualification describe ascend.framework_transfer
python -m qualification env inspect
python -m qualification source verify
python -m qualification server start --profile p2
python -m qualification gate run ascend.framework_transfer
python -m qualification inspect RUN_ID --json
python -m qualification events RUN_ID --follow
python -m qualification server stop RUN_ID
python -m qualification report RUN_ID
python -m qualification run --suite ascend-p2-smoke
python -m qualification run-plan plan.json
```

### 3.3 Reuse through thin adapters

Existing tools remain their own source of truth.  The workbench invokes them,
retains their raw output, and normalizes only the fields needed for common
reporting.

Initial adapter families:

- KTransformers `kt-kernel/test/run_suite.py`;
- KTransformers model-E2E evidence/provenance concepts;
- SGLang serving benchmarks and profiler;
- public `torch_npu` / CANN runtime probes;
- current Round 3–5 qualification tools;
- pytest and arbitrary registered commands.

An upstream interface change should require an adapter change, not a runner
rewrite.  Existing scripts are not copied or silently modified.

### 3.4 Artifact first

Every result is bound to source and input identity.  At minimum a hardware run
records:

- parent and SGLang child commit;
- dirty-tree state and relevant diff identity;
- model revision and tokenizer hash;
- GGUF, placement, corpus, and contract hashes;
- OS, Python, CANN, torch, torch_npu, and kt-kernel versions;
- host architecture, CPU affinity, NUMA policy, visible NPU, and container;
- exact command, working directory, environment delta, timestamps, and exit
  status;
- raw stdout/stderr and normalized result.

The workbench prepares compact Git evidence but does not commit or push it.

### 3.5 Correctness and performance are separate modes

Correctness runs may enable route/tensor dumps or diagnostic synchronization.
They never publish performance claims.  Performance runs disable intrusive
diagnostics and freeze warmup, request shape, concurrency, and sampling.

Performance is observational in the first version.  A performance gate may be
promoted only after at least three controlled baseline runs and review of a
candidate contract.

## 4. Capability model

Backend adapters declare capabilities rather than forcing every test to branch
on device names.

Example backend configuration:

```toml
id = "ascend-a3"
host_arch = "aarch64"
accelerator = "ascend"

[capabilities]
framework_transfer = true
device_stream = true
route_recording = true
tensor_dump = true
graph = false
multi_device = false
hccl = false

[qualification]
bf16_unquantized = "qualified"
w8a8 = "unqualified"
mxfp4 = "unqualified"
```

A gate declares the capabilities and assets it needs:

```toml
id = "ascend.framework_transfer"
kind = "active_gate"
failure_layer = "transfer"
adapter = "pytest"
cost = "smoke"
timeout_seconds = 600
requires = ["framework_transfer", "device_stream"]
```

New hardware is added by registering a backend and adapters.  A common semantic
contract may be reused across backends; backend-specific behavior gets an
explicit backend-specific gate.

## 5. Test asset lifecycle

Existing and future tests have one of four roles:

| Role | Meaning | Default execution |
|---|---|---|
| `ACTIVE_GATE` | Stable contract protecting current production behavior | conditional or enabled |
| `REUSABLE_TOOL` | Runner/analyzer used by a gate | never alone |
| `INCIDENT_REPLAY` | Diagnostic for a historical or current incident | opt-in |
| `ARCHIVED_OBSOLETE` | Evidence tied to retired architecture | never |

New AI-authored tests move through:

```text
AD_HOC -> CANDIDATE -> ACTIVE -> RETIRED -> ARCHIVED
```

Promotion to `ACTIVE` requires a stable input, explicit contract, bounded cost,
repeatability, and a demonstrated production invariant.  A historical
experiment is not made permanent merely because it once found a bug.  Its
stable invariant should be extracted into the smallest useful regression.

Examples:

| Existing asset | Intended classification |
|---|---|
| Round 5 framework transfer 5-test suite | `ACTIVE_GATE` |
| Round 5 R2A lifecycle tests | `ACTIVE_GATE` |
| Same-path determinism runner | `REUSABLE_TOOL` used by an active gate |
| P2 layer-pair bisection | `INCIDENT_REPLAY` |
| raw ACL pointer transfer production variants | `ARCHIVED_OBSOLETE` |
| teacher-forced sweep | `REUSABLE_TOOL` |

## 6. Execution levels

Levels are recommendations for planning, not restrictions.

| Level | Purpose | Typical cost |
|---|---|---:|
| L0 Preflight | source, schema, import, fixture and CPU checks | minutes |
| L1 A3 Smoke | real NPU and minimum real-model path | 10–20 minutes |
| L2 Full Qualification | P1/P2/P3, numerical, quality, stability | long/manual/nightly |

An AI may stop after one gate, inspect evidence, add an incident replay, or
promote the run to a more expensive level.

## 7. Status and failure-layer vocabulary

Statuses distinguish a product failure from an unavailable test environment:

| Status | Meaning |
|---|---|
| `PASS` | Executed and satisfied the contract |
| `FAIL` | Executed and violated the contract |
| `BLOCKED` | Required model, data, dependency, hardware, or permission unavailable |
| `INFRA_ERROR` | Runner, SSH, container, port, disk, or network failure |
| `NOT_APPLICABLE` | Gate does not apply to the selected capabilities |
| `NOT_RUN` | Not executed, commonly after an earlier failure |
| `UNQUALIFIED` | Code exists without qualification evidence |

The first failed gate reports one earliest failure layer:

```text
environment
build
runtime_primitives
transfer
cpu_expert
routing_mapping
hybrid_layer
model_numerics
serving_determinism
quality
performance
stability
```

This is triage, not an automatic root-cause claim.

## 8. Event and recovery model

Each run emits append-only JSON Lines alongside human-readable logs:

```text
events.jsonl
commands.jsonl
console.log
state.json
summary.json
report.md
```

Representative events:

```json
{"event":"run_started","run_id":"..."}
{"event":"gate_started","gate":"ascend.framework_transfer"}
{"event":"command_started","argv":["python","-m","pytest","..."]}
{"event":"gate_finished","gate":"ascend.framework_transfer","status":"PASS"}
{"event":"gate_finished","gate":"p2.minimal_determinism","status":"FAIL","failure_layer":"serving_determinism"}
```

`state.json` is written atomically and points to the last durable event.  A
later process can inspect the run or resume gates that are explicitly marked
resumable.  Raw command output remains authoritative if normalization fails.

## 9. Executor model

The first version uses already-provisioned environments and composes four
executor types:

```text
local
ssh
docker_exec
ssh_docker_exec
```

The framework helps discover devices and ports, records owned PIDs, and can
stop services it started.  These are usability features rather than an AI
permission system.  An AI or engineer may operate outside the workbench.

Provisioning drivers, CANN, base images, models, datasets, and network settings
is out of the initial scope.  Environment provisioning may become a separate
provider interface later.

## 10. Artifact storage

Artifacts have three storage tiers:

```text
run workspace
  complete transient state, logs, and dumps

persistent artifact store
  host storage organized by run ID

compact Git evidence candidate
  normalized JSON, report, and SHA256SUMS
```

Suggested run layout:

```text
RUN_ID/
├── run.json
├── environment.json
├── plan.json
├── events.jsonl
├── commands.jsonl
├── state.json
├── raw/
├── normalized/
├── logs/
├── metrics/
├── summary.json
└── report.md
```

Retention is declarative.  Tensor dumps may be short-lived; normalized results
and compact evidence may be permanent.  Deletion is not part of the initial
runner implementation.

## 11. Contract governance

A result references the exact contract SHA used for evaluation.  AI may propose
a candidate contract but cannot modify a frozen contract and use the new value
to pass the same run.

Contract states:

```text
candidate
observational
frozen
retired
```

New hardware, model, quantization, graph, or parallel mode must explicitly
reuse, derive, or replace a contract.  Source-code presence never implies a
`qualified` state.

## 12. First end-to-end MVP

The first scenario diagnoses a change to CPU/NPU transfer or stream handling:

```text
source/environment verify
  -> Ascend framework transfer
  -> CPUInfer + LLAMAFILE lifecycle
  -> SGLang KTEP mapping and active-stream contracts
  -> P2 minimal same-path determinism
  -> common report and artifacts
```

Initial registered capabilities:

```text
env.inspect
source.verify
ascend.framework_transfer
cpuinfer.llamafile_lifecycle
ktep.mapping_contract
ktep.active_stream
p2.minimal_determinism
artifacts.verify
report.generate
```

Full Q2/H2, downstream quality, P3 stability, concurrency, and performance may
still be called through generic/legacy adapters, but they are not first-version
native gates.

## 13. Success criteria

The MVP is successful when:

1. an AI can discover and understand registered capabilities;
2. an AI can submit a custom plan without changing runner code;
3. adding a gate requires registry and adapter work, not runner branching;
4. local and A3 commands preserve raw logs and source/input provenance;
5. an interrupted run remains inspectable and eligible gates can resume;
6. the historical transfer-failure class is assigned to the transfer path
   before expensive model/quality gates;
7. a P2 minimal real-model check verifies that primitive success survives the
   serving lifecycle;
8. the workbench remains usable without a Web service or embedded AI;
9. Round 2–5 history and immutable tags are not rewritten;
10. observed maintenance cost is lower than repeated manual orchestration.

The MVP does not succeed merely because all tests pass.  It must demonstrate
useful failure localization and reproducible evidence.
