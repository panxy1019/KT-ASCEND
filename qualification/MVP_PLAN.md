# Qualification Workbench MVP Plan

Status: `APPROVED_FOR_IMPLEMENTATION`

This plan implements the design in `qualification/README.md` without moving or
rewriting historical `migration_bridge` assets.

## 1. Source baseline

Development starts from repository `main` at or after:

```text
parent main observed during design: 5feb57f91aa9c786d2bc0002585a8958a456eb0f
Round 5 baseline tag: round5-upstream-aligned-tp1-a3-verified
Round 5 runtime parent: be5b48621db32fb894402e3e886060dea263cdc9
Round 5 SGLang child: 412219b4b05a888808f0b88d2963a8b6164a0fda
```

Implementation must fail clearly when a required submodule object cannot be
resolved.  It must not silently test a different SGLang checkout.

## 2. Work packages

### WP0 — Inventory and fixtures

Deliverables:

- machine-readable inventory for the first nine capabilities;
- classification of each referenced existing test as active/tool/incident or
  archived;
- immutable Round 5 baseline descriptor;
- example local and A3 executor configuration;
- example subject and run-plan documents.

Acceptance:

- every registered gate names its source file, failure layer, dependencies,
  timeout, expected parser, and test lifecycle state;
- no production or historical test source is modified;
- baseline parent, child, model, placement, GGUF, tokenizer, corpora, numerical
  contract, and quality hashes are representable even if an asset is absent.

### WP1 — Standard-library core

Deliverables:

- `python -m qualification` entry point;
- dataclass models for capability, backend, gate, suite, run plan, command,
  event, gate result, and run summary;
- TOML registry loader;
- JSON plan/result serialization;
- capability matching and applicability evaluation;
- status and failure-layer enums;
- `list`, `describe`, and `plan validate` commands.

Acceptance:

- core imports without torch, torch_npu, SGLang, Paramiko, PyYAML, or Pydantic;
- malformed and unknown fields fail with an actionable path and message;
- round-trip JSON preserves all evidence identity fields;
- duplicate IDs and dependency cycles are rejected;
- capability mismatch returns `NOT_APPLICABLE`, not `FAIL`.

### WP2 — Event, state, and artifact core

Deliverables:

- run-ID generation;
- append-only `events.jsonl` and `commands.jsonl`;
- atomic `state.json` and `summary.json` writes;
- raw stdout/stderr capture;
- artifact directory builder;
- canonical hash helpers and `SHA256SUMS` generation;
- `inspect`, `events`, `logs`, and `report` commands.

Acceptance:

- abrupt child-process termination leaves a parseable event log;
- an interrupted run is distinguishable from a failed gate;
- raw stdout/stderr is preserved if the parser raises;
- generated evidence never embeds secrets or full inherited environment;
- report output can be recreated solely from normalized artifacts.

### WP3 — Local executor and generic adapters

Deliverables:

- local subprocess executor;
- generic command adapter;
- pytest adapter;
- legacy-command adapter;
- timeout, cancellation, environment delta, and working-directory handling;
- dry-run command rendering.

Acceptance:

- argv is executed without shell interpolation by default;
- shell mode, when explicitly needed by a legacy tool, is visible in the plan
  and command evidence;
- timeout and operator cancellation have distinct results;
- exit code, start/end time, cwd, argv, and safe environment delta are recorded;
- adapter normalization cannot change the raw command result.

### WP4 — Environment and source identity

Deliverables:

- `env.inspect` native capability;
- `source.verify` native capability;
- parent/submodule commit and dirty-state collection;
- Python/package/runtime version collection through adapters;
- model/GGUF/placement/corpus/contract identity checks;
- subject-versus-baseline comparison in the report.

Acceptance:

- a wrong SGLang gitlink or checkout is a source failure with both expected and
  actual SHAs;
- missing heavyweight assets are `BLOCKED`, not `FAIL`;
- unavailable torch_npu is reported without preventing core CLI use;
- moving `main` never changes the immutable Round 5 baseline descriptor.

### WP5 — Remote executors

Deliverables:

- SSH executor using the system SSH client;
- Docker exec executor;
- composed SSH + Docker executor;
- command streaming into the common event/log layer;
- helpers for remote run directory, PID, server status, and framework-owned
  service stop;
- connectivity and environment preflight.

Acceptance:

- all actual remote commands are visible before or during execution;
- SSH disconnect produces `INFRA_ERROR` unless the remote durable state proves
  the command completed;
- reconnect can inspect a remote run directory;
- the framework can stop a service it started without selecting unrelated
  processes by name;
- use outside these helpers remains possible; this is not an AI sandbox.

### WP6 — First Ascend and KTransformers gates

Deliverables:

- adapter for Round 5 Ascend framework-transfer pytest;
- adapter for retained R2A CPUInfer/LLAMAFILE lifecycle subset;
- adapter for SGLang KTEP mapping and active-stream tests;
- normalized pass counts, durations, skip reasons, and failure excerpts;
- upstream source/SHA metadata for every reused test.

Acceptance:

- known Round 5 results normalize to 5/5 transfer, 18/18 selected R2A, and
  4/4 KTEP contract passes when the exact retained suites are selected;
- pytest collection failure is not reported as a product assertion failure;
- use of the wrong Python environment is classified before model serving;
- stale private-attribute tests remain excluded and documented.

### WP7 — P2 minimal serving gate

Deliverables:

- P2 server lifecycle adapter using the existing Round 5 launcher/profile;
- readiness check;
- minimal frozen prompt selection;
- same-process repeated greedy output and prefix validation;
- deterministic summary normalization;
- optional promotion to the existing full same-path runner;
- explicit server-stop operation.

Initial smoke proposal:

```text
prompts: v_en_01, v_struct_01
repeats: 3
tokens: 32
prefixes: 8, 16
```

The proposal is a smoke workload, not a replacement for the frozen full P2
gate.  Full qualification continues to use 9 prompts × 10 repeats × 64 tokens
and 8/16/32 prefixes.

Acceptance:

- output instability reports `FAIL` at `serving_determinism`;
- unavailable model/GGUF/placement reports `BLOCKED`;
- startup, readiness, request, and shutdown failures are distinguishable;
- normal serving runs do not enable tensor dumps or deep timing by accident;
- a successful smoke result explicitly states that full P2 remains a separate
  gate unless it was also executed.

### WP8 — AI-oriented planning templates

Deliverables:

- `transfer_change` template;
- `routing_change` template;
- `cpu_kernel_change` template;
- `serving_change` template;
- changed-file rule suggestions;
- example AI-authored custom plan;
- rationale and skipped-test fields in reports.

Acceptance:

- templates are suggestions and can be overridden;
- no file rule changes an acceptance threshold;
- a custom gate sequence can be submitted without editing Python core;
- the report distinguishes framework suggestion from AI/user decision.

### WP9 — Value validation

Deliverables:

- one clean Round 5 baseline run;
- one controlled primitive-transfer failure or safe fixture failure;
- one P2 serving-determinism failure replay where practical, otherwise the
  retained historical artifact plus a controlled parser/contract fixture;
- comparison of manual and workbench diagnosis steps;
- flaky/repeatability record across at least three runs;
- go/no-go review for expansion.

Acceptance:

- the controlled primitive failure is assigned no later than `transfer`;
- higher-cost P2 is not required to identify a primitive failure;
- primitive success followed by P2 failure points to a later layer;
- repeated clean runs produce equivalent normalized results;
- maintenance and operator friction are recorded honestly;
- expansion is not approved solely because the framework code exists.

## 3. First registry set

| ID | Role | Level | Failure layer | Existing source |
|---|---|---:|---|---|
| `env.inspect` | native active | L0 | environment | new core |
| `source.verify` | native active | L0 | environment/build | new core |
| `ascend.framework_transfer` | active gate | L1 | transfer | `kt-kernel/test/round5/test_ascend_framework_transfer.py` |
| `cpuinfer.llamafile_lifecycle` | active gate | L0/L1 | cpu_expert | retained R2A selection |
| `ktep.mapping_contract` | active gate | L0/L1 | routing_mapping | SGLang aligned contract tests |
| `ktep.active_stream` | active gate | L0/L1 | runtime_primitives | SGLang aligned contract tests |
| `p2.minimal_determinism` | active smoke | L1 | serving_determinism | existing same-path runner |
| `artifacts.verify` | native active | L0 | environment | new core |
| `report.generate` | native tool | L0 | n/a | new core |

## 4. Deferred registry candidates

These are supported by the architecture but are not required for MVP closure:

- P1/P2/P3 full same-path and clean restart;
- Q2/H2 teacher-forced pairwise qualification;
- route coverage and CPU-not-hit;
- frozen downstream quality;
- P3 stability/memory campaigns;
- SGLang `bench_serving` performance mode;
- torch_npu/CANN profiling adapters;
- Graph, Deferred, Dynamic, MTP, Speculative, W8A8, MXFP4;
- TP2/HCCL, multi-NPU, multi-node;
- KML production and alternate CPU/GPU architectures;
- environment provisioning and model/dataset resolvers;
- dashboard or historical result service.

## 5. Review checkpoints

Implementation should pause for review after:

1. WP0–WP2: schemas, events, and artifact shape;
2. WP3–WP4: local usability and provenance;
3. WP5–WP7: first real A3 end-to-end run;
4. WP9: evidence that the workbench reduces diagnosis cost.

At each checkpoint, interfaces may be simplified.  The design intentionally
does not require completing every deferred feature before it becomes useful.

## 6. Explicit non-goals for MVP

- replace pytest, SGLang benchmark tools, or Round 3–5 analyzers;
- run every historical experiment on every change;
- infer a source-code root cause automatically;
- provision or mutate host drivers and CANN installations;
- enforce AI permissions;
- publish performance claims;
- promote candidate thresholds automatically;
- make Ascend qualification a mandatory GitHub merge gate;
- rewrite immutable evidence or tags.
