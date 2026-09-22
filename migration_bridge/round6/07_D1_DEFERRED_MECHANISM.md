# Round 6 D1 Deferred Mechanism Qualification

Evidence state: `A3_VERIFIED` for topology, scheduling and pending/consumer
correspondence.  Numerical exactly-once reconstruction remains a later D1
gate.

## Frozen profile

```text
model: deepseek-ai/DeepSeek-V2-Lite
revision: 604d5664dddd88a0433dbae533b7fe9472482de0
TP: 1
NPU: NPU0 only
CPU backend: LLAMAFILE / CPUInfer
Graph / Dynamic Placement / MTP / Speculative: OFF
layers: {7, 8}
deferred producer: 7
block terminal: 8
max deferred experts per token: 1 on layer 7, 0 on layer 8
placement SHA256: 23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c
GGUF SHA256: 4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f
```

The production server log explicitly reported:

```text
Disabling KT deferred experts for terminal CPU block layer 8;
the immediately following layer has no KT CPU wrapper
```

The first D1 generation request completed successfully on the default
production overlap path.

## Observed empty-pending defect

The default-off JSONL trace initially showed:

| Layer | Submit events | Real CPU deferred | Outgoing pending | Incremental input |
|---:|---:|---:|---:|---:|
| 7 | 134 | 13 | 134 | 0 |
| 8 | 134 | 0 | 0 | 134 |

The selector may defer an accelerator-owned route.  `submit_forward()` used
only `deferred_ids is not None`, so those backend no-ops were still queued and
advertised as pending to the next layer.

## Fix

Parent commit:

```text
0fbbc572b1b43894a3205c39f24a92132ebf302b
fix(round6): suppress empty deferred CPU tasks
```

On Ascend and standalone CPU paths, the pinned route buffer is already
host-visible at the decision point.  The implementation now submits deferred
work and sets pending only when the selected deferred IDs include at least one
CPU-owned expert.  The CUDA asynchronous path is unchanged because a host read
there would race its outstanding D2H copy.

The trace facility is disabled unless `KT_DEBUG_DEFERRED_TRACE_FILE` is set,
is capped by `KT_DEBUG_DEFERRED_TRACE_LIMIT`, and records routing/scheduling
metadata only.

## Post-fix A3 result

Four S6 prompts covered English, Chinese, math and structured JSON.  They
generated 32 tokens each.  Prefill plus decode produced:

| Layer | Submit | Rows | Immediate CPU routes | Deferred CPU routes | Pending | Incremental | Sync allow-pending |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 134 | 176 | 162 | 13 | 13 | 0 | 13 |
| 8 | 134 | 176 | 137 | 0 | 0 | 13 | 0 |

Additional invariants:

```text
false pending events: 0
missed pending events: 0
producer pending == adjacent consumer incremental: 13 == 13
layer 7 CPU experts observed: {16, 38, 41, 50}
layer 8 CPU experts observed: {8, 25, 33, 55}
pre-fix vs post-fix token IDs: exact for all 4 prompts
```

Artifacts:

```text
evidence/d1-mechanism/deferred.jsonl
  SHA256 e4874a945cbd0beb3cf1179e053e1c1c53849f9503e5bcd939be7e581189776f

evidence/d1-mechanism/summary.json
  canonical payload SHA256 82ba4e7720e93b38203352ff16fd8df89c67656410cc8ac9081395e595a80f5a
```

## Gate

```text
D1_TOPOLOGY_GUARD = A3_VERIFIED
D1_REAL_DEFERRED_WORK = A3_VERIFIED
D1_EMPTY_PENDING_SUPPRESSION = A3_VERIFIED
D1_ADJACENT_PENDING_CONSUMPTION = A3_VERIFIED
D1_NUMERICAL_EXACTLY_ONCE = NOT_YET_RUN
```
