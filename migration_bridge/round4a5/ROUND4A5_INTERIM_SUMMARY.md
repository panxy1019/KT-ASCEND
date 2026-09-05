# Round 4A.5 Interim Summary: P2 Determinism Investigation

Status: `ROOT_CAUSE_ATTRIBUTED; FIX_QUALIFICATION_PENDING`

This document records diagnostic evidence only.  It does not change the frozen
P2 placement, numerical contract, `B_pair`, coverage target, or production
execution semantics.  P3 remains `NOT_RUN` pending full P2 qualification of a
production fix.

## Reproduced blocker

In a clean A3 disposable container, frozen P2 still produced non-identical
greedy outputs in the same serving process:

| Prompt | 64-token unique outputs / 10 | Repeat | Prefix |
|---|---:|---|---|
| `v_en_01` | 9 | fail | fail |
| `v_struct_01` | 10 | fail | fail |

The failure also remains with `SGLANG_KT_HYBRID_NO_CPU_STREAM=1`, so it is not
an overlap-only issue.

## First valid stage capture

For a matched teacher-forced `v_en_01` history, all captured Layer 1, 9, and
17 values matched byte-for-byte.  At Layer 26, the input, router IDs/weights,
and NPU partial matched, while the CPU output and merged routed output differed.
This establishes a first observed divergent stage, not a completed cause.

## Layer bisection

Diagnostic placements preserve the original P2 CPU experts.  Each uses ten
64-token `v_en_01` sequential-control repeats.

| CPU layers | Result | Unique hashes |
|---|---|---:|
| `{1}` | exact | 1 |
| `{9}` | exact | 1 |
| `{17}` | P1-established control | 1 |
| `{26}` | exact | 1 |
| `{17,26}` | exact | 1 |
| `{1,17}` | nondeterministic | 5 |
| `{9,17}` | nondeterministic | 3 |

Thus the current minimal known failing sets are `{1,17}` and `{9,17}`.

## A/B results

- Per-wrapper CPUInfer/WorkerPool: four distinct instances were created; full
  P2 remained nondeterministic (8 unique `v_en_01` hashes).
- Private LLAMAFILE scratch and private TP merge-output diagnostic buffers:
  each remained nondeterministic in `{1,17}`.
- CPU worker count changes the `v_en_01` outcome for `{1,17}`:
  1/2/4 workers were exact; 8 workers failed with two hashes; frozen P2 uses
  16 workers.
- Full P2 with one CPU worker made `v_en_01` exact but left `v_struct_01`
  nondeterministic.  It is a localization control, not a fix.

## Root-cause attribution: raw ACL transfer integration

The Ascend-specific path copied CPU-expert inputs and outputs with raw
`acl.rt.memcpy` pointer calls.  Those calls bypass framework-owned NPU stream
and tensor-lifetime tracking.  This is now the attributed integration defect:
the raw host-pointer transfer path can expose non-deterministic visibility or
ordering across the SGLang/CANN request lifecycle.

Evidence under the same frozen P2 placement, 16 CPUInfer workers, one request,
sequential-control mode, 64 generated tokens, and ten repeats:

| Variant | Prompt | Result | Unique hashes |
|---|---|---|---:|
| raw ACL D2H destination cloned before CPU task | `v_en_01` | fail | 7 |
| framework-managed D2H + H2D (`KT_DEBUG_NPU_TORCH_TRANSFERS=1`) | `v_en_01` | exact | 1 |
| framework-managed D2H + H2D (`KT_DEBUG_NPU_TORCH_TRANSFERS=1`) | `v_struct_01` | exact | 1 |

The first control rejects D2H destination-buffer ownership as the sole cause.
The second and third controls change only the transfer API and make both prior
failure classes exact.  They also agree with fixed-input direct CPU replay,
which was exact.  This attributes the P2 blocker to the raw ACL transfer
integration rather than LLAMAFILE arithmetic, layer placement, or a shared
CPU scratch buffer.  It does not attribute a defect to CANN itself.

## 修复后的复验状态

The framework-managed transfers are now the normal Ascend CPU-expert path; the
diagnostic environment gate has been removed.  Requalification passed the P2
same-process/restart exact gates, Q2/H2 frozen numerical contract, 4/4-layer
and 16/16-expert coverage, and the CPU-not-hit control.  The frozen downstream
quality input is absent from both the repository and persisted A3 artifacts,
so the quality A/B cannot be validly rerun yet.  P2 is consequently not marked
fully qualified and P3 remains blocked.  See `06_P2_FIX_REQUALIFICATION.md`.

## Retained Round 2B correction

The previous standalone D2H/CPUInfer/H2D regression was traced to its test
allocating BF16 output storage for LLAMAFILE's FP32 routed output.  Correcting
the test buffer dtype makes the pipeline pass; this is separate from, and does
not resolve, P2 same-path nondeterminism.

Evidence hashes and commands are indexed in `evidence/README.md`; detailed
findings are in `01_FIRST_DIVERGENCE_CAPTURE.md`, `02_LAYER_BISECTION.md`,
`03_PARALLEL_EXECUTION_AB.md`, and `05_RAW_ACL_TRANSFER_ROOT_CAUSE.md`.
