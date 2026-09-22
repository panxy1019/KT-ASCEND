# Round 6.1 explicit Deferred replay oracle

Evidence state: `A3_VERIFIED`.

## Oracle construction

The diagnostic C path does not use the production pending-state dictionary for
the deferred contribution. It:

1. uses the same captured layer-7 hidden state, IDs, and weights;
2. computes the immediate L7 task normally;
3. computes the deferred L7 task into a separate pinned tensor;
4. keeps that tensor out of the L7 visible result;
5. drains the CPU task before diagnostic injection;
6. pops the explicit tensor from a separate diagnostic map;
7. adds it exactly once at the L8 CPU incremental boundary;
8. continues with the same matched token history.

The mechanism is opt-in through `KT_R61_EXPLICIT_REPLAY=1`. With the variable
unset, the production path is unchanged.

## B versus C result

Production Deferred B and explicit replay C have:

```text
common aligned stage/call tensors = 436
exact aligned tensors             = 436 / 436
non-finite values                 = 0
```

The three semantic aliases are also exact:

| Semantic tensor | Elements | B SHA256 | C SHA256 | Result |
| --- | ---: | --- | --- | --- |
| L7 deferred partial | 110,592 | `9987ed67...` | `9987ed67...` | exact |
| L8 incremental input | 110,592 | `9987ed67...` | `9987ed67...` | exact |
| L8 incremental output | 110,592 | `aa2156da...` | `aa2156da...` | exact |

Final full-vocabulary logits are exact across all 102,400 elements:

```text
B SHA256 = 98177a8cb38a920d0f70ead9ae60b947a9a56ed4342fbc16933fb32f89145c33
C SHA256 = 98177a8cb38a920d0f70ead9ae60b947a9a56ed4342fbc16933fb32f89145c33
max abs  = 0
rel L2   = 0
top-1    = 21
margin   = 11.5
```

By contrast, baseline A is deterministic but has SHA256
`273d708423e0828c5f2c98162ec482f1436ea234079e8252710e90dd314ae257`
and margin 10.75.

## Gate A2

The oracle reproduces both the intended scheduling boundary and the complete
production result without relying on production pending bookkeeping. Therefore:

```text
PRODUCTION_DEFERRED_VS_EXPLICIT_REPLAY = EXACT
ORACLE_FIRST_MISMATCH                  = NONE
GATE_A2                                = PASS
```

This is direct evidence for `INTENDED_DEFERRED_APPROXIMATION`; the final ADR
also considers the runtime controls and whole-request control before resolving
the root-cause class.

## Evidence

- Full normalized matrix: `evidence/explicit_replay/result.json`
- Replay request: `evidence/explicit_replay/request.json`
- Raw tensor index: `evidence/explicit_replay/raw_sha256sums.txt`
- Raw A3 root: `/workspace/results/round6_1-stage-v2-replay`

```text
ROUND6_1_EXPLICIT_REPLAY_ORACLE = PASS
```
