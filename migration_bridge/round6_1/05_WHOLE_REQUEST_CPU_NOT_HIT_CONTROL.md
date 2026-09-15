# Round 6.1 whole-request CPU-not-hit control

Evidence state: `A3_VERIFIED`.

The historical QD/HD target rows were not valid whole-request controls because
their matched prefixes could contain CPU and Deferred work. This work package
therefore used a separate diagnostic search that is not part of QD or HD.

## Predeclared search

```text
rule       = ascending single-token IDs, first match
start      = 1000
budget     = 32
first hit  = token ID 1000
attempts   = 1
```

The request was accepted only when every D1 trace submit had zero immediate
CPU routes, zero deferred CPU routes, and the final pending set was empty.

## D1 Deferred and D1_NODEFER

| Field | D1 Deferred | D1_NODEFER |
| --- | ---: | ---: |
| Input IDs | `[1000]` | `[1000]` |
| Submit records | 4 | 4 |
| Immediate CPU routes | 0 | 0 |
| Deferred CPU routes | 0 | 0 |
| Final pending | `[]` | `[]` |
| Generated token | 11 | 11 |

For both captured one-row calls (prefill and generated step), at both selected
layers 7 and 8:

```text
CPU partial = exact zero
routed merge = NPU-owned partial after the declared BF16 cast
max abs = 0
```

The actual generation-decision full logits are exact across all 102,400 FP32
values:

```text
D1 Deferred SHA256 = dc63761c671da706a83085bc6460a5f485842aa7760549d3459bd0f98042d52e
D1_NODEFER SHA256  = dc63761c671da706a83085bc6460a5f485842aa7760549d3459bd0f98042d52e
max abs            = 0
top-1              = 11
```

## Debug-dump interpretation

SGLang also dumps an auxiliary pass keyed by the input token ID while input
logprobs are requested. That auxiliary `input_ids=[1000]` scoring tensor was
not the generation-decision pass and differed across clean processes. It is
not used for the gate. The pass keyed by the generated token 11 corresponds to
the actual decision evidence and is exact, consistent with both APIs returning
token 11. This distinction is retained explicitly to prevent a dump-label
artifact from being misreported as model contamination.

## Evidence

- D1 search: `evidence/cpu_not_hit/deferred.json`
- D1_NODEFER verification: `evidence/cpu_not_hit/nodefer.json`
- Full-logit comparison: `evidence/cpu_not_hit/comparison.json`
- Boundary checks: `evidence/cpu_not_hit/boundary_call5.json` and
  `boundary_call6.json`

```text
ROUND6_1_WHOLE_REQUEST_CPU_NOT_HIT_CONTROL = PASS
```
