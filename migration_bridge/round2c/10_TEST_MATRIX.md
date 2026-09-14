# Round 2C Test Matrix

| Test | Result |
|---|---:|
| Round 2B exact final SHA `6f50c378...` frozen/tagged | PASS |
| fixed placement bool mask | PASS |
| `E_cpu ∩ E_npu = ∅` | PASS |
| `E_cpu ∪ E_npu = E_all` | PASS |
| CPU-only routing | PASS, NPU exact zero |
| NPU-only routing | PASS, CPU exact zero |
| mixed CPU+NPU routing | PASS |
| reversed mixed order | PASS |
| identity mapping | PASS |
| `[2,0,3,1]` mapping | PASS |
| reverse mapping | PASS |
| alternate deterministic mapping | PASS |
| qlen 1 / 8 / 32 | PASS |
| weights 1/0, 0/1, 0.5/0.5, 0.99/0.01 | PASS |
| router weights applied exactly once | PASS |
| unselected contribution zero | PASS |
| independent float reference | PASS |
| sequential Hybrid | PASS |
| overlapped Hybrid | PASS |
| sequential vs overlapped | bitwise PASS |
| real CPU/NPU interval intersection | PASS, lower bound 1.346712 ms |
| shared expert contract | `BYPASSED_WITH_CONTRACT` |
| 1,000 mixed cycles / RSS | PASS / delta 0 |
| wrapper recreate ×20 | PASS |
| stream lifecycle ×100 | PASS |
| invalid placement/routes/stream | PASS |
| Round 2A regression | PASS, 21/21 |
| Round 2B regression | PASS, 23 passed/1 expected skip |
| no full model / TP / HCCL / Graph / deferred / dynamic placement | PASS |
| no SGLang or model-code changes | PASS |

Final status:

```text
HYBRID_MOE_SINGLE_LAYER = A3_VERIFIED_READY
```
