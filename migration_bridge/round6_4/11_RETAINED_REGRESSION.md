# Retained regression on Round6.4 actual runtime

No production change has been made during this round. Qualification uses
the actual source/extension hashes in each A3 provenance file, not a claim
that the documentation branch HEAD itself is the remote checkout.

| Gate | New Round6.4 evidence | State |
|---|---|---|
| R2A CPU detect/GGUF/routing/lifecycle | r64-cpu-regression-v1,18PASS;3 inherited private-API exclusions | A3_VERIFIED |
| R2B framework BF16/FP32/int64 transfer | r64-preflight-v1,5PASS including1000-roundtrip tests | A3_VERIFIED |
| R2C/SGLang mapping/ownership contracts | r64-preflight-v1,7PASS | A3_VERIFIED |
| Deferred selection | r64-preflight-v1,7PASS | A3_VERIFIED |
| Round3 real Layer17/Expert8 | r64-round3-regression-v2 | A3_VERIFIED |
| Frozen nonDeferred P1/P2/P3 compact | r64-compact-v1/v2,20requests per profile,old references exact | A3_VERIFIED |
| D1 own-path lifecycle/determinism | r64-d1-runtime-v1/r64-d1-lifecycle-v1 | A3_VERIFIED |

These are focused retained suites, not a fabricated rerun of all historical
test counts. D1 final quality and stability remain separate, pending gates.

## Round3 anchor

v2 guard EXIT_ZERO. Real CPU routes2, distinct framework streams and staging,
overlap mode, all finite. KTEP versus explicit CPU+NPU reference max_abs0 and
relative-L2=0. CPU-not-hit row exact. Mixed versus all-NPU relative-L2
0.003270712430139086; final shared-once relative-L2=0.003124126269339239.
No cross-backend bitwise equality is claimed.

Provenance SHA:
`180939c76b6f23cd9505a3c78e7e8ebc1a695882a691ae164eb094af56e550ac`.
Evidence: `evidence/evidence-round3-v2/`.

## Initial harness environment failure retained

v1 reached the compiled mask/remap helper, then failed because the new shell
overwrote the Python path established by CANN set_env.sh: `No module named tbe`.
This is not a reason to install host libraries or silently turn on model Graph.
The wrapper was corrected to preserve the existing path, exactly as the
production launch script does. The same retained Python test then passed.
No model/kernel/mapping implementation change; no Graph qualification claim.

Original error and guard result remain in `evidence/evidence-round3-v1/`.
The separate batch4 OFF repeat failure is **not** hidden by this regression
PASS; it remains explicitly unqualified in09_BATCH_QUALIFICATION.md.
