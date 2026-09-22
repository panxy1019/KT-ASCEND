# Static / Dynamic abbreviated performance — scope closure

User explicitly requested stopping the current performance campaign and moving
to Deferred D0/D1. Status: **STOPPED_BY_USER_SCOPE_CHANGE**, eight of twelve cells
completed on 110.129.0.22. The interrupted long workload is not scored.

| Workload | Static request seconds | Dynamic request seconds | Evidence boundary |
|---|---|---|---|
| Short | 119.83 / 120.55 | 121.49 / 123.01 | Four clean resource windows; one ABBA only |
| Medium | 245.41 / 246.51 | 245.71 / 243.03 | Turns 0 and 2 failed full-window resource eligibility |
| Long | Incomplete | Not run | No conclusion |

Short median E2E latency is approximately 1.7% worse with Dynamic. Medium
measurements cannot establish an isolated benefit. No claim of comprehensive
Dynamic performance failure or successful formal performance qualification is made.

**Selection: Static Uniform R8-C64, Dynamic OFF.** Dynamic remains implemented
with prior system and quality evidence, but benefit is not established by this
incomplete campaign. The conservative selection is not based on CMMLU score ranking.

The exact controller PID 3714897 was interrupted; its active owned guard reported
complete reaping. Raw files and interrupted long-run evidence remain in:

`/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-development-v1/round7/return22-static-dynamic-abbreviated-v1`

`scope-stop-and-window-audit.json` contains the eight post-audited resource windows.
The window field `formal_eligible` refers only to resource isolation, not completion
of the original formal three-block performance campaign.

Next: Deferred system qualification on Static Uniform (D0 is retained baseline),
then D0/D1 performance and frozen quality guardrail. Default Deferred remains OFF
until evidence supports changing it. Round7 is not yet complete.
