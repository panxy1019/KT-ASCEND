# User-directed performance truncation, 2026-09-17

Remaining performance matrix cancelled by explicit user instruction at ~08:52 UTC.
Original progress and interrupted summary are preserved in performance-contended-v1.
First complete ABBA block (20 shape observations) is separately derived into
performance-first-abba-v1, with source SHA. PASS here means complete abbreviated
collection, not full performance qualification; formal_performance remains false.
No confidence or production speedup claim from a three-block test is permitted.

New driver: tools-v14/continue_authorized_screen.py --completed-first-abba.
State: /workspace/c64-dev/abbreviated-screen-progress.json.
Order: memory audit → R8-C64 canary/extended smoke → R16-C64 quality → R8-C64 quality.

R16-C32 completed naturally: 173/256, invalid0, vs R64 174/256, delta -0.390625pp.
Paired: three regressions, two improvements, bootstrap95% CI [-2.34375,1.171875]pp.
Frozen product gate PASS, no population noninferiority claim.

The old serial controller automatically entered R8-C32 after finishing R16.
Its owned controller guard PID3831166 received SIGTERM after verifying completed
R16 results, implementing the user's R8-C32 resource-priority stop. Completed
R16 evidence is preserved. No business workload is targeted.
