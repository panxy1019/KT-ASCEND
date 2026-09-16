# Four-layer ratio probe — IN PROGRESS

Execution date:2026-09-16,target110.129.0.22/NPU0. CPU32workers/one
pool,CPU affinity0–31,Deferred0,GraphOFF,DynamicOFF,production overlap.
Frozen layers:1,9,17,26. Order:R32→R16→R8. No expert reselection.

The serial driver at `/workspace/results/r65-scale-matrix-v2` entered
`four-R32-det` after all three single-layer compact gates passed. This is not
yet four-layer qualification,nor all-MoE capacity qualification.

Per profile the driver requires:

1. Four frozen S6 prompts:three64-token repeats,8/16/32prefixes,independent
   process restart.28requests total.
2. Same target R64 oracle,512matched-history NUM positions,B_pair2.1875/K32.
   Actual pre-force serving sampler token is used,not CPU argmax tie-breaking.
3. Four8-token routing diagnostics with all26MoE layers accounted for,
   including one lookahead per request. Capture/parity and exact FP32 merge.
4. Actual resident tensor dim0,mapping and dtype through existing model hooks.
5. One frozen CPU expert per selected layer,first real captured hidden rows,
   expert relative-L2≤0.01,10repeats exact,controlled32real-nohit rows;L17T1 too.
6. Independent owned-process guard/reaping audit.

First failure stops the queue and the downstream capacity queue. No threshold
fitting,no concurrent model loading,no business process manipulation.

Final gate state remains **PENDING** until complete artifacts are evaluated.
