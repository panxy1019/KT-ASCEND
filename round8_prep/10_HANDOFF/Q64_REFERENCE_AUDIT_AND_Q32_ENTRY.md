# Q64 engineering reference audited — Q32 entry, 2026-09-19

Q64CMMLU completed222/256=86.71875%,invalid0. Independent raw re-score
confirmed all256orderedIDs, exactfrozeninputIDs, generationparams, candidateIDs,
response requestIDs/promptcounts, fourfinitecandidate scores, deterministic
argmax rule andmanifestSHA. PredictedA/B/C/D counts65/60/62/69; no collapse.
Actual1689importedPython/nativefilehashes rechecked onhost:0mismatches.
Rawarchive`q64-quality-evidence-v1.tar.gz` SHA256
8942a99c199e69ac3c06731092081365f0c5f8566466490536bf3e2f053e6ec0.

Status **Q64_ENGINEERING_REFERENCE_FROZEN** for thisproject's staticproduct
screen. Not All-NPU equivalence, publicbenchmark parity, formalperformance,
long-lifecycle qualification, Dynamic, Deferred orfullRound8completion.

Predecessors: fullcheckpoint/expertidentity,early/middle/lateoperator gates,
GDN/convprimitives,actual40layermodelbringup,mapping/ownership,sharedcomposition,
4prompt32requestrepeat/prefix,4promptcleanrestart,32distinctpromptrepeat screen.
Known limitation: operator operands synthetic andNPUlocaloraclecompact8slot;
modeltest coversactual64slotforward butnot exhaustiveperexpertcertification.

## Q32 entry frozen, not outcome-selected

Reuse original`04_MEMORY/q32_profile.json`, nested32/256residents onall40layers.
No experts selected fromqualityanswers. Samecheckpoint/tokenizer/120GiBF32GGUF,
LLAMAFILE C64(two32subpoolsNUMA4/5),TP1,batch1,physicalNPU2,Graph/Dynamic/
DeferredOFF. Onlyresidentbudget/mask changes. No additionalconversion.

Q32 orderedgates: freshadmission→modelbringup/structure/shared→sameprocess
repeat/prefix→cleanrestart→32distinctpromptscreen→samefrozenCMMLU256→rawpaired
audit. Reuse unchangedCPUoperatorimplementationevidence explicitly, notnewrun.
Priorqualityrule unchanged: observed Q32−Q64≥−5pp. With222reference,minimum
integercorrect=210/256;209fails. No refit, samplechanges or maskreselection.
Runtime tools now acceptresident32 andauditactualmaskcount ratherthan64constant.
This is test/orchestration adaptation; productionarithmetic unchanged.
