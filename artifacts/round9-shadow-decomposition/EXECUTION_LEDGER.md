# Execution ledger — docs/plans/2026-09-22-shadow-decomposition.md
Base a5b26444e. User authorized revised08 and parallel09 2026-09-22.
E0 IN_PROGRESS; E1 fixture; E2 observation; E3 offline+2physical; E4 freeze.
Ruling: minimal defaultOFF core instrumentation explicitly described in revised11 is covered by current revision/execution request; immutable oldremote builds untouched. Cost: new snapshot must prove OFF semantics.
Ruling: reuse real Layerwise provider with captured input, BF16 oracle; v20 mixed output not expectedexact. Cost: scope limited to input-derived shadow component, nofullprefill equivalence.
Ruling:09 evidence-only independent agent, no extra physicalbudget. M2/M3 not entered.

E0 complete: originalprovider, BF16/mixed reference distinction andreservation audited. E1 capturedinputA andderivedB230→231 defined. E2 phase-only defaultOFF instrumentation andparent/child/device reporting implemented.19 new/relatedobserver tests andASTbaselineequivalence pass.
Independent solmediumreview found5important issues: immutableOFF/ON, phasecostsummaries, outcomeclassification, casecorrelation, peakprojectionaccounting. All RED→GREEN; added exactphysicalOFF/ONresultcomparison, tensorstorage metadata andloadedprovideridentity guards.
Ruling: prior v20 mixed output is provenanceonly, new allBF16oracle follows existingLayerwise math. Cost: no capturedprefill equivalence claim.
Ruling: snapshot contains minimumdefaultOFF phase inserts into dynamic_adapter/expert_source andnew dependency copiedbyoverlay installer; oldsnapshots untouched. Cost: actualNPUoff/on needsnewgate.
Ruling: Host-only torch regression uses one1CPU/8GiB/120s imagecheck withoutNPU, separatefrom2physicalstarts; localPythonmissingtorch collectionerror preserved. Cost: hostcheckdoesnotproveNPUbehavior.
63local tests pass;78frozenCPU tests pass;sourcecompile/diffcheckpass.
E3 physicalBLOCKED: initialCPUchecks andformalOFFgate reportselectedCPUsbusy;quantifiedidle40.576%<strict80%.No componentstarts,no fullmodel/HTTP, no cacheadvice. No bypass/othercontainerstop/timers. E4device costs/numeric remainNOT_RUN/null.
09 complete:19rows andcanarycontract, sourceidentity andhistorynotrevalidated distinctions. M2/M3notentered.
Final ruling: hardware numeric/allocation/event completion/physicalOFFON remainunknown perreviewer; no PASS claim. NextfrozenOFFONrequiresfreshresourceadmission. No backgroundmonitor.
