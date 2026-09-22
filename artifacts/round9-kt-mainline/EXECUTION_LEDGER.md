# SDD ledger — plan: docs/plans/2026-09-22-round9-kt-mainline-source.md

User authorized execution 2026-09-22. Spec: supplied plan, revised by prior review (scope T0–T4 + decision + T6).
Pre-flight T0→T4: v61 explicit ASYNC profile differs from global OFF; preserve explicit profile.
Pre-flight T2→T4: real activation missing; develop harness first and bounded capture in T4, then replay.
Pre-flight T3→T4: host timeline cannot prove device overlap; device timing remains UNKNOWN unless collected and correlated.
Ruling: native worktree tool targets wrong app repository; created git worktree from exact baseline — avoids original checkout mutation — cost if wrong: branch metadata requires manual association.
Ruling: reuse one established mixed oracle (quantized CPU bank and BF16 NPU) — separates numerical contracts — cost if wrong: real replay fails instead of silently widening tolerance.
Ruling: coarse provider spans precede internal kernel timing; no native rebuild — limits instrumentation — cost: worker packing/GEMM split may stay UNKNOWN.
Ruling: T5 deferred, no scheduling changes — approved revised scope — cost: no speedup candidate this round.
Ruling: extra component/capture/disk budgets fixed in profile-budget.json — bounds ancillary work — cost: partial evidence if exceeded.
T0 in progress: 203 source files matched; monitor prefix matched, final three records appended. NUMA2 startup denied at about31.7GiB; no NPU process started. Prior approved project-file advice path under review.

T0 evidence: 203 source +2323 overlay files zero mismatch, exact training ID running. Prior-approved scoped FADV_DONTNEED on179 own checkpoint/bank files restored NUMA2 to139GiB, all inode/mtime/size unchanged; no global cache advice. Fresh full-model sample passed, baseline HBM2878MiB, no training NPU PID.
T1 source map:13 files, eager GDN/QSA paths confirmed; native extension614ae60f...; KT-SGLang f02f304a... port. Official pages WEB_REFERENCE_ONLY.
T2/T3: replay identity/scope tests RED→GREEN. Seven targeted tests pass; integration34 host tests pass. Frozen container test namespace initially absolute symlink failed collection; corrected relative symlink and30 tests passed. This is not an NPU failure/retry.
T4: v62 first of2 model starts. OFF base fixture output271, wall69.2337s; second request64 input/4output running. Request budgets counted before submissions.
Ruling: captured true decode row0 from at most32 physical rows (possible Ascend padding), retain original physical row count — avoids capturing padding as real tokens — cost: cannot infer full physical-batch throughput from replay.
Ruling: sampled layer0/3 only, event pairs capped64 and span buffer128/batch, no added synchronize for timing — bounds perturbation — cost: partial attribution and uncalibrated device duration only.

Final fresh-context code review (gpt-5.6-sol medium): four Important findings fixed in one RED→GREEN pass: provider behavior preserved on timing/sink failures; decode kind requires>=2 requested tokens and verifies actual artifact (EOS unchanged); empty/malformed trace yields incomplete; replay provenance/padding/layout tampering rejected.39 related tests pass.
Final: Ruling: unavailable device timing disables further pairs, records bounded observer errors and keeps original compute/exception semantics — observer must not change inference — cost: attribution may be incomplete.
Final: Ruling: no forced ignore_eos; early EOS becomes NO_DECODE_EVIDENCE — explicit plan requires normal EOS — cost: real capture may remain unavailable within8requests.
Final: minor (deferred): direct test-file invocation has unittest.main before later test classes; module discovery used throughout includes all classes.
Final: minor (deferred): HTTP probe unused hashlib import.
Final: Ruling: pending events, physical duration fidelity, sample perturbation, padding semantics and numeric replay are physical evidence obligations, not assumed successes — wait for v63/replay — cost: may end PARTIAL.
Final: Ruling: worker-internal packing/GEMM, DRAM bandwidth and cross-stream absolute overlap remain UNKNOWN — observer has no native timestamp/calibration API — cost: no defensible full critical-path ranking.
V62 normal64+4 complete, actual3 decode forwards, wall288.1502s. Container stopped by exact recorded ID. NUMA6 then below48GiB; same previously recorded owned files advised DONTNEED after stop, second preflight must pass. No hardware retry or limit change.

Physical capture:10prompt/8outputs, first actual decode row hasphysical_rows1,actualNZ29. Found real contiguous resident0–31 versus oldsynthetic stride16 fixture. Ruling: replay actual captured/frozen q32-physical mapping (0–31), retain legacy synthetic spread mask in non-replay mode — otherwise precision split would be wrong — cost: replay snapshot distinct from v63 observation build. Regression real-map RED→GREEN; no physical replay attempt or model restart consumed. Active v63 immutable source remains untouched.

Final physical closure: v63 CANCEL/REUSE PASS with339.116509s send→seen and40.032ms seen→cleanup; limitation explicitly retained. v20 real replay PASS first attempt, full-model comparison0, repeated3 hashes exact.49 related tests pass; compileall/diff check pass. No observer errors/drops. T0–T4+decision+T6 complete; T5 not entered. Attribution remains partial, no behavior change selected. Next proposal only: expert prepare decomposition. Used2/2 model starts,7/8 HTTP,3/3 profiles,1/2 component starts. All owned containers exited; training exactID remains running, zero observed active training NPU usage. No push/merge/tag or timers.
