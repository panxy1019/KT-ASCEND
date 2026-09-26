# Round10 current execution

Updated: 2026-09-26 UTC

Status: `ROUND10_AB_NARROW_QUALIFIED`

Round10 T0–T6 is complete under the frozen serial TP1, input≤128 narrow contract. HFINAL equals the H0 Hybrid129 static configuration: C128/MXFP4 CPU experts, Q32 BF16/NZ29 resident NPU experts, chunk32, PLE ASYNC and Dynamic OFF. No new performance candidate is promoted.

T1 qualified Hybrid policy, deterministic canaries, edge lengths through 128, multi-step Decode, cancellation and same-process reuse. T2 froze host-only cost attribution and a real replay fixture. T3 verified the current native Kunpeng worker path against an independent same-precision reference and retained the existing backend. T4 proved the current Hybrid→Dynamic transaction, then rejected enablement because its 64-token median was 4.558176× H0. T5 independently qualified current GDN, QSA and PLE semantics/state and selected no B-path performance candidate.

T6 v107 completed 8/8 generations and 1/1 abort with exact matching token outputs against the corresponding v95 rows. Its 64-token measurements were 5.409330s and 5.832713s, median 5.621022s versus historical H0 median 5.188200s (1.083424×); this is reported as same-configuration repeat dispersion and does not support a speedup claim. The supervisor passed, guard acknowledged exit, the owned container was removed, and physical12 is empty. The outer controller's post-cleanup receipt filename bug is preserved and reconciled; its generator regression is fixed locally without another model start.

Final Round10 budget: 4 model starts, 36 generations, 4 abort controls, zero automatic retries; T3 used 6 component processes and T5 used 3. Targeted cache retirement covered 179 unreferenced project files without global drop_caches. Automation remains paused. No push, merge or tag occurred.

Primary final evidence: `artifacts/round10/ROUND10_FINAL_STATUS.json`, `T6_FINAL_REGRESSION.json`, `RESOURCE_RETIREMENT_RECEIPT.json`, and `docs/reports/2026-09-26-round10-final-zh.md`. Live 129/130 Layerwise boundary, long context, concurrency and deployment remain outside qualification.
