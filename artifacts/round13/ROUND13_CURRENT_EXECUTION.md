# Round13 Current Execution

- Status: `ROUND13_PARTIAL_WITH_GAPS`; B/F request gain qualified, A mirror coverage gap preserved.
- Parent: Round12a v174/v177 qualified KT compute identity.
- Completed offline: baseline/workload/resource contracts; deterministic four-placement definitions; actual tokenizer DEV fixture; P0 observer/driver/snapshot tests.
- Prelaunch failures: v184 immutable-tree log placement; v186 Round13 monitor allowlist. Neither created a model process.
- Device evidence: v187 started one S0 process and one DEV generation attempt, then failed because the Expert observer did not accept KTEP's keyword `layer=` call. No baseline result was produced.
- Cleanup: v187 guard ACK, owned container removed, physical12 empty; evidence fully hashed; targeted registered-file cache retirement PASS.
- Budget used: P0 1 model process / 1 generation; repair reserve not yet consumed; automatic retries 0.
- Next: build/test v188 with full `*args/**kwargs` passthrough, fresh 2x60s admission, one modified validation run; then run independent P0 block B v185 or a fresh rebuilt B identity.


## P0 fresh profile disposition (v188/v189)

- v188 and v189 both completed with status `PASS_ROUND13_P0_PROCESS`; each guard acknowledged cleanup, the owned container was removed, physical12 was empty, evidence was fully hashed and archived, and the registered read-only bank/checkpoint cache was retired only afterward.
- Fresh DEV route assignments: 3,180 across the two independent blocks. The frozen profiles are published in `PLACEMENT_PROFILES.json`; the earlier pre-profile definitions remain in `PLACEMENT_PROFILES_BOOTSTRAP.json`.
- Two-block attributable maxima: GDN 60.79% / 62.16%, QSA 66.02% / 64.47%, Expert 21.03% / 20.69%, PLE 0.43% / 0.42%. Parent spans and child spans are not added.
- CPU Expert exposed wait reached 6.70% / 6.48%, below the 10% C-line trigger. C is therefore not eligible for a device experiment and will be frozen as `C_DEFERRED_WITH_REASON` after A/B disposition.
- B hotspot candidate pool is GDN or QSA. The single candidate will be selected only after source-level eliminable-work review.
- Budget after v189: P0 2 processes / 8 generation attempts; repair reserve 1 process / 7 generation attempts; total 3 processes / 15 generation attempts.
- Next: adapt and component-qualify q=0/q=512/nonuniform placement, then execute A in the frozen mirror order.

## A static-placement component gate and model entry

- Frozen global-budget placements: uniform/frequency/front-loading/random(seed42), each exactly 1,536 routed experts and nominal BF16 14.0625 GiB. The fresh profile document SHA-256 is `27987bceb25506c9ab01115c328f519c14ecb24fbce19e89967cc5cea063ecad`.
- Real component coverage passed for q512 pure NPU, q0 KTEP CPU-only, q242 mixed Frequency, and q44 mixed Random. Each used the actual MXFP4 bank, C128 NUMA2/3/4/7, BF16/NZ29 NPU tensor where nonempty, independent reference, and exact repeat hash.
- Component evidence status: `PASS_ROUND13_A_COMPONENT_GATE`; five component processes were consumed, including the v194 environment failure and v195 modified repair validation. v190 failed before process creation and consumed no process budget.
- Physical12 is empty after all four successful component archives and targeted cache retirements. No unchanged automatic retry occurred.
- The A model queue is frozen as v196..v203 in mirror order U/Frequency/Front/Random/Random/Front/Frequency/U, 24 generation attempts per process. Local model/gate tests: 24 PASS. Next execution is v196 after fresh admission.

## A model integration and bounded-budget state (v196-v208)

- v196 failed before container creation on the host QSA consumer source gate; v204 created an owned container but the guard rejected its then-unlisted name before model Python started. Both are preserved as zero model-process launch-boundary failures under the established accounting convention.
- v205 started and loaded a model but the overlay QSA source tuple rejected the placement-aware consumer before readiness. It consumes the second and final global repair-reserve model slot; v188 consumed the first.
- v206 passed all 24 U32 block-A requests and cleanup, but a later source revision added nonuniform-only prefill ownership and observer handling. It remains useful U32 evidence and consumes one A model slot; it is not silently treated as an exact common-source mirror arm.
- v197 started a Frequency model and failed before requests because the then-current prefill binding incorrectly required KTEP on q512 pure-NPU layers. Its cleanup/archive/cache retirement passed and it consumes one A model slot.
- v207 uses the final common A source and passed 24/24 requests, one cancel/reuse control, strict cleanup, archive and 179-file targeted cache retirement. v208 Frequency block A is currently running after fresh two-window admission.
- A accounting after v207: 3/8 model processes, 48/192 generation attempts, 2/4 abort controls, 5/8 component processes. The global repair model reserve is 2/2 used. Five A model slots remain, so the original eight-success mirror cannot be fully restored after v197; all four strategies will receive at least one successful request block, and remaining second blocks will be assigned by the frozen first-block evidence without exceeding the limit. The missing pair remains an explicit A coverage gap.

## B candidate and C disposition

- C is closed as `C_DEFERRED_WITH_REASON`: CPU Expert exposed-wait maxima were 6.7043% and 6.4842%, below the 10% entry threshold. No d=1 code or device run was created and no semantics changed.
- The single B candidate is QSA Decode direct resident-BF16 row consumption. P0 measured QSA as the largest Decode hotspot in both blocks; the current ROW path still stages already-resident BF16 K/V through CPU on every Decode row. The candidate retains the promoted vectorized Prefill, sparse-GQA equations, BF16/FP32 precision, causal selection and fallback behavior, and removes only the qualified Decode staging.
- Test-first source work is complete: the red test captured the absent candidate builder; current Round13 B source/snapshot/preflight tests pass 7 tests with one Torch-only local skip. Six model snapshots dry-run through the exact launcher/runtime source tuple in local preflight. The two real-device component blocks v215/v216 remain NOT_RUN until A releases the device.

## A first-block continuation (v208-v210) and v209 evidence-validator repair

- v208 Frequency block A passed all 24 requests, the formal cancel/reuse control, strict cleanup, archive, and 179-file targeted cache retirement. Against fresh common-source U32/v207, W2 TPOT improved only about 3.0%, below the required 5%; W0 and W3 also showed material regressions, so Frequency cannot be promoted even if another process were available.
- v209 Front loaded successfully and completed its first W0 request. The request selected H513 Hybrid and returned the observed q0 CPU layer path, but the inherited Round11 validator required `NPU_RESIDENT_APPLY` in the same bounded KTEP observer window. Front's first three q512 layers are pure NPU and bypass that KTEP observer; the remaining q0 layers are pure CPU. The immutable result is `FAIL_ROUND13_A_MODEL_DRIVER`, followed by guard ACK, owned-container removal, physical12 empty, hashed fail archive, and cache retirement PASS.
- The validator defect was reproduced before modification. The v6 driver now accepts the NPU half only from a frozen q512 profile whose exact physical receipt is independently checked, while still requiring Hybrid selection, CPU submit/result/apply, no Layerwise event, and all existing output/QSA/PLE/resource checks. Two focused tests and the fresh uniform/v207 regression pass. The repair has new v223 identity; no unmodified retry is allowed.
- Budget after v209: A 5/8 model processes, 73/192 generation attempts, 2/4 formal abort controls, 5/8 component processes; total 9 model processes and 88 generation attempts. The global repair reserve remains exhausted. Completing v210 Random and v223 Front leaves only one A model slot, insufficient to form any complete second paired block, so A can report all four first-block measurements but cannot promote a placement within this frozen budget.
- v210 Random block A is currently executing after a new two-window admission. The v223 immutable snapshot has already been built and source-checked, but will not start until v210 exits and its evidence is archived.

## A pre-model control fixes and current v225 execution

- v210 passed fresh admission but failed before container creation because its root-owned snapshot could not create `host-controller.json` from an admin control shell. The admission evidence was hashed and archived; no model process, generation, cache load, or NPU process occurred.
- A new root-control run identity v224 then reached container creation, but its invented `random-a-r1` suffix was outside the frozen guard allowlist. The guard rejected it before container start. The exact Created-state container ID was matched to `container-created.json`, removed, and physical12 was confirmed empty. Its full registered-file cache audit returned `NOOP_ZERO_RESIDENCY`, which is the correct teardown result because no model or bank page was loaded.
- The final control repair does not broaden the guard. v225/v226 retain new version identities but use the already-approved `front-a` and `random-a` container labels. The teardown runner now accepts either an audited `PASS_CACHE_RETIRED` or audited `NOOP_ZERO_RESIDENCY`, while still rejecting active references or changed file identities.
- v225 Front has passed admission, guard, model load, and the repaired real-request Hybrid evidence check; it is executing its 24-request block. v226 Random is built and source-checked but remains NOT_RUN until v225 archives and retires its cache.

## A disposition and B component entry

- v225 Front and v226 Random each passed 24/24 requests, the block-A cancel/reuse control, exact frozen placement budget, QSA/PLE/path checks, resource guard, cleanup, immutable archive, and 179-file targeted cache retirement.
- Fresh common-source U32/v207 W2 TPOT was 0.42499178065 s. The single fresh block ratios were Random 0.968891, Frequency 0.969882, and Front 0.992245. Every candidate therefore fails the necessary first-block `<=0.95` primary gate. Frequency also regressed W0/W3 materially; Front regressed W0 and W4 TTFT; Random regressed W0/W1.
- `A_RESULT.json` is frozen as `A_NO_PROMOTION_WITH_MIRROR_GAPS`: all four strategies have one same-budget fresh block, but the original eight-process mirror cannot be completed after counted integration failures. Since each candidate already fails a necessary first block, the remaining one A model slot is intentionally unused; no static placement can be promoted.
- A final accounting: 7/8 model processes, 121/192 generation attempts, 4/4 abort controls, and 5/8 component processes. The missing second blocks remain an explicit Round13 coverage gap even though the no-promotion disposition is determined.
- B QSA Decode component v215 is now executing after fresh admission. v216 will run only after v215 cleanup and cache retirement; request models remain gated on both blocks having all nine full-call ratios at or below 0.90.


## B v215 immutable failure and allocator-baseline repair

- v215 completed 9/9 real-weight rows (layers 3/23/47, contexts 64/128/512). Numerical errors were zero against the independent sparse-GQA oracle, repeat hashes were exact, causal/fallback/lifecycle/cancel/stale-completion checks passed, and every complete-call ratio was below 0.19.
- The immutable run failed only because the probe required `memory_allocated == memory_reserved == 0`. torch_npu created a process-lifetime RNG state (16,896 allocated bytes in a 2 MiB segment), so this gate measured provider initialization instead of experiment leakage. Guard cleanup, owned-container removal, physical12 emptiness, archive SHA and 179-file targeted cache retirement all passed.
- The repair was reproduced by a failing unit test before modification. v2 explicitly primes the NPU RNG/provider, records allocated/reserved baseline, and requires exact return to that baseline after the workload. New immutable identities are v231/v232; the v215 FAIL is not rewritten and v216 is not reused.
- v231 is running after a fresh 2x60s admission. Request-model snapshots remain gated on both repaired independent blocks.


## B v231 diagnosis and final component source revision

- v231 again passed all nine numerical/performance rows and lifecycle contracts, but its initialized allocator baseline was zero while the final live allocation was 16,896 bytes. That byte count matches the allocator-rounded 1×2051 int64 `future` tensor used only for the negative causal check. The v215/v231 leak was therefore in the probe fixture itself, not provider RNG state and not the QSA candidate.
- The second source revision deletes this negative fixture immediately after the causal/fallback assertions. Its test failed before the edit and passes afterward. v233/v234 freeze this final source for the two required independent blocks; v232 remains unused so there is no unchanged retry.


## B component promotion gate passed

- Final frozen v3 blocks v233/v234 both passed all 9 real-weight rows, exact numerical/repeat-hash checks, causal/fallback/lifecycle/cancel/stale-completion checks, allocator baseline cleanup, guard cleanup, archive and targeted cache retirement.
- Worst full-call candidate/baseline ratios were 0.197938 (A) and 0.188911 (B); all 18 rows satisfy the required `<= 0.90` gate. `B_COMPONENT_GATE.json` opens the real-request gate.
- Next is a fresh baseline/candidate model smoke (v217/v218), followed only on smoke success by S0/B/B/S0 request blocks v219-v222.


## B request ABBA promoted

- Baseline/candidate smoke v217/v218 passed five canaries, multistep Decode, cancel/reuse, QSA/PLE/path and cleanup. Candidate evidence shows `DIRECT_BF16_ROW` on every qualified Decode layer while vectorized Prefill remains active.
- Request ABBA v219/v220/v221/v222 passed 20/20 attempts per process with exact workload/output-token budgets. W2 TPOT ratios were 0.663446 and 0.683139; W2 E2E ratios were 0.702797 and 0.720786; W2 serial-throughput ratios were 1.437468 and 1.404853.
- No workload had a two-block >5% TTFT/E2E regression or serial-throughput regression. Resource, quality, QSA and PLE checks passed. `B_REQUEST_RESULT.json` is `B_PROMOTED`; F integration is now mandatory.


## F final mirror and Round13 terminal disposition

- v227/v228/v229/v230 completed the fresh S0/FINAL/FINAL/S0 mirror. All four processes passed 24/24 generation attempts, QSA/PLE/path/quality gates, the two planned cancel/reuse controls, guard cleanup, archive SHA and targeted cache retirement.
- F W2 TPOT FINAL/S0 ratios are 0.677706 and 0.679009; E2E ratios are 0.701938 and 0.707040; serial-throughput ratios are 1.416469 and 1.417520. No workload has a two-block protection regression. `F_RESULT.json` is `F_FINAL_PROMOTED`.
- The final limited configuration retains Uniform Q32 and promotes only QSA Decode `DIRECT_BF16_ROW`. Dynamic/R2/R3/transpose/shadow320/deferred remain OFF; H129 remains the default and W3 H513 remains experimental.
- Final budget: 21 model processes, 328 generation attempts, 9 abort controls and 9 component processes; all are within the frozen hard limits, with zero unchanged automatic retries.
- Terminal remote audit: no Round13 container or owned controller remains and physical12 reports `No process in device.` v230 retired all resident pages for 179 registered read-only files after archive/SHA and found no active fd/mmap references.
- Overall status is `ROUND13_PARTIAL_WITH_GAPS`: a real B/F request gain is qualified, but A did not complete its required eight-process mirror. No Round14 work is started automatically.
