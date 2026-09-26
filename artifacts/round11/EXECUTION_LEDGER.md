# Round11 append-only execution ledger

## 2026-09-26 — T0 baseline recovery

- Created `codex/round11-hybrid-performance-backend` at unchanged HEAD `b41e901c059222b8c1e4367abe193d7f58cc1d83`; preserved dirty Round9/Round10 evidence.
- Verified Round10 status `ROUND10_AB_NARROW_QUALIFIED` and the v107 2,568-file local manifest.
- Read-only remote inspection found no owned task process/container and no physical12 process; training full ID remains present; matching timers remain paused.
- Frozen independent Round11 limits: 21 model starts, 108 generations, 5 aborts, and stage-specific component caps. T0 used no device budget.

## 2026-09-26 — T1 v108 boundary in progress

- Fresh admission passed 12/12 windows for physical12/card6/chip0 and NUMA
  2/3/4/7; the registered training container identity and cgroup matched.
- v108 BOUNDARY launched exactly once with zero automatic retry. Its frozen
  three-request queue is fully charged to the Round11 ledger while the final
  request remains in flight.
- edge128 selected Hybrid and passed ACK/EOS/finite-logprob validation in
  26.103200164 seconds. edge129 selected Layerwise, covered chunks
  32/32/32/32/1, and passed the same quality checks in 1175.462570734 seconds.
- edge130 is still running. No T1 promotion or expanded-boundary conclusion is
  recorded before supervisor cleanup and the forced-Hybrid hierarchy.

## 2026-09-26 — T1 v108 boundary complete

- edge130 selected Layerwise with chunks 32/32/32/32/2 and passed the frozen
  ACK/EOS/finite-logprob contract in 1173.084273998 seconds. v108 therefore
  completed all three charged generations with status `PASS_ROUND11_T1_PROCESS`.
- Supervisor exit, guard ACK, container removal, `OOMKilled=false`, and a fresh
  read-only physical12 check all passed. The result SHA-256 is
  `83e4ecedc16b0414cef59d5d3dd9fcbb646bb0eca73449dc440d04beb184da04`.
- Raw run and admission archives were copied without modifying their contents,
  matched the remote SHA-256 values, passed path-traversal validation, and were
  extracted under `t1-results/v108`. The original failed archive copy and its
  failure receipt remain preserved.
- Budget remains one model start, three generations, zero aborts, and zero
  automatic retries. No expanded-Hybrid conclusion is recorded until v109,
  v110, v111, and the strict four-process aggregate complete.

## 2026-09-26 — T1 v109 admission recovery and launch

- Admission attempt 1 sampled all 12 windows but failed closed before launch:
  NUMA4 minimum free memory was 22.834 GiB, below the unchanged 48 GiB gate.
  CPU idle and physical12 identity were healthy. No model budget was consumed.
- Read-only inspection attributed the pressure to file pages left after the
  owned v108 process. The frozen retirement tool audited every process for
  target-file fd/mmap references, found none, then applied only
  `POSIX_FADV_DONTNEED` to the 179 registered project `.gguf`/`.safetensors`
  files. File stat identities remained exact; no global cache clear or external
  process action occurred. NUMA4 free memory rose to 162.79 GiB.
- A fresh admission identity passed 12/12 windows with minimum mean idle
  0.9967705344 and NUMA2/3/4/7 minima of
  230.41/229.92/162.80/224.37 GiB. Training full ID/cgroup remained exact.
- v109 HYBRID256 launched exactly once as owned container
  `5983dc84e4e5da53bfbe42750520f26c78e93093c4e1507795c88d84fa6afe66`.
  The frozen queue charges one model start, 15 generations, one controlled
  abort, and zero retries. Global Round11 use is now two model starts,
  18 generations, one abort, and zero retries.

## 2026-09-26 — T1 v109 raw failure and bounded repair decision

- v109 completed 15 physical generations and one abort. Its first 13 rows all
  passed, including Hybrid lengths 129/130/160/192/256, exact 256 repeatability,
  and 32/64-step Decode. The reuse row also passed exact known-answer checks.
- The raw cancellation trace emitted the established runtime sequence
  `CANCEL_SEEN -> SUBMIT_FENCE_PUBLISHED -> REQUEST_RETIRED ->
  CANCEL_CLEANUP_COMMITTED`. The Round11-only validator incorrectly required
  an event named `CANCEL_FENCE_ADVANCED`, which neither the frozen runtime nor
  the prior Round10 receipt emits. The driver therefore exited 1 and v109 is
  permanently retained as FAIL; no result was relabeled.
- Supervisor cleanup passed with guard ACK, `OOMKilled=false`, container
  removal, and empty physical12. The immutable result SHA-256 is
  `06bee31bef74dd21378bf2660ca9f4783a81df0237cbdebc710d4793d2c068a1`.
- A failing regression test reproduced the event-name drift. The repaired
  validator now requires the real event, strict event order, a shared epoch,
  no chunk after start32, no late commit, lease return, and passing reuse.
  Twelve protocol/composition/aggregate tests pass.
- The bounded repair is a new v112 model identity with only two generations:
  cancel256 and same-process reuse. A strict composer will retain v109 FAIL and
  record 17 physical generations across two model processes while using the
  13 immutable v109 PASS rows plus two v112 rows for logical T1 coverage.

## 2026-09-26 — T1 v112 bounded cancel repair running

- The v112 repair snapshot contains the corrected lifecycle validator and only
  the frozen cancel256/reuse schedule. Its exact transfer had 2,574 files.
- Targeted project-file cache retirement again found zero active references,
  preserved all 179 file identities, and restored all four NUMA nodes without
  global cache clearing. Fresh v112 admission then passed 12/12 windows with
  minimum mean idle 0.9968704218 and NUMA minima
  230.42/229.98/162.89/224.47 GiB.
- v112 launched exactly once with two generations, one abort, and zero retry.
  Global/T1 use is now three model starts, 20 generations, two aborts, and zero
  automatic retries. No composite PASS exists until raw v112 cleanup and the
  strict source-hash composer complete.

## 2026-09-26 — T1 v112 repair and composition complete

- v112 passed both physical generations. cancel256 triggered at start32 and
  emitted the ordered same-epoch lifecycle required by the corrected contract;
  no later selection or late commit occurred, the lease returned, and reuse
  produced exact IDs `[18,24,16,248046]`.
- Supervisor PASS, guard ACK, `OOMKilled=false`, container removal, and a fresh
  empty-physical12 check passed. Result SHA-256 is
  `74350392a3f673d9e2dac01cf1482b104809b1469e7684c6e87439aa19a70524`.
- The source-hash composer passed and produced
  `T1_HYBRID256_COMPOSITE.json` with SHA-256
  `1f48cb9ea4f888cf77b0368e8348510db5643430048be6d05e2a9e0efa51e0fd`.
  It explicitly records 17 physical generations and two processes; v109 stays
  FAIL and only its 13 independently validated rows are reused.
- T1 use remains three model starts, 20 generations, two aborts, zero retries.
  The hierarchical boundary remains incomplete until v110 and v111 pass.

## 2026-09-26 — T1 v110 HYBRID512 running

- v110 exact transfer and the post-v112 zero-reference targeted cache
  retirement passed. Fresh admission passed 12/12 windows with minimum mean
  idle 0.9968223796 and NUMA minima 230.41/229.98/162.94/224.33 GiB.
- v110 launched exactly once with four generations, zero abort, and zero
  retries. Its frozen queue is edge384, edge512, Decode128, Decode256.
- Global/T1 use is four model starts, 24 generations, two aborts, zero retries.

## 2026-09-26 — T1 v110 complete; v111 HYBRID1024 running

- v110 passed Hybrid384, Hybrid512, Decode128, and Decode256. E2E values were
  86.899203524, 125.920623928, 83.174828253, and 319.373423398 seconds.
  Supervisor PASS, guard ACK, container removal, archive hashes, and an empty
  physical12 process view passed. Result SHA-256 is
  `c73ad2af1173d5393e9c40184899a560b2e5a7e13d0b91642a149f8e809ff14d`.
- Before v111, the zero-reference targeted cache retirement preserved all 179
  model/bank file identities and restored NUMA2/3/4/7 to
  230.52/229.98/162.93/224.56 GiB. No global cache clear was used.
- Fresh v111 admission passed 12/12 windows with minimum mean idle
  0.9968485650, exact external-training identity, and no unknown physical12
  process. v111 launched exactly once with two frozen generations
  (Hybrid768 and Hybrid1024), zero abort, and zero retry.
- Global/T1 use is five model starts, 26 generations, two aborts, zero retries.

## 2026-09-26 — T1 boundary frozen at 768; T2 preparation

- v111 Hybrid768 passed with exact ACK/EOS, finite logprobs, 24 chunk32
  dispatches, and 282.039209205 seconds E2E. During Hybrid1024 the live guard
  stopped only the owned container when NUMA4 free plus conservative reclaim
  headroom fell below the unchanged runtime reserve.
- v111 remains an immutable raw FAIL. Exit 137 was guard initiated,
  `OOMKilled=false`; guard ACK, container removal, and empty physical12 passed.
  No automatic retry was made. The strict cutoff composite hashes the raw
  result, supervisor, monitor, and post-cleanup process view.
- T1 closes as `PASS_ROUND11_T1_WITH_RESOURCE_CUTOFF`: prompt up to 768 and
  Decode up to 256 are qualified for serial Hybrid; 1024 is unqualified. The
  default threshold remains 129, and no expanded-Hybrid crossover was found
  inside the qualified range.
- T1 final physical use is five model starts, 26 generation attempts, two
  abort controls, and zero automatic retries.
- A broad-root cache-retirement command failed closed before mutation because
  it counted symlink aliases. Its new replacement identity audited the exact
  frozen 179 files, found no live references, preserved stat identity, and
  used only `POSIX_FADV_DONTNEED`. NUMA4 remains below the 48 GiB startup gate
  due to other file pages, so no T2 component process has launched.

## 2026-09-26 — T2 attribution build ready; v113 attempt 1 failed closed

- Attribution build v1 failed before any component/device launch because the copied CMake cache expected `/build` while the container mounted another path. The raw failure receipt and log remain immutable.
- A failing regression test captured that identity error. Build v2 uses `/build`, passed, and produced attribution-only native SHA-256 `dff773ecabb3c7419afd8e3586e3c09934ebd9f23c7d9604eca5fa1218b39183`. Its only semantic change is environment-gated TaskQueue wait/execute timestamps; it is not an optimization candidate.
- A root-level inode/device audit found zero fd/map references to the exact 179 frozen project files. Targeted `POSIX_FADV_DONTNEED` left all 179 with zero resident bytes under root `fincore`; no global cache action or external process change occurred.
- Fresh v113 admission attempt 1 completed all 12 windows but failed before container creation because NUMA4 had only 18.345 GiB free, below the unchanged 48 GiB gate. CPU idle and identities passed. No component budget was consumed.
- The remaining NUMA4 file pages are outside the project-file set and are treated as external resource use. T2 will use a new admission identity when the node recovers; meanwhile its strict three-run analyzer and 36 passing Round11 tests are ready.

## 2026-09-26 — T2 v113 admission attempt 2 failed closed

- A new two-round/12-window identity again passed CPU idle, system memory, external training identity, and physical12 process checks. Minimum conservative idle was 0.9966739455.
- NUMA4 remained stable at a minimum 18.346 GiB free, below the frozen 48 GiB gate; NUMA2/3/7 remained 230.595/229.825/124.410 GiB. Validation failed before container creation.
- Raw discovery/validation/log hashes and failure receipt are preserved locally and remotely. Physical12 stayed empty. Component budget remains 0/6 and automatic retries remain zero.

## 2026-09-26 — T2 v113 baseline complete

- Fresh admission attempt 3 passed all 12 windows: conservative SMT-min idle
  was at least 0.9967775821 and NUMA2/3/4/7 minima were
  230.607/229.948/160.709/224.478 GiB.
- v113 launched once and completed 3,000 real layer0 C128/MXFP4 replay
  iterations. The parent median was 375.5825 us for the three-token fixture;
  repeat output hash and the frozen output identity passed.
- Controller exit 0, no OOM, guard exit ACK, container removal, and empty
  physical12 all passed. Result SHA-256 is
  `62643302e183e1b1585574dbb7daa86db0a908da876372273bdf22307d0c35c1`.
- Component use is 1/6. The two prior admission failures consumed no component
  or retry budget. v114 is gated on post-hash targeted cache retirement and a
  new two-round admission identity.

## 2026-09-26 — T2 complete; T3 no bounded candidate

- v114 counters and v115 attribution each passed a new 12-window admission,
  launched exactly once, completed 3,000 real replay iterations, and returned
  exit 0 with guard ACK, container removal, no OOM, and empty physical12.
- The three-run strict composer passed. Baseline/counters/attribution parent
  medians were 375.5825/371.548/366.8575 us. Native parallel compute+merge is
  244.7 us (66.70%); queue admission is 4.4 us (1.20%). The 128 worker runtime
  max/min ratio is 1.010896. `perf_event_open` returned EPERM, so hardware
  counters remain explicitly UNKNOWN.
- The exact post-hash cache retirement ran after v113, v114, and v115. Each
  audited 179 registered files, observed 326,401 resident pages before, verified
  zero pages afterward, preserved alias/stat identity, and did not call global
  drop_caches.
- The scheduling and shared-quantization candidate gates are false; the
  remaining dominant mechanism requires a new MXFP4 kernel. T3 closes as
  `KUNPENG_NO_PROMOTION_NO_BOUNDED_CANDIDATE` with zero T3 processes. The
  original C128/MXFP4 backend remains selected and T4 follows.

## 2026-09-26 — T4 v120 preserved failure and v124–v127 repair freeze

- v120 passed fresh 12-window admission and launched one Static-A model process, but the driver failed before its first generation because the generated supervisor passed the Round11 boundary fixture, which has no `edge64` case. The raw FAIL, driver log, supervisor, host-controller receipt, cleanup, and empty physical12 evidence remain immutable. This consumed one model start and zero generations.
- v121/v122/v123 were never transferred or launched and are explicitly rejected because they share the frozen fixture-wiring defect. They are not reused or rewritten.
- Regression tests bind T4 to the HFINAL Round10 exact 64-token fixture and require an explicit failure if a child exits before publishing a raw result. All 60 Round11 tests and Python compilation pass.
- New immutable identities v124 Static-A, v125 Dynamic-A, v126 Dynamic-B, and v127 Static-B were built from HFINAL v107. `T4_REPAIR_SNAPSHOT_FREEZE.json` SHA-256 is `0bc1c95cd7241312f564652db250d994038f79d6d2318cbf62f519f2d417d0dc`.
- Post-v120 cache retirement audited the exact 179 registered read-only files, found no live references, reduced resident pages from 68,711,731 to zero with only `POSIX_FADV_DONTNEED`, and preserved stat identity. The retirement tar SHA-256 is `81f05d66a4eac1ea83eb0b99bede149092bd09e46717a6e9bd615fb32c533a10`.
- The repair reserve is explicit: T4 may consume five physical model starts total (one failed plus four ABBA processes), while the Round11 global limit remains 21. Global use is now six model starts, 26 generations, two aborts, and zero automatic retries.

## 2026-09-26 — T4 Static-A PASS and two fail-closed Dynamic wiring repairs

- v124 Static-A passed five generations and complete cleanup. Decode32/64/128/256 E2E was 15.853957/27.680919/54.292791/116.525245 seconds. Its strict single-run validation SHA-256 is `462cd393669920785b102dc56e89eb9b3bcaafa137a50ff87a0d1a17d72cd204`.
- v125 was rejected before container creation because timing-instrumented `dynamic.py` did not have an explicit derivation from the earlier qualified source hash. It consumed zero model starts and generations. The immutable failure receipt SHA-256 is `d69756bbd5502666f26594f9079221a130a33b5c186f87c586551a9967b2f16d`.
- The derived gate keeps `dynamic_adapter.py` and `expert_source.py` byte-exact, binds both base and instrumented `dynamic.py`, and recomputes the exact 111-line unified diff SHA-256 `aa1a7df1509bf3c5780e9ccf17fed44c285bd9cd5c5843c69d0ba8b1694427bf`. Round11 tests 62/62 and Dynamic lifecycle tests 35/35 pass.
- v128 passed that source gate and launched, then its first request failed because HFINAL's candidate profile still said `static`; the scheduler required `request_boundary`. The raw run remains FAIL after one model start and one generation. Cleanup and post-hash retirement passed; failure receipt SHA-256 is `0e2028e80426233a09c347a92c35963ce2d9cce8733e1dbbf3343faccf260cbf`.
- v132/v133 change only the candidate profile binding and its receipt profile hash. All 48 bank entries, catalog, source identity, and canonical bank payload SHA-256 `1bc5b1e00bc0722fefa8aab05799ac8d2b48b572b8dd57947e5972fde4d6e1e2` remain exact. v129 is unrun and rejected with the shared static-profile defect.
- T4 physical use is three model starts and six generations; one successful Static-A block is available. Global Round11 use is eight model starts, 32 generations, two abort controls, and zero automatic retries.

## 2026-09-26 — T4 v132 transaction evidence complete, observer raced drain append

- v132 passed both derived source and profile gates. Its first Dynamic request executed a real 48-layer migration and completed the requested 32-token stream, but the driver read `dynamic-events.jsonl` before the asynchronous drain line was appended and failed closed.
- The immutable post-run JSONL contains exactly one commit, one timing receipt, and one request-drained receipt for the same request. The raw FAIL is preserved; cleanup passed. Failure receipt SHA-256 is `229f9e871c5c7af57535b2eab9afc3b4552f7ab9941bd5d5e1a5a8dc50f79298`.
- The bounded driver repair waits at most 30 seconds at 50 ms intervals for those three exact-once statuses. Duplicate evidence fails immediately; incomplete evidence times out. Round11 tests 63/63 and Dynamic lifecycle tests 35/35 pass.
- New v134/v135 Dynamic A/B snapshots are frozen; v133 is unrun and rejected because it shares the early-read defect. Snapshot freeze SHA-256 is `0b1d12bc7e76a908a4687bb4151c5af7831776e5bf95f1b78fda972798684a08`.
- Post-v132 targeted cache retirement again reached zero pages. T4 physical use is four model starts and seven generations; global use is nine starts and 33 generations.

## 2026-09-26 T4 Dynamic A/B completion checkpoint

- v134 Dynamic-A and v135 Dynamic-B each passed one warmup plus Decode 32/64/128/256, exact-once commit/timing/drain evidence, guard ACK, container removal, and physical12-empty cleanup.
- Dynamic A/B output token IDs are exact at every measured length. v135 raw result archive SHA-256 is `b63600d172ee18bd7de7be996079953417c3a3f3073ec6850a9f8c9fca33140c`; single-run validation SHA-256 is `b58803d9b49131dbbf7559240a7c9de9f683b29c02287ec8530ca84cdd406b61`.
- Exact cache retirement ran after all evidence hashing: v134 retired 69,208,537 resident pages and v135 retired 70,850,560 resident pages; both ended at zero across the registered 179-file set.
- T4 now has three successful blocks (Static-A, Dynamic-A, Dynamic-B). The sole remaining block is frozen Static-B v127, requiring fresh admission.

## 2026-09-26 T4 final disposition

- v127 Static-B passed the same Decode 32/64/128/256 schedule, exact Static-A/B output token IDs, guard ACK, container removal, and physical12-empty cleanup.
- Four-block comparator SHA-256 `bb455c715818e0c485c1cc4bfd2effa17b4c2b0ce645eb6612f8ab6cee88380d`: Static steady median `0.3964049145 s/token`; Dynamic steady median `0.4052964265 s/token`; exposed migration median `13.824945036 s`; disposition `NO_FINITE_BREAK_EVEN`.
- Prefill-End Dynamic remains transaction-qualified and default OFF. v127 exact cache retirement retired 69,138,928 pages to zero.

## 2026-09-26 T5 hotspot ranking and QSA disposition

- Two independent bounded-observer model processes, v139 and v140, passed four generations each with complete cleanup. Parent/child accounting ranked QSA at 53.871% median attributable request share, GDN at 19.065%, PLE consumer at 0.278%, and PLE exposed wait at 0.0177%. QSA was the sole permitted B-route candidate.
- The query-aware vectorized sparse-GQA candidate used three real-device component processes. v142 preserved an allocator-observer false failure; v143 passed after that observer repair; v144 passed the final numerical and component-performance contract. Maximum relative L2 was `5.645496551606974e-05`. Candidate/baseline medians were about 0.0565–0.0569 for 32/64/128-token prefill and 1.0051 for Decode.
- v145 failed before container creation because the launch gate still required the original QSA source hash. v146 repaired that launcher boundary and loaded the model, but the independent runtime `context_reference_binding.py` gate again required the original hash; the server exited before readiness and before any request. Both failures are immutable and fully cleaned. v146 is an integration-gate failure, not a numerical candidate failure.
- T5 consumed 3/3 model starts, 8/12 generations, 0 aborts, and 3/9 component processes. The QSA candidate closes as `COMPONENT_ONLY_NO_PROMOTION`; original HFINAL QSA remains selected.

## 2026-09-26 T6 four-process final integration

- Frozen v147/v148/v149/v150 ran in BASELINE-A / FINAL-A / FINAL-B / BASELINE-B order. All four labels execute the same HFINAL compute identity because no Kunpeng, Dynamic, or B-route candidate was promoted.
- Every block passed one warmup and the same input128 Decode32/64/128 performance requests. Output token IDs were exact across all four processes, all logprobs were finite, every path was Hybrid, and every process returned guard ACK, container removal, and empty physical12.
- v147 and v149 each passed a mid-prefill cancel plus same-process known-answer reuse. v148 supplied marker/JSON canaries; v150 supplied quote/transform canaries. Together with arithmetic, the five deterministic classes passed.
- Baseline versus Final medians were 27.753959 vs 27.584503 seconds at Decode32 (ratio 0.993894), 40.775423 vs 38.996865 at Decode64 (0.956382), and 71.157255 vs 68.513867 at Decode128 (0.962851). These are independent-process repeat dispersion under identical HFINAL code, not a candidate gain.
- Final T6 use is 4/4 model starts, 24/24 generations, 2/2 abort controls, and zero retry. Evidence archive SHA-256 values are v147 `0834b7139b8480fbf083f16bdfb9491964371aa93e6efe2945e5cc33dc24131c`, v148 `26d65cf039b6082c5975be30905bf71470aea8480a18577e5df618f1457604c5`, v149 `ec80137516febc416db5fbebaf1daf5faa56fa9216aa53b2626e1f4da18cf130`, and v150 `257e7bbc4348fb0fbd80c5df573d6c2dc223ffeace883ae71aac2254f94f2754`.
- After each archive hash was fixed, the exact 179-file cache-retirement hook audited fd/mmap references, used only `POSIX_FADV_DONTNEED`, preserved stat identity, and verified zero resident pages. It never called global drop_caches or changed an external task.

## 2026-09-26 Round11 final freeze

- Terminal status is `ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED` under the serial TP1 scope. The final runtime is original HFINAL: threshold129, C128/MXFP4, Static Q32 BF16/NZ29, chunk32, PLE ASYNC, Dynamic OFF, original QSA, and original Layerwise shadow32.
- Hybrid is experimentally qualified through prompt768 and Decode256. Prompt1024 hit the resource guard; prompt769–1023 remain untested and unqualified. The default threshold remains 129.
- Kunpeng closes with no bounded candidate, Dynamic with `NO_FINITE_BREAK_EVEN`, and QSA with component-only no promotion. No performance candidate is enabled.
- Final Round11 physical budget is 19/21 model starts, 80/108 generations, 4/5 abort controls, and zero automatic retries. Current physical12 is empty; no Round11 container or process remains. Automations remain paused; no push, merge, tag, reset, or clean was performed.
