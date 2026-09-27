# Round12a current execution

Updated: 2026-09-27 UTC

Status: `ROUND12A_COMPLETE_KT_STYLE_VALUE_CONFIRMED`.

The tied-parameter-aware OffloaderV1 fix is qualified by v172. The first same-revision non-KT request passed in v173. The fair promoted-QSA KT/NON_KT/NON_KT/KT sequence v174-v177 passed exact quality, resource, cleanup and performance gates. KT median E2E was 2.982178643 s versus OffloaderV1 1204.699036151 s in block A and 3.386335781 s versus 1349.630512928 s in block B, a 99.752454% and 99.749092% reduction respectively.

The user-added Native SGLang TP8/old-QSA baseline also completed. v180-v183 are identical-code independent A/B/B/A repeats on physical0-7. Their process E2E medians were 3.428952584, 3.371347328, 3.489020653 and 3.325310193 seconds; max/min=1.049231636, all twelve measured outputs were exactly `[3897, 248046]`, cross-run per-device HBM spread was at most 2 MiB, and Host peak ratio was 1.000052514. This is repeatability evidence, not a fair speedup comparison against the promoted-QSA KT arm.

Every model block was fully archived and SHA-verified before targeted POSIX_FADV_DONTNEED. Nine retirement runs ended at zero registered resident pages without global drop_caches. Physical0-7 and physical12 are empty, no Round12 container/process/timer remains, and the registered training container identity/cgroup is unchanged. Automation remains paused.

Fresh final verification reran 148 tests and compiled 34 Python files. Both comparators exactly reproduce their frozen results under their canonical serialization, all archived SHA receipts pass, and a 2026-09-27 06:22:31 UTC read-only remote audit reconfirms physical0-7/12 cleanup and the unchanged training identity. The first audit parser false-positive for three exited legacy Round9 containers is preserved separately; no legacy or external container was modified.

Primary evidence: `ROUND12A_FINAL_STATUS.json`, `ROUND12A_KT_NONKT_ABBA_RESULT.json`, `ROUND12A_CROSS_BACKEND_SYNTHESIS.json`, `native_tp8/NATIVE_TP8_ABBA_RESULT.json`, and `ROUND12A_FINAL_REPORT_ZH.md`.
