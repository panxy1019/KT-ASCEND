# Round10 T6 final fixed-configuration regression

Status: **PASS_ROUND10_T6_FINAL_REGRESSION_WITH_CONTROLLER_RECONCILIATION**.

The single v107 model process completed all eight frozen generations and one abort with zero retry. Quote and transform canaries, 64-token ACK/EOS, 32-step Decode, mid-prefill cancellation, lease return and same-process `391` reuse passed. Every available output-token sequence is exact against the matching first eight v95 rows.

Unprofiled 64-token measurements were 5.409330s and 5.832713s, median 5.621022s. The four v95 H0 samples had median 5.188200s, so this repeat is 1.083424× H0. HFINAL equals H0 in behavior; no performance candidate is promoted and the two-sample repeat is reported without a speedup or regression-promotion claim.

The model supervisor is PASS: driver exit 0, eight requests recorded, guard ACK received and the owned container removed. The outer controller then looked for the old T1 supervisor filename and marked itself FAIL after cleanup. The raw FAIL is retained; the generator regression is fixed locally and no model rerun was used.

Peak sampled board/owned HBM was 28778/25762 MiB and peak host cgroup usage was 338.977 GiB. Targeted cache retirement covered 179 unreferenced model/bank files, left file identities unchanged, and did not use global drop_caches.
