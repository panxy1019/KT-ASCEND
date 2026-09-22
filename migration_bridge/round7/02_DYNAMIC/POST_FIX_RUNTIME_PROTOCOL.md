# Post-fix runtime handoff and predecessor orchestration

2026-09-17 UTC. Tools LOCAL_VERIFIED, not full-model Dynamic qualification.

New native SHA:
`46182176279b072b4ab52d21bb8cc0ecf4d217399d486155c7fcd4f0690e21e2`.
Original native/source were checked unchanged after build. It remains isolated
under `backing-fix-build-v1/runtime`; no installed package was overwritten.

The original launcher explicitly prioritized the old CPU runtime. Therefore
merely building a new .so would not exercise the repair in serving. The new
adapter requires explicit `cpu_runtime_root` matched to the new provenance's
root and frozen native path/hash. Existing provenance/file preflight checks
remain. CPU import precedence is published through PYTHONPATH before the model
interpreter execs; original configs retain the old path. Seven pure-Python
regressions cover missing selection, out-of-scope roots, provenance mismatch,
missing native and mismatched hashes. The full tool suite is78/78 local PASS.

Preparation requires all three A3 eviction regressions PASS first, snapshots
only mask/config inputs into a new runtime directory, and records all private
Python/native hashes plus the original provenance hash. It **does not copy old
PASS summaries** into the new qualification lineage. It retains original runtime
metadata explicitly as reference history, not as the repaired binary's build.

`run_backing_requalification.py` is a serial owned driver with an exclusive lock:
representative operators under the repaired environment → Uniform compact →
Frequency compact → fresh R64 CMMLU256 → original DEV migration probes256/512/1024.
Per-stage logs/results remain, software failures stop successors for autonomous
agent diagnosis. The R64 lane changes only resident ratio to1.0; same frozen
model/tokenizer/CMMLU/evaluator. It requires historical174/256 reproduction with
invalid0 before proceeding, and never silently replaces the reference.

The driver ends before threshold fitting/selection, candidate commit or HD/CMMLU
Dynamic use. All future selected-threshold consumers still require the explicit
local candidate commit acknowledgment. Diagnostics remain absent from timed
performance. Existing static selection and historical numerical failures are not
rewritten by this repair or by a later full-model result.
