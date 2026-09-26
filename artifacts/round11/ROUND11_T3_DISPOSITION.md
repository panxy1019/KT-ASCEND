# Round11 T3 Kunpeng candidate disposition

Status: **KUNPENG_NO_PROMOTION_NO_BOUNDED_CANDIDATE**.

T2 measured the frozen three-token layer-0 replay at 375.5825 µs median. The attribution run assigns 244.70000000000005 µs (66.70%) to the parallel native compute/merge critical path, 43.262 µs (11.79%) to enqueue/task construction, and only 4.4 µs (1.20%) to queue admission. The 128 named compute threads have max/min runtime 1.010896 and CV 0.002098.

The frozen source contract allows the guided-block candidate only when scheduling or imbalance is material. Those gates are false. Input quantization is 2.0–2.6 µs per row and is likewise not material enough to support the 10% component gate. The dominant remaining root is the MXFP4 NEON+SDOT kernel itself; the audited SVE/BF16/I8MM switches do not select another MXFP4 path, so a credible change requires a new kernel and its own numerical/backend qualification. That is a new backend project rather than this round's bounded single-variable candidate.

No T3 component or model process was spent. The original C128/MXFP4 backend remains selected and Round11 proceeds to T4. No performance gain is claimed. Hardware cycles/IPC/cache/bandwidth/remote-NUMA remain UNKNOWN because `perf_event_open` was denied and no validated uncore mapping exists.
