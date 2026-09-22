# One-container topology

User explicitly rejects the multi-container test-farm proposal for this round.
The existing `kt-r65-static-offload` on110.129.0.22 is reused. Only one model
server process runs at a time, loopback port31000, NPU0, CPU0–31,32workers,
threadpool1, NUMA0. All model files remain mounted read-only; existing GGUF is
shared without duplication. Container memory limit192GiB unchanged.

Startup inventory: host~1TiB RAM, swap0, no NPU test processes, NPU0 initial
HBM~3414MiB/65536MiB. Fresh machine/cgroup/software inventory is preserved in
the study preflight rather than inferred from historical values.

NPU0 is PCI C1:00.0, NUMA6. CPU0–31 are NUMA0. Locality mismatch is a recorded
limitation; neither worker count nor topology is modified during ratio sweep.
Only owned model descendants are terminated/reaped between sessions. Business
container `verl` and other devices are untouched.
