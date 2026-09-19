# Full-model resource admission amendment (before Q64 outputs)

Qwen routed F32 payload alone is120GiB. The single-expert container's192GiB
limit does not safely cover backing plus possible per-subpool runtime copies,
checkpoint staging, model Python and allocator overhead. Set the OWNED Round8
container limit to384GiB, memory-swap also384GiB (no additional container swap).
CPU128–191/NUMA4,5/two32 subpools and NPU2 do not change. Host available RAM
must remain above max64GiB/20% after conservative384GiB budget. No other
container, host swap or system libraries may be changed.

Host has approximately48MiB historical swap. Do not disable host swap or evict
unrelated pages. Amend the PREP resource guard (not numerical/quality gate):
residual swap is allowed ONLY with >=30s observed zero host swap-in/out counter
deltas and owned-cgroup swap.current=0. Missing measurements or active paging
reject admission. Preserve actual nonzero total swap in reports; never claim
zero host swap. This distinguishes stored historical pages from memory pressure.

Full-model launch still requires fresh NPU-process inventory, physical CPU
topology, host reserve and disk checks. Conversion identity PASS does not waive
resource admission. Resource errors stop only the new Round8 owned process.
