# Frozen profile matrix

| Profile | NPU/layer | CPU/layer | NPU resident ratio | All MoE layers |
|---|---:|---:|---:|---|
| R64 |64|0|100%|1–26|
| R32 |32|32|50%|1–26|
| R16 |16|48|25%|1–26|
| R8 |8|56|12.5%|1–26|

R32 NPU IDs are multiples of2, R16 multiples of4, R8 multiples of8, R64
allIDs. CPU IDs are the exact complement. Logical masks are those frozen in
Round6.5, not selected using new quality output. `freeze_serial_study.py`
copies exact JSON/PT bytes, verifies canonical and PT hashes, then freezes the
new CMMLU-only product contract and runner hashes before collection.

Existing implementation invariants (identity/ownership/weight/scaling/shared/
transfer lifetime) are retained. This round adds physical-layout and route
coverage checks on all26layers; it does not assert every mathematical invariant
is exhaustively re-proven by a32request compact screen.
