# Real Expert Identity Matrix

P1 selected `L17E{6,8,25,36}`. Every expert completed a real one-token CPU/NPU
identity comparison; E6/E25/E36 used five CPU repeats and the E8 anchor used ten.

| Expert | Unique CPU hashes | CPU vs BF16-rounded FP32 rel L2 | CPU vs NPU rel L2 |
|---:|---:|---:|---:|
| 6 | 1 | 0.0 | 0.0038499988353798906 |
| 8 | 1 | 5.29088030502319e-05 | 0.004299063928345015 |
| 25 | 1 | 1.5360325773518376e-05 | 0.004289701401451686 |
| 36 | 1 | 1.3520655896058197e-06 | 0.004597487464640683 |

All outputs were finite and every repeat maximum absolute difference was zero.
All four experts pass both hard gates (`5e-4` and `1e-2`). P2/P3 identities were
not run after the ordered P1 stop.
