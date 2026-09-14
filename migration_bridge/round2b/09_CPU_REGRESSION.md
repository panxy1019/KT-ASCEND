# Round 2A CPU Regression

## Fresh Ascend-OFF build

The current Round 2B source was rebuilt in new directories with all device backends and KML disabled. Observed configuration:

```text
KTRANSFORMERS_USE_ASCEND:BOOL=OFF
ASCEND_HOME_PATH:PATH=
No GPU support enabled, building for CPU only
```

`readelf -d` found no ACL/Ascend `NEEDED` entry. Import selected aarch64 variant `arm`; device-stream callback methods are compiled out for this build.

## Round 2A core matrix

| Module | Coverage |
|---|---|
| `test_cpu_detect_metadata.py` | ARM actual metadata and mocked architecture fallbacks |
| `test_llamafile_gguf_e2e.py` | deterministic GGUF and production wrapper E2E |
| `test_llamafile_routed_correctness.py` | 4/8 experts, top-k, mapping, decode/prefill |
| `test_llamafile_cpuinfer_lifecycle.py` | 1,000 forwards, 20 lifecycle paths, layers, pools/NUMA |

Result: **21 passed in 5.67s**. The 1,000-forward RSS result was `353,378,304 -> 353,378,304`, delta 0. Worst numerical values remain within the Round 2A envelope: max abs `6.103515625e-05`, mean abs `1.4901161193847656e-08`, relative L2 `3.9346272514744805e-05`.

Evidence: `cpu-only-build.log`, `cpu-build-py.log`, `cpu-regression.log`.
