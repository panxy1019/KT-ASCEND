# Round 2C Regression Results

All final regressions used the post-`b8ba787` Python sources in the disposable A3 container.

## Round 2A CPU expert plane

Fresh Python staging with NPU backend autoload disabled, using the frozen CPU extension:

```text
test_cpu_detect_metadata.py
test_llamafile_gguf_e2e.py
test_llamafile_routed_correctness.py
test_llamafile_cpuinfer_lifecycle.py
21 passed in 5.69s
```

This covers ARM variant metadata, GGUF E2E, multi-expert top-k, physical/logical mapping, CPUInfer lifecycle, and the 1,000 CPU-forward stress.

## Round 2B Ascend runtime plane

```text
test_ascend_vendor_adapter.py
test_ascend_stream_handle.py
test_ascend_host_callback.py
test_ascend_cpuinfer_callback.py
test_ascend_transfers.py
test_ascend_lifecycle.py
test_ascend_runtime_pipeline.py
23 passed, 1 skipped in 27.50s
```

The skip is the pre-existing CUDA-only compatibility-handle case in an Ascend-only container. Ascend native stream, 10k callback, CPUInfer callback, pinned D2H/H2D, callback/stream lifetime, and 1,000-cycle runtime pipeline all executed.

## Round 2C

The five focused modules collect 43 tests. After the merge-boundary fix, two consecutive full launches passed:

```text
43 passed, 1 warning in 18.51s
43 passed, 1 warning in 18.11s
```

The warning is torch_npu reporting base tensor format because internal format is disabled; it is non-failing and unrelated to numerical correctness.

Authoritative retained logs are under `/home/admin/kt_round2c_6f50c37/logs/round2c/` on A3. The disposable container was the only place where ordinary CPU build dependencies were installed.
