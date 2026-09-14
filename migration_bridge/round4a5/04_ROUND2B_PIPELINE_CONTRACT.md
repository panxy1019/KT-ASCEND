# Round 4A.5: retained Round 2B pipeline contract correction

Status: `RESOLVED_AS_STALE_TEST_CONTRACT`

The retained `test_d2h_cpuinfer_moe_h2d_pipeline` previously reported
`max_abs_error = 1.7014118346046923e+38` and infinite relative L2.  The
current A3 reproduction identified a test-side storage mismatch:

```text
LLAMAFILE CPU routed output: FP32
test output_host/output_npu: BF16
```

The CPU task writes a FP32 contribution into the provided host output pointer.
Providing BF16 storage allocated half the required byte count, producing the
observed invalid value.  This test is distinct from P2: the P2 Ascend path
already allocates `output_cpu` at `wrapper.output_dtype` (FP32).

The retained test now allocates FP32 host and NPU output buffers and casts only
at its final comparison boundary.  On A3 it passes with:

```text
max_abs_error = 0.000244140625
mean_abs_error = 0.00004513934254646301
relative_l2_error = 0.004940330512084555
```

This closes the old Round 2B dtype/buffer-size regression.  It does not change
the P2 placement, arithmetic, or P2 same-path blocker conclusion.
