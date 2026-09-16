# WP2 — captured-token mixed-expert reconstruction

Target evidence: `/workspace/results/r65-5a-canonical-v1/probe/canonical_mixed_oracle.json`.
State: A3_VERIFIED for the four captured failing-token layer inputs only.

For layers 1,9,17,26, the harness replays the actual LLAMAFILE CPU wrapper on
the complete six routes, and independently obtains each expert output using
the CPU wrapper or the deployed NPU grouped-MoE implementation. It applies
BF16-valued routing weights once, accumulates in ascending logical-ID order
in FP32 and casts to the captured routed output dtype.

- Four complete CPU replay outputs exactly match the captured CPU partials.
- Four canonical final routed outputs exactly match production captures.
- Fixed-order versus split CPU/NPU partial accumulation also gives the same
  final BF16 output in all four cases.
- Small FP32 pre-cast accumulation differences are retained in the report.

The largest per-expert CPU-versus-GMM relative-L2 in this bounded experiment
is approximately 0.0005173, below the unchanged 0.01 local safety gate.
Earlier F.linear-based NPU reference errors are a different comparator and
must not be silently relabeled as GMM errors.

This is the pre-outer-scaling routed sum. Shared expert/residual and outer
scaling are separately audited; this test alone does not prove them correct.
It does not exhaust prefill or every captured token/route combination.
No production implementation defect has been demonstrated by this test.
