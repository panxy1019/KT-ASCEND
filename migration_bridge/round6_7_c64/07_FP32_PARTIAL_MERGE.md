# FP32 TP partial merge

For F32 multi-TP only, down-projection partials are not BF16-rounded individually.
Gate/up/SwiGLU/route-weight boundaries remain unchanged. Slot0 then slot1 are summed
in FP32 using the retained deterministic merge. Single-TP C32 is unchanged.

The CPU routed-partial interface is FP32, so its output conversion is an identity
conversion. The model-level CPU/NPU FP32 merge and final BF16 cast remain in the
retained SGLang boundary; they are not replaced by an early CPU-only BF16 cast.

Test-build-only capture records raw partial0/partial1 before mutation, post-merge FP32,
and actual visible bytes. Layer17 smoke confirms byte-exact offline FP32 sum and
actual FP32 output. Capture is disabled unless both its build macro and output
environment are enabled; no capture variable is set in model/performance runs.

`tools/test_merge_contract.py` preserves a negative control: casting each partial
before reduction yields 1.0, versus 1.0078125 for the single final BF16 cast. This is
an analytic unit test, separate from the real native capture evidence.

Expanded real-operand qualification is still running; full-model status is pending.
