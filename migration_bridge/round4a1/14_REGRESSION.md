# Regression

## Completed

- Python compile checks for modified wrappers and diagnostics: PASS
- parent and child `git diff --check`: PASS
- C++ A3 rebuild after final LLAMAFILE changes: PASS
- registered A3 Ascend CPU+NPU routing fixture: PASS (`1 passed`)
- full-model P1 service startup and 45 requests: PASS operationally
- finite and prefix-deterministic checks: PASS

## Numerical regression recipe

The committed recipe is `tools/probe_real_expert_numerics.py`. The capture is
identified by model revision/path, prompt, layer, expert IDs, pass indices and
SHA256 values in the compact JSON. Large hidden-state tensors are intentionally
not committed.

Expected envelope after F1/F2:

- gate/up and fused multiply frequently bitwise equal to captured NPU;
- down relative L2 normally below 0.001 for R2 and often zero;
- no non-finite values;
- P1 must still be treated as failed unless 45/45 tokens and <=0.20 logprob are
  both achieved.

## Not run due to gate

The broad Round2A/Round2B/Round2C/Round3/P2/P3 regression campaign was not
started after P1 remained red. The targeted registered test protects the
modified CPU/NPU boundary while avoiding a false claim of release readiness.
