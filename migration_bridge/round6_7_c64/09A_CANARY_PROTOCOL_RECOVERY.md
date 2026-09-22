# C64 canary protocol recovery — 2026-09-17

## Cause and fix

The tool incorrectly required eight prompts and four budgets. The unchanged
frozen `r67-compact-v1` manifest has four prompts and budgets
`[16,8,8,8,8,8,4,2]`: 32 requests. The startup assertion failed before serving.
The fix validates the actual frozen structure, unique prompt IDs, and replay
budgets; restart/observer requests now read their budgets from the manifest.
No model, placement, numerical threshold, native binary, or CPU wrapper changed.

## Evidence

- LOCAL_VERIFIED: two protocol unit tests, including rejection of changed
  budgets, replay budgets and duplicate IDs; syntax and diff checks pass.
- A3_VERIFIED: actual frozen manifest validation passes; shortened budget
  negative control is rejected.
- Completed operator matrix: 52/52 configurations; integrity audit PASS;
  156 real decode cases, 64 prefill-shape diagnostics, ten repeats;
  maximum relative L2 0.0017353741484431432.
- C32 real decode parity against original: 156/156 exact.
- At 06:26 UTC, R16-C64 canary retry had completed 6/32 requests, with
  same-path/prefix checks passing so far and 64-worker affinity verified.
  This is successful test entry, **not** full-model qualification PASS.

## Preserved attempts and active run

Container: `kt-r67-c64-development-v1` (physical NPU2, NUMA4/5, CPU128–191).
Original failure: `/workspace/c64-dev/run_canary.log`, `r16-canary-v1`.
A deployment attempt before the corrected file arrived also failed the same
pre-model assertion: `r16-canary-v2.log`; retained without overwriting.
Corrected tool snapshot: `tools-v12/.../tools/run_canary.py`.
Active output: `/workspace/c64-dev/r16-canary-v3/`.
Active log: `/workspace/c64-dev/r16-canary-v3.log`.
Exit record: `/workspace/c64-dev/r16-canary-v3.exit`.

Native SHA256:
`a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077`.
Compact manifest SHA256:
`555f578af6d816b25ee5efe6590a472b654f27a74614c011bba599a8f8c24933`.

The standalone retry automatically runs compact, clean restart and observer
parity. The earlier full continuation remains blocked and still references
v1; it has **not** been silently promoted or resumed. Performance and CMMLU
must be resumed explicitly against the new successful summary and its hashes.
Existing R16/R8 C32 product jobs were not modified. Active monitoring stops
after verified test entry, per the user's request.
