# Round14 current execution

Status: `ROUND14_PARTIAL_WITH_GAPS` (terminal, no background controller, timers paused).

- A: complete. Formal order v235 KT-A → v251 Native-A → v252 Native-B → v238 KT-B. Native W2 serial output throughput was 3.228323 and 3.341694 tok/s; KT was 3.042707 and 2.941903 tok/s. This is a 1-NPU heterogeneous versus 8-NPU pure deployment comparison.
- B: partial. v273 completed 64/64 public-proxy requests and H512 resource/cancel/reuse boundary, but 8 frozen English automatic checks failed and 9 code items remain human-review pending. Only one complete block exists; H129 remains default; H768 was not run.
- C profile: complete. v274/v275 identify GDN as the largest stable hotspot (about 61.16% Extend and 23.52% Decode attributable share).
- C candidate: rejected at component resource gate. Full Qwen3.5 parent was numerically invalid because it used swish instead of frozen sigmoid gating. NPU GDN core plus frozen Qwen4 sigmoid gate passed relative-L2 checks, but exact replay grew allocator by 171,424,768 bytes per sequence. No component timing, model smoke, or request ABBA claim is permitted.
- Optional concurrency: not run.
- Final resources: physical12 empty; no owned container/process; 179-file targeted cache retirement PASS, resident pages 34,048 → 0; training container unchanged.
- Budget: 14/24 model groups, 238/664 generations, 1/14 aborts, 6/24 component attempts, 4.16464/128 NPU device-hours.
- Final verification: 62 Round14 unit tests PASS, 34 Python files compile, formal archive hashes and budget bounds PASS; see `FINAL_VERIFICATION.json` and `HASHES.json`.

Canonical next action: none inside Round14. Any continuation requires a new task and must start from `FINAL_CONFIG.json`; do not enable the GDN candidate or H512 by default.
