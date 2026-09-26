# Round10 T5 current provider and state qualification

Status: **PASS_T5_CURRENT_PROVIDER_AND_STATE_NO_PERFORMANCE_CANDIDATE**.

GDN v103 covered real weights at layers 0/24/46, FP32 and BF16, and 1/3/17 tokens against an independent FP64 equation. Maximum relative error was 1.0874964e-06 for FP32 and 0.00506596066 for BF16; repeat hashes were exact, decode/reset states matched, and allocator state returned to zero.

QSA v103 covered real weights at layers 3/23/47 and lengths 1/3/4/9/17. Maximum relative error was 1.07935475e-06 for FP32 and 0.00511659834 for BF16. Compressed-key/index checks, signed multi-head ranking, Top-K stress, repeat determinism, and allocator retirement passed.

PLE v105 ran on physical12 and passed all eight original-row/ownership/lifecycle checks. Three cases were byte exact, including reuse after cancellation; peak extra allocation was 493056 bytes. v103 and v104 failures are retained as evidence of gate and controller defects; neither failure is overwritten.

The current full-model state pool is covered by four v102 commits. Every commit preserved all 124 live tensor records: 72 GDN, 49 QSA, one PLE, and two request metadata tensors. T4 also supplied the current cancel/reuse proof.

T2 did not identify a GDN/QSA/PLE hotspot that justified a single-variable B candidate. The existing providers remain enabled unchanged; performance-candidate usage is zero. This is a narrow current-provider and state qualification, not long-context or concurrent qualification.
