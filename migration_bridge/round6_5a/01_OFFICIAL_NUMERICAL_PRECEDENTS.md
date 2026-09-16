# Official precedents — applicability boundaries

## RAWINT4 load equivalence, PR 2092

[Merged upstream PR](https://github.com/kvcache-ai/ktransformers/pull/2092)
requires exact equality for identical kernels/weight bytes loaded through
different paths. Its one-versus-two-subpool comparison instead uses a mean
relative tolerance of 5e-3 because split down-projection accumulation changes
order. This is an x86 RAWINT4 precedent, not proof of an ARM F32 root cause.
The present launch fixes threadpool_count=1, so that specific multi-subpool
mechanism is not established here.

## BF16 operator safety

[Pinned BF16 CPU test](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/kt-kernel/test/per_commit/test_moe_avx2_accuracy_bf16.py)
compares AVX2 output against a float32 PyTorch computation rounded to BF16.
The actual assertion is mean absolute error divided by mean reference absolute
value <0.02 (not relative-L2, and not the nearby comment's 0.01).
Neither its metric nor its threshold replaces this project's frozen gates.

## Ascend bring-up, PR 2157

[Merged upstream PR](https://github.com/kvcache-ai/ktransformers/pull/2157)
documents an ACL callback-dispatch failure yielding zero CPU output and ARM ISA
and vec-dot compatibility problems. These are genuine implementation failures;
successful serving alone is insufficient evidence of expert correctness.

The requested specific streaming/hybrid near-tie statement has not yet been
re-fetched successfully in this audit. It is not used as independently verified
evidence here. In all cases, official examples do not justify accepting the
observed 9.59375 distortion or altering B_pair=2.1875.
