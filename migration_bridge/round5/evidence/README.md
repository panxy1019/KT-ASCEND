# Round 5 Evidence

This directory stores compact machine-readable and textual evidence collected
during the upstream-alignment audit and A3 requalification. Large builds,
model artifacts, tensor captures, and serving logs remain in the documented A3
artifact root and are referenced by SHA256.

`aligned/` contains the compact final P1/P2/P3 qualification summaries and
the post-P3 retained-regression logs.  `aligned/SHA256SUMS` covers every
committed compact artifact in that subtree.  Raw request bodies, full logits,
large expert-distribution tensors, GGUF files and model weights intentionally
remain outside Git; their identities are frozen in the profile documents.
