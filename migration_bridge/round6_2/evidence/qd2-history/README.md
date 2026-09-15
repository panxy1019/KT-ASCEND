# Frozen QD2 baseline history

A3 defer0 generation completed 128 prompts x 64 tokens (8192 tokens).
Client supervisor reported EXIT_ZERO, returncode 0, success true.
No retries or prompt filtering were performed.

`manifest.json.gz` is a deterministic gzip archive (`gzip -n`) of the exact
original manifest, including all API responses and generated token histories.
It is not a reconstructed or summarized substitute.

Original file-byte SHA256:
`d1c547a36bba5d09d8e665e43933b44e2ba2406f411f35645c970f14bf1f3188`.

Canonical manifest SHA256 (excluding its own sha256 field):
`b1a6a6b70c2bc5c88d947e0b164260910af8de3d24dfba52d1c21c949bba2630`.

QD2 canonical corpus SHA256:
`ca8b8ba8e15585c93467cd2f26a80939a2de69775ab562bc5f91cb54963081a9`.

Verify without replacing any file:

```bash
gzip -dc manifest.json.gz | sha256sum
```

A3 original: `/workspace/results/r62-qd2-history/manifest.json`.
Client evidence: `/workspace/results/r62-client-qd2-history/`.
Server evidence: `/workspace/results/r62-guard-qd2-history/`.
The server was stopped only after the history client completed successfully.
Both upcoming modes must replay these exact histories. HD2 remains unrun.
