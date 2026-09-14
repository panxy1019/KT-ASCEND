# Operand Identity

The neutral harness loads one captured hidden row and the source model weights
once, then passes the same contiguous F32 materializations to each backend.
Every source weight is BF16 and all candidates share the same conversion.

Shapes:

- hidden X: `[1,2048]`
- gate/up W: `[1408,2048]`
- down X: `[1,1408]`
- down W: `[2048,1408]`

For each of E6, E8, E25 and E36, `operand_manifest` records source dtype,
shape, BF16-value SHA256, and F32-materialized SHA256 for gate/up/down. Every
captured row records X shape, dtype and SHA256. These values are shared by
LLAMAFILE, OpenBLAS, BLIS, ATLAS and ACL; no backend-specific weight conversion
was used.

Canonical manifest: `evidence/round4a2-openblas-all.json`.

ACL requires a physical B matrix in `[K,N]` layout. Its probe-only shim performs
an explicit value-preserving transpose from the common logical W; this layout
materialization is recorded and does not change any value.
