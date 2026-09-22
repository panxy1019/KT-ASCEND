# F9 CPU FP8 K/V page-record reference format

Date: 2026-09-21

Status: `CPU_REFERENCE_ONLY`

Native/backend status: `NOT_IMPLEMENTED_UNQUALIFIED`

This format is the canonical host representation for F9 CPU experiments. It
does not install a page table, free or load a physical page, alter SGLang's K/V
pool, enable the native FP8 cache option, or qualify an NPU FP8 consumer. The
QSA compressed index remains BF16 and is outside this record.

## Fixed geometry and ownership

Version 1 is intentionally specific to the actual TP1 Qwen4 reference geometry:

- page capacity: 4 tokens;
- valid tokens: 1 through 4, allowing a bounded tail record;
- K/V heads: 2;
- head width: 256;
- source and decoded dtype: CPU BF16;
- quantized payload: FP8 E4M3FN, one byte per element;
- QSA layers: model layer IDs `3, 7, 11, ..., 47`;
- logical pages: `0..32768`, covering the 128K plan plus its reserved page.

Every record owns exactly one K plane and one V plane for one layer and one
logical page. Its `ContextStateIdentity` must have component `QSA_KV`, a
nonempty request ID, a nonnegative allocation epoch, and `prefix_id=None`.
Prefix-shared records are rejected until a later integration binds prefix
ownership, recurrent-state compatibility, allocation epochs, and release.

Decode requires the caller to supply the expected identity, logical page, and
layer. A different request, allocation epoch, page, layer, component, or prefix
state fails before tensors are returned.

## Quantization

K and V use independent, positive, finite FP32 scales:

```text
scale = 1                                      when amax == 0
scale = FP32(amax / finfo(float8_e4m3fn).max) otherwise
payload = E4M3FN(clamp(FP32(BF16_input) / scale, -max, max))
decoded = BF16(FP32(E4M3FN(payload)) * scale)
```

NaN or infinity in either source plane is rejected. Decode also rejects an
E4M3FN NaN byte even when an independently constructed public record has
canonical metadata and a recomputed matching digest. The scale is stored as the
exact four little-endian IEEE-754 bytes, hex-encoded in metadata. This avoids
depending on JSON decimal-float rendering and lets validation reject a scale
that is not exactly representable as canonical FP32. An all-zero plane uses
scale 1 and has all-zero payload bytes.

A full layer record holds 2,048 K bytes and 2,048 V bytes. Across all 12 QSA
layers, a full logical page therefore holds 49,152 FP8 payload bytes and 96
scale bytes, matching the model-geometry audit before metadata and digests.

## Canonical metadata

Metadata is ASCII JSON encoded with sorted keys and separators `,` and `:` with
no whitespace or NaN values. Version 1 binds:

- scope and format version;
- request ID, null prefix ID, allocation epoch, and `QSA_KV` component;
- logical page, layer ID, valid token count, and exact shape;
- page capacity, head count, head width, and BF16 source dtype;
- K/V format names and exact little-endian FP32 scale bytes.

Decode reconstructs this canonical metadata from the record fields and requires
byte equality. Metadata is limited to 4,096 bytes during encode, direct record
serialization, and parse. The format version must be an exact integer, so a
boolean cannot pass by Python integer equality. Adding, removing, reordering,
or changing metadata without a new format version is invalid.

## Binary record

All integers use little-endian unsigned 32-bit encoding:

```text
offset  size  field
0       8     magic: "R9KVFP8\\0"
8       4     metadata byte length
12      4     K payload byte length
16      4     V payload byte length
20      M     canonical metadata JSON
20+M    K     K E4M3FN bytes
...     V     V E4M3FN bytes
...     32    SHA-256 digest bytes
```

The digest covers the header, metadata, K bytes, and V bytes. Parsing rejects a
wrong magic, zero or excessive lengths, truncation, trailing bytes, a digest
mismatch, noncanonical metadata, invalid scales, and payload lengths that do
not equal `valid_tokens * 2 * 256`. Re-encoding identical inputs and identity
produces byte-identical records and digests.

Decode copies each payload into newly allocated CPU `uint8` storage before
interpreting it as E4M3FN. It does not use a buffer-aliasing constructor whose
placement could be redirected by an accelerator transfer shim. Returned BF16
tensors therefore belong to the CPU reference path.

## Validation scope

`tests/test_kv_quant_reference.py` covers:

- lazy module import without importing PyTorch;
- deterministic full-page encode, parse, decode, and record bytes;
- zero and partial-tail pages;
- BF16/CPU/fixed-geometry enforcement, including rejection of the erroneous
  one-head TP8 shape;
- source NaN/infinity and forged E4M3FN NaN-payload rejection;
- wrong request, stale allocation epoch, wrong page, wrong layer, wrong
  component, and prefix-sharing rejection;
- payload, metadata, digest, scale, truncation, magic, and trailing-byte
  tampering;
- bounded metadata, exact-integer versions, and explicit CPU-owned decode
  storage.

The current local system Python does not contain PyTorch. It can validate the
lazy-import contract and syntax but skips tensor/FP8 cases. Those tests must be
run in the existing CPU PyTorch environment before this reference format is
accepted. Passing them still makes no device, native FP8, allocator, long
context, or quality-threshold claim.
