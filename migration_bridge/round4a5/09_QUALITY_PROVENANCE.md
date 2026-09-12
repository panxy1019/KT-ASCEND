# Round 4A.5 Frozen Quality Input Provenance

Status: `ROUND4A4_FROZEN_QUALITY_INPUT_RECOVERED`

Evidence state: `CODE_INSPECTED`, `LOCAL_VERIFIED`, and `A3_VERIFIED`

Date: 2026-09-11 UTC

## Decision

The exact Round 4A.4 frozen downstream-quality input is present in reachable
Git history and in the current Round 4A.5 tree. It is not necessary or truthful
to replace it with a reconstructed `round4a5-quality-v2` corpus.

Canonical path:

```text
migration_bridge/round4a4/corpora/d_manifest.json
```

The manifest contains the complete input records, not only dataset pointers:

| Benchmark | Frozen rows | Protocol |
|---|---:|---|
| C-Eval | 128 | conditional log-likelihood over A/B/C/D |
| GSM8K MC | 128 | deterministic four-choice conditional log-likelihood |

## Hash verification

The quality-freeze script defines the manifest identity as SHA-256 over the
UTF-8 JSON payload before the `sha256` field is inserted, serialized with:

```python
json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

An independent recomputation from the checked-out manifest produced:

```text
manifest-declared canonical SHA256 =
c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce

independently recomputed canonical SHA256 =
c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
```

This exactly matches the frozen Round 4A.4 quality manifest identity required
by the closure guide.

For clarity, the pretty-printed file bytes have a different transport hash:

```text
file size        = 212508 bytes
file SHA256      = 4cc516078e899e890958e23e3a8a006469e84ef02e163bd56fe83cbc0b6b7564
canonical SHA256 = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
```

The difference is expected: the canonical hash covers compact canonical JSON
without the self-referential `sha256` field, while the file hash covers the
indented manifest including that field.

## Reachable Git provenance

The manifest was first introduced by reachable commit:

```text
96f2c306e4b83a1a0f6d20effba08f24d177d5b6
test(round4a4): add pairwise qualification harness
```

Its Git blob is:

```text
5e647f615729bbd5fc6268dc2eece318584394bb
```

Reading the file directly from `96f2c306...` produces the same byte SHA256
`4cc51607...`. The file is therefore preserved in normal reachable history;
reflog or dangling-object recovery is not part of the identity chain.

`git fsck --full --no-reflogs --unreachable` was also run without pruning or
deleting any object. Unreachable objects exist in the local object database,
but none is needed to recover or define this corpus.

## Independent A3 copy

The preserved A3 source tree at:

```text
/home/admin/projects/KT-ASCEND-round4a5/
```

contains a byte-identical manifest:

```text
file SHA256      = 4cc516078e899e890958e23e3a8a006469e84ef02e163bd56fe83cbc0b6b7564
canonical SHA256 = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
C-Eval rows      = 128
GSM8K rows       = 128
```

That A3 tree also retains the Round 4A.4 All-NPU and P1 raw quality-result
artifacts. Their file hashes are:

| Artifact | SHA256 |
|---|---|
| `quality-allnpu-formal.json` | `7a5e3d73ae8dead2fcd259ed67102370a0e2a43ebdd366d4e9d3cc165f3457ec` |
| `quality-hybrid-p1-formal.json` | `dd76f78d12eda02c61b80cc10d2e3501d584509339c4263b79d3a7de17da42fb` |
| `quality-ab-summary.json` | `dd8ce2f50cbab6d8c35f5a3407cc67647c5e7939332355e3cfe637fdcd69373f` |

The existing A3 worktree is intentionally left untouched because it is dirty
with historical diagnostic changes. It is evidence storage, not the clean
closure execution tree.

## Historical quality evidence boundary

The recovered input makes an exact-contract P2 run possible. Historical P1
results remain evidence from their original A3 run:

| Benchmark | All-NPU | Hybrid P1 |
|---|---:|---:|
| C-Eval | 57/128 | 57/128 |
| GSM8K MC | 45/128 | 47/128 |

These scores are not promoted into new post-cleanup evidence merely because the
input was recovered.  The final P2 closure reproduced All-NPU and P2 in the
same Transformers `5.3.0.dev0` environment.  Their absolute scores were 27/128
for C-Eval and 31/128 for GSM8K in both modes.  The historical run used
Transformers `5.6.0`; its 57/45 scores therefore remain an explicitly separate
environment lineage.  See `11_P2_FINAL_QUALITY.md` for the paired result and
why cross-environment absolute scores were not mixed.

## Provenance gate

```text
EXPECTED_CANONICAL_SHA                = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
RECOMPUTED_LOCAL_CANONICAL_SHA        = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
RECOMPUTED_A3_CANONICAL_SHA           = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
ROUND4A4_FROZEN_QUALITY_INPUT_RECOVERED = YES
QUALITY_V2_REQUIRED                   = NO
QUALITY_INPUT_PROVENANCE              = CLOSED
```

The recovered input was run on A3 for current-environment All-NPU and P2.  Both
benchmarks had zero discordant correctness pairs, zero invalid outputs, and a
paired bootstrap 95% interval `[0,0]`; the downstream-quality execution gate is
closed.
