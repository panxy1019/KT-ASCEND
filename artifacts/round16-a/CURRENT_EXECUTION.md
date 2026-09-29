# Round16-A current execution

Updated: 2026-09-29 11:31 UTC.

Status: `A1_INSUFFICIENT_ATTRIBUTION`; selected candidate: `null`.

Two fresh S0 profile processes completed. Both contain six frozen requests,
complete serialized W1/W2 measurement batches, all-layer Host expert
boundaries, and sampled early/middle/late QSA, GDN and resident-expert NPU
events. No historical profile replaced either fresh block.

Direct diagnostic NPU event medians were:

| Block | QSA indexer | GDN | resident expert |
|---|---:|---:|---:|
| A | 2.868 ms | 1.057 ms | 0.416 ms |
| B | 2.986 ms | 1.088 ms | 0.434 ms |

These spans are not added. They do not establish a request critical-path
percentage. The native C128 worker ABI exposes no start/end callback, so CPU
Expert duration remains `UNKNOWN_NATIVE_ABI_NO_CALLBACK`. The observer sees
Host submit, sync, merge and the routed-expert parent, but it does not provide
one directly isolated `HOST_SYNC_EXPOSED_WAIT` interval or one directly
isolated `HOST_LAUNCH_METADATA_CRITICAL` interval. Parent-minus-child,
overlapping-span addition and Host/device clock addition remain forbidden.

Consequently:

- A2 exact same-layer overlap: `NOT_SELECTED_EVIDENCE_GAP`.
- A3 larger NPU segment graph: `NOT_SELECTED_EVIDENCE_GAP`.
- A model smoke and A request ABBA: not run because no component candidate was
  eligible.
- A16 compute identity: S0 unchanged.

This is not `NO_GAIN`; no A candidate reached a performance test. The precise
remaining contract gap is a reviewed direct interval for CPU-result exposed
wait and Host launch/metadata cost without changing production synchronization.

Canonical outputs are `A1_CRITICAL_PATH.json`, `A1_CRITICAL_PATH.md`,
`A_REGION_DEPENDENCY_DAG.json`, and `A_CANDIDATE_DECISION.json`. Their earlier
pre-device placeholders are preserved with `_PRE_DEVICE` suffixes. Both fresh
evidence blocks remain under `fresh-profile-a-v003` and
`fresh-profile-b-v003`.
