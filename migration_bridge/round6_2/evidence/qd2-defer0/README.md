# QD2 defer0 — complete captured baseline

Client EXIT_ZERO, 128 prompts ×64 positions =8192. All 16384 complete forwards
passed the collection driver's input/history association and tensor-hash
checks. Independent audits additionally verify one target/lookahead pair per
request, sequence length, finite logits, exactly-once CPU ownership and
producer/consumer terminal transitions for both forwards. Target transition
audit:8192/8192, failures0.

This is a completed **single-mode captured baseline**, not QD2 numerical
acceptance. No cross-mode metric or reserve was evaluated at this commit.

## Exact artifacts

| Artifact | SHA256 |
| --- | --- |
| Raw manifest.json | 806b3ffaec9de0169bbc12c78a17bdb89bb306fd15bba5339ef27308e82302fe |
| Manifest canonical payload | 536df41e2bc8b59c041bc6d04eaf71387985cbe4d9b0fe274115cb1ca4522b8a |
| manifest.json.gz | e82e69285ef312234371d0ab1b634e1c15cf53888f81bf7f3ae1f97076c4d3f6 |
| Raw forward_audit.json | 5d0c793e4de44413c3deb704396c5ad43e36f28cdc8e9297b1eb6d7fbddea322 |

The gzip files are lossless `gzip -n -k` archives. Local decompression hashes
were verified against the original A3 files. No original capture was removed.
All raw captures remain under
`/workspace/results/r62-qd2-defer0-server/tensors/`; per-prompt logits and
per-position metadata remain under `/workspace/results/r62-qd2-defer0/`.

`complete_matrix_audit.json` and `transition_audit.json` contain the independent
audit results. `../qd2-defer0-client-result.json` records client execution.

## Planned shutdown and resource boundary

The service was stopped only after successful client completion. Exact
guardian281573 was validated from its command line and sent SIGTERM inside
the dedicated container. It cleaned only its recorded child process group
281578. `planned-server-stop.json` therefore says CANCELLED/returncode-9;
that is a deliberate post-success shutdown, not an inference failure.

Only our process group was signaled. During subsequent read-only preflight,
NPU0 health was OK and HBM usage4%. The previously observed external PID1510639
was then absent; no command was issued against it. Shared workload conditions
changed during the collection window. Consequently these instrumented run
durations must not be used as an isolated defer0/defer1 performance comparison.
