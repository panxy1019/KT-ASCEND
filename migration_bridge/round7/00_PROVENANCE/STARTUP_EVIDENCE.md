# Round7 startup evidence — not final qualification

Date: 2026-09-17 UTC. Status: DEV statistics collection in progress.

## Evidence classification

| Check | State | Evidence |
|---|---|---|
| Full local base resolved | CODE_INSPECTED | `bc2e5d53e05d97db83b40c09a6f80cddff0549ab` |
| Actual A3 imported files/native hashes | A3_VERIFIED | `../raw/startup/runtime-v2/provenance.json` |
| Pilot repeated greedy output / observer parity | A3_VERIFIED | Running attempt2 pilot, progress JSON on A3 |
| Exact pilot forward association | A3_VERIFIED | `../raw/startup/runtime-v2/uniform-dev-v1/forward-attribution-epoch-1.json` |
| Full DEV completeness | NOT_COMPLETE | Must audit all 256 requests before fitting |
| Frequency mask/system/quality/performance | NOT_RUN | Candidate not derived from partial statistics |
| Dynamic / Deferred qualification | NOT_RUN | Tool preparation is not A3 qualification |
| Pure contract/migration/SSE timing tests | LOCAL_VERIFIED | 13 tests passed; no accelerator qualification implied |

## Preserved recovery, not a numerical relaxation

Attempt1 rejected an integrity expectation before DEV collection because default
production overlap evaluated one extra final-token forward. Raw evidence and
reaping remain in `raw/startup/runtime-v1`; the failed status remains failed.

Attempt2 associates recorder forwards with exact input IDs and positions and
returned greedy histories. The pilot contains nine raw forwards: eight required
for the eight-token response, plus one proven unused final-token forward. All
nine are attributed, unknown=0. The extra pass is retained in raw evidence and
excluded from clean activation fitting. An unknown or missing pass still fails.
This treatment does not change model arithmetic, ownership, quality criterion,
corpus, or expert selection. Full DEV will undergo the same association audit.

## Startup artifact hashes

- Recovery archive SHA256: `e9281c8561075bbb4c366ed4b06adccd5c8f5c6b84e7d034233ad4559b90e4b2`.
- Attempt2 provenance: `5d98ea12e65a6ceaade2d80a99f4f515f332bbff66340ace80bcba988ea0d507`.
- Raw pilot stat: `0b49ac07ebe93040d9c844ed2afacf1a1bfcc2405975cf975602e598a1bf984a`.
- Pilot attribution: `ae18f7de9159d0833b0337297e4b6f994c2ed1426f60426d2b119aebe9fc3871`.

The startup archive is a byte-frozen recovery bundle, not a final artifact
manifest. Active DEV outputs will be imported and SHA-frozen after completion.

## Subsequent tooling seam

Prepared default-off Dynamic auditing checks all copied BF16 slots against the
official full-context source, per-layer fixed slot count, logical/device mapping,
Python/device/wrapper masks and serial generation publication. It tests
no-trigger, request A repeat, A-to-B adaptation and restart. It has not yet run on
A3; it cannot establish arbitrary-concurrency safety.

Separate observer-OFF performance measures serial three-block ABBA workloads.
Client SSE TTFT is an upper-bound delivery metric, not internal device prefill
time. Coalesced output cannot support a decode measurement and is reported as
unavailable. Internal CPU-wait instrumentation remains a separate untimed task.

Current healthy collection uses frozen release-v5 tools. Later hooks are staged
locally and must be deployed separately after the current run has reaped.
No current model weights, arithmetic, CPU64 topology or imported serving source
were replaced. No push, main merge, tag or business-container operation occurred.

## Continuation

The current task has a 15-minute continuation heartbeat (`round7`) to resume
ordered work when long runs finish. Keep the Codex app available for these
follow-ups; the A3 collector itself is already detached and does not require an
open SSH session. Unchanged state stays quiet; completed stages/failures/fixes
are reported. See `../AUTONOMOUS_CONTINUATION.md` for execution boundaries.
