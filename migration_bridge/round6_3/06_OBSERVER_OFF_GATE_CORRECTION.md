# Observer qualification protocol correction

State: IN_PROGRESS; discovered during execution review, before any numerical
reserve derivation. Round6.2 historical state unchanged.

## Omission and evidence boundary

Full and compact observers both copy tensors to CPU. Their retained-byte parity
is valid evidence for compact capture, but does not establish no-observer parity.
The existing minimal-reference hook copies only final logits and is also not
fully observer-free. Required comparisons must distinguish all three profiles.
No numerical failure has been established by this omission.

Official pinned SGLang dump hook also performs `.cpu()` copying:
[upstream dumper](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/debug_utils/tensor_dump_forward_hook.py).
Such copying can affect synchronization; agreement between two observers alone
does not exclude observer-induced timing effects.

## Premature pilot disposition

- Pipeline root: `/workspace/results/r63-prequalification-pipeline-control`.
- Guardian cancellation end Unix1789485946.6839688, reasonCANCELLED.
- Pilot baseline completed32x64 targets/2080 forwards.
- Pilot defer1 reached206 indexed/forcing records, including partial request.
- Both retained, superseded, NOT_FOR_CALIBRATION. No numerical envelope fitting.
- No size plan existed at cancellation. QD3/HD3/quality NOT_RUN.
- Do not overwrite these roots. Fresh pilot execution is required after the
  missing gate, with unchanged frozen inputs and preregistered sample-size rule.

Nested guardian cancellation left the owned model group301697 alive. Its command
was checked against the exact pilot capture root and PGID301697 before TERM;
after TERM failed to exit, identity was rechecked and only that group received
KILL. Subsequent process inspection found no project serving process. Evidence
is in the remote `manual-owned-cleanup.json`; external tasks were not targeted.

## Correction

1. Disable the historical pipeline so it cannot silently resume.
2. Explicit launcher profiles: full, reference-logits-only, truly OFF.
3. OFF cannot force history. Route recorder/runtime settings remain unchanged.
4. Run natural20x64 engineering requests per defer mode; compare OFF trajectories
   against natural observed trajectories and finite HTTP logprobs.
5. Minimal-reference vs full compares complete final logits on identical natural
   histories. Audit all physical reference captures after stopping the server.
6. If any difference appears, preserve first failure; do not declare parity or
   advance pilot/qualification.
7. Session cancellation raises into cleanup; avoid outer guardians with shorter
   grace than their nested model guardians.

No production selector, placement, deferred budget, thread topology, numerical
contract, or quality corpus change is authorized by this correction.

## First OFF launch: startup-tool failure, not model failure

`r63-natural-off0-server` reached readiness but no generation was submitted.
The shared empty-session checker requires the capture directory to exist;
OFF intentionally does not instantiate a dumper. The controller therefore
cancelled its owned server. Its result remained STARTING due inadequate exception
reporting; no requests or parity evidence were produced. The correction creates
an empty audit directory in OFF mode and records exceptions explicitly.
Fresh retry root: `r63-natural-off0-retry1-server/control` (separate directories
with these suffixes), followed by `r63-observer-correction-retry1-pipeline`.
Original failed directories remain untouched. Local37 tests:31 PASS,6 skipped
because local torch is unavailable; these are not A3 verification claims.
