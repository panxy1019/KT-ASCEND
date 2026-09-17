# Explicit threshold commit freeze before post-DEV evaluation

Date: 2026-09-17. Status: **LOCAL_VERIFIED**, prepared tooling only.
No Dynamic threshold has been selected or qualified on A3 yet.

## Gap closed

The prepared threshold sweep correctly uses DEV only and writes a selected
candidate JSON. Previously, later scripts checked its `FROZEN_NOT_HELDOUT_QUALIFIED`
label/profile hash but did not require an acknowledgment of the local commit.
Formal performance also accepted a bare CLI threshold integer. That was not a
sufficient handoff enforcing the user's commit-before-HOLDOUT/CMMLU boundary.

The new pure-Python `frozen_threshold.py` requires matching candidate bytes,
Static profile hash, DEV provenance, an explicit full40-character local candidate
commit and false quality/held-out fitting flags. It is used before any session
launch by Dynamic HOLDOUT, runtime quality, formal Dynamic performance, and
Deferred evaluation on a selected Dynamic base. Static-only paths do not require
a threshold acknowledgment. The healthy Static release-v6 is unchanged.

## Ordered freeze procedure

1. Finish all three DEV migration system probes and the DEV-only sweep. Preserve
   closed raw sessions, candidate results, selected threshold and source/runtime
   identity. Do not evaluate HOLDOUT or quality while selection is pending.
2. Pull complete artifacts after remote SHA verification and completed SCP.
   Verify all hashes locally, review DEV-only selection and commit candidate
   files on the Round7 branch. No push/tag/main merge.
3. Resolve the full commit. Check that `git show <commit>:<candidate-path>`
   produces exactly the frozen selected JSON bytes; verify its profile and DEV
   hashes against the independently frozen artifacts. Do not infer this from
   the current worktree alone.
4. Only then create a separate acknowledgment:

```json
{
  "local_candidate_commit": "<full40 SHA verified to contain selected bytes>",
  "threshold_freeze_sha256": "<SHA256 of selected candidate JSON>",
  "profile_sha256": "<SHA256 of selected Static mask JSON>",
  "dev_sha256": "<independently frozen SCHED_DEV SHA256>"
}
```

5. Stage the acknowledgment beside the immutable selected candidate on A3.
   Every post-DEV Dynamic consumer must use `--threshold-freeze` and
   `--threshold-ack`. These bind the exact candidate and profile; optional
   performance `--threshold` must equal the selected value or is rejected.

The acknowledgment is an explicit provenance handoff, **not a cryptographic
Git attestation**: the deployed runtime lacks the local audit Git tree and cannot
independently prove a named commit contains the bytes. Step3 and the preserved
local evidence are essential. No script is authorized to fabricate an ack merely
because a JSON contains a freeze label.

## Rejection and verification

Missing/short commit, changed selected bytes, changed Static profile, changed or
missing DEV hash, unsupported threshold, missing freeze state, or quality/held-out
fitting flags fail before a server session starts. Do not repair an identity
mismatch by rewriting the acknowledgment: investigate and preserve the failed
invocation and artifacts.

Six targeted unit tests cover matching bindings and the negative cases. Complete
local suite: **71/71 PASS**; `git diff --check` PASS. No A3 model/runtime or
threshold qualification is claimed. This change must be staged as a new future
tool release, not edited into active v6 or staged v10 in place.
