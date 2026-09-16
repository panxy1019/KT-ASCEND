# WP3 — cumulative diagnostic profiles

All runs use the original failing prompt, 64-position baseline history,
unchanged R32 masks on enabled layers, 32 workers and production overlap.
These are cause-localization profiles, not new qualification profiles.

| Profile | CPU layers | Maximum pairwise distortion | Overflow pairs | First failing position |
|---|---|---:|---:|---|
| C1 | 1 | 8.953125 | 34 | token54 |
| C2 | 1,9 | 11.84375 | 34 | token54 |
| C3 | 1,9,17 | 9.59375 | 27 | token54 |
| C4 (frozen historical replay) | 1,9,17,26 | 9.59375 | 27 | token54 |

All first-failure top1 IDs remain 82. This does not override the envelope.
C3/C4 equal maxima do not imply every logit vector is equal.

## Corrected causal interpretation

The minimum tested cumulative prefix already failing is C1, not C3/C4.
Prior single-layer qualification used Layer17, not Layer1. The data does not
support a requirement for multiple CPU layers before amplification can occur.
Additional CPU layers have non-monotonic effects on the final distortion.

C1 alone reproduces the Layer17 expert-set change E62→E29 at token54.
Its Layer1 prefill decoder-output relative-L2 is approximately 3.47262e-5.
At token54, Layer1 complete hidden/residual output and Layer2 input are exact,
but Layer2 attention differs. Historical attention state is therefore a necessary
part of further analysis; exact current-token input alone is not an exact-cache
control. Direct KV-buffer equality has not been measured by these captures.

## Tooling failure retained

The first C1 capture passed integrity, but the original full-corpus evaluator
rejected a forced diagnostic baseline (`WRONG_COMPARISON`). No original
evaluator check was removed. A separate diagnostic comparator joins the exact
input/history to the original natural oracle and reuses its metric. The valid
C1 capture was reused after checking its canonical placement hash. The failed
orchestration record remains preserved as tooling evidence.

Target paths:
`/workspace/results/r65-5a-bisection-v1/C1` and
`/workspace/results/r65-5a-bisection-v2/`.
