# V2 tail-statistics implementation checkpoint

State: `LOCAL_VERIFIED`; this is NOT the frozen Contract V2 specification.
No QD2/HD2 outputs have been produced or inspected.

`tools/tail_statistics.py` provides dependency-free offline calculations:

- one-sided binomial Clopper-Pearson upper confidence bound;
- smallest valid one-sided order-statistic tolerance rank;
- strict exceedance counting, separate from decision hard gates;
- complete-position validation and per-prompt maximum aggregation;
- fail-fast on nonfinite values, insufficient calibration units, missing or
  duplicate positions. No network dependencies or model thresholds are embedded.

The binomial inversion follows the
[NIST exact-interval construction](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
using alpha rather than alpha/2 for a one-sided bound. Tests check both the
published approximate example and the defining CDF equation. The NIST printed
endpoint is approximate; its comparison allows 1e-6 absolute error while the
defining equation is checked to 12 decimal places.

## Important interpretation

Position risks may be aggregated into one maximum per prompt. This removes
within-prompt pseudo-replication, but does NOT prove independence or identical
distribution between authored prompts. The eventual corpus sampling design
must justify its statistical unit and scope. Candidate pairs, repeated runs,
and templated variants must not inflate independent N.

Illustrative synthetic test only: for 128 independent units, 95% coverage and
alpha=0.05/3, rank 127 is the first admissible tolerance order statistic.
These numbers are NOT a chosen experiment risk budget or fitted reserve.
Zero exceedances from ten independent units cannot establish a 5% tail budget
at 95% one-sided confidence. Conversely a sufficiently large held-out sample
can satisfy a tail-rate gate despite a nonzero exceedance count. Stable-token
flips and ambiguity escapes remain independent hard failures.

## Verification

Command: `python3 -m unittest discover -s migration_bridge/round6_2/tests -v`

Result: 13 tests passed (4 capture-boundary tests plus 9 statistical tests).
`git diff --check` passed. All tests used local Python and synthetic inputs;
none establishes A3 numerical qualification.

Remaining: select and preregister the statistical sampling plan, freeze both
corpora and telemetry, complete A3 capture smoke, then derive from QD2 only.
