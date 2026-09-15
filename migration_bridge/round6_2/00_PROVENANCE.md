# Round 6.2 execution checkpoint

Started 2026-09-15. Parent base:
`cee33e53c37d0bd6de5dd1ab05b1c0966cb97438`.
SGLang starting revision: `35b12430806221ca01574f550a9b7f9eb8b4cf65`.
Branch: `codex/round6-2-deferred-contract-v2`.
Historical annotated tag: `round6-1-intended-deferred-approximation`, target
the parent base above. Not a production-promotion tag.

V1 remains `HISTORICAL_HELDOUT_REJECTED`; D1 remains blocked pending a qualified
new contract. D2/D3 are not authorized in this round.

Preserve the inherited llama.cpp checkout mismatch and dirty state. No reset,
staging or build using that checkout has been performed in this round.
The inherited frozen model/D1 identities are recorded in
`../round6_1/00_PROVENANCE.md`; full machine-readable revalidation is pending.

SSH to admin@110.123.0.3 succeeded and returned `a3-server-00`. This proves
connectivity only, not NPU availability or execution-environment equivalence.
No A3 process, container or host configuration has been modified this round.

## Execution sequence and current state

1. Provenance: historical tag/branch established; artifact revalidation pending.
2. Contract preregistration and telemetry: pending; no numerical values fitted.
3. Freeze independent QD2 and HD2 before producing QD2 model output: pending.
4. Capture integrity: static defect reproduced locally; opt-in fix added.
   A3 smoke remains pending and must follow corpus/spec freeze.
5. QD2 calibration, retained regression, HD2 one-shot, quality, free generation,
   performance and final ADR: NOT_RUN.

No candidate or final contract JSON is emitted with placeholder thresholds.
The statistics design must account for within-prompt position dependence;
thousands of correlated candidate pairs cannot be counted as independent trials.
