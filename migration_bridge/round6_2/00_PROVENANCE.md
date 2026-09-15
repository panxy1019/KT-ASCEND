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
`../round6_1/00_PROVENANCE.md`. The 2026-09-15 A3 QD2 preflight rehashed the
actual tokenizer, D1 placement and GGUF and matched all frozen values.

SSH to admin@110.123.0.3 succeeded and returned `a3-server-00`. This proves
connectivity only, not NPU availability or execution-environment equivalence.
This was the initial connectivity-only checkpoint, not the current execution
state. Later actions are recorded in `EXECUTION_CHECKPOINT.md` and
`TELEMETRY_SMOKE_INCIDENTS.md`. Only this project's disposable container and
explicitly owned test groups were used; host libraries/business tasks were
not modified.

## Execution sequence and current state

1. Provenance: tag/branch established; A3 source overlays and artifact hashes recorded.
2. Contract method preregistered; no numerical values fitted.
3. QD2/HD2 inputs jointly frozen before QD2 model output.
4. Capture integrity: direct-forward fix, A3 smoke and full-logit parity passed
   within the documented measurement scope.
5. QD2 baseline history in progress; numerical comparison, retained regression,
   HD2 one-shot, quality, free generation, performance and final ADR remain pending.

A3 still checks out parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`
and child `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` with hash-recorded
source overlays. Do not label these runs as unmodified local branch HEAD.
The current run metadata is
`/workspace/results/round6_2-prequalification/qd2-history-provenance.json`.
Its byte SHA256 is `2c095d7151bad7d6e94f40bab45f2537d609d8cd2c66d86b853dde46c151808a`.

No candidate or final contract JSON is emitted with placeholder thresholds.
The statistics design must account for within-prompt position dependence;
thousands of correlated candidate pairs cannot be counted as independent trials.
