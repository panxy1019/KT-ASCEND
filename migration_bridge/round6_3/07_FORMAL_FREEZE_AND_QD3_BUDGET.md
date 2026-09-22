# Formal input/method freeze and QD3 entry

Round6.2 remains BLOCKED_INSUFFICIENT_EXPOSURE. No retrospective pass.
The superseded early pilot is excluded. Fresh pilot and observer corrections
are prerequisites, not formal numerical qualification.

## Pilot decision — A3_VERIFIED / ANALYTIC_DERIVED

Fresh pilot32x64:2048 positions,1814 exposed,32/32 exposed prompts.
Final physical audit2080 forwards=2048 targets+32 lookahead; unknown0/incomplete0.
Mean exposure56.6875 per prompt. Registered lower mean40.4999105568.
N128 conservative exposure estimate4147.9828269, so first eligible size is128.
This is a conditional statistical planning bound, not a guarantee for arbitrary
traffic. It does not replace the formal500-position exposure gate.

## New inputs

| Corpus | Prompts | Targets | Canonical SHA256 |
| --- | ---: | ---: | --- |
| QD3 |128|8192|bec960772ccad4fbbe945ad76211852b86ff0cc9c553cdf962742ef45fc710e2|
| HD3 |128|8192|bc1aac578b88806693e9ee3327d69cba24c02e3f144fc0e53150eec135dcc6fa|
| FD3 |8|512|59d35f221350969f5f17b92d2158212b5988da0d7eae7c30983ee9d7dade013c|

Exact normalized-text audit covers1625 historical strings and264 new tasks.
Initial draft semantic restatements were replaced BEFORE formal inference;
all audit versions and git history retained. The final46 lexical flags were
reviewed as predominantly instruction boilerplate or distinct tasks sharing
general concepts. See `corpora/formal_source_review.json` for limitations.
No routing/accuracy-based filtering. Historical token-ID audit covers398 unique
inputs including the R63 pilot and engineering controls: all three corpora PASS.
Cross-QD3/HD3/FD3 token duplicates0. Pinned tokenizer hash checked on A3.

Input-only freeze does not generate HD3 histories. Those remain forbidden before
candidate commit plus candidate-bound D1 runtime PASS.

## Statistical protocol

`PREREGISTRATION.json` canonical SHA:
`19685bff9c72337873c466067ba76c579e4e562b25170d84d46e3d5185f83eb0`.

V2 risk metrics unchanged: contraction, full-logit maxabs, relative L2;
baseline top32 plus natural serving anchor; same natural baseline history.
Prompt-block coverage95%, alpha=.05/3, tail risk budget5%, N128 rank127.
Hard gates unchanged: stable top1, stable candidate pair ordering, ambiguity
membership, outside-U zero, NO_EXPOSURE byte equality, finite outputs.
No fit to pilot numerical outputs; none were evaluated.

## Tool verification

A3 CPU-only27/27 tests PASS, including a SYNTHETIC identity evaluator fixture.
Its zero reserves are a unit-test expectation, NOT a derived model contract.
True observer OFF/full token parity20/20 in both modes, minimal reference/full
logits1300/1300 exact per mode, plus prior full/compact retained-byte parity.

## QD3 execution budget

Three sequential model runs: natural defer0, forced defer0, forced defer1.
Each128 requests,8192 target positions,8320 actual forwards including lookahead.
Total24576 targets and24960 physical forwards across the three runs.

Pilot request P50=9.115s, P95=9.921s. Estimated model requests3809.7s;
including startup/audit allowance4709.7s (~79minutes). This is an estimate.
Per-client timeout3000s; owned server guard3600s; analytical step1800s.
Minimum free disk40GiB; last read-only preflight showed about1.3TiB available.
Only NPU0 and the project container; production overlap/batch1 unchanged.

Sequence: all physical/index audits -> defer0 self-history byte parity -> causal
position exports -> exposure>=500 -> numerical derivation. Any failure stops
downstream stages. Successful derivation still requires candidate commit and
D1 runtime; the pipeline cannot run HD3 or quality and cannot promote READY.
