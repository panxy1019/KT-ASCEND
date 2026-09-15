# QD3 / HD3 / FD3 authoring constraints — before pilot exposure results

Formal inputs have not yet been created. N is selected only by the committed
pilot planning rule, from128/256/512. Independently of pilot category hit rates,
freeze these proportions (matching the pilot and prior broad workload mix):

| Category | Fraction | Count if N=128 |
| --- | ---: | ---: |
| English | 3/16 | 24 |
| Chinese | 3/16 | 24 |
| Code/reasoning | 4/16 | 32 |
| Math/numeric | 3/16 | 24 |
| Structured formats | 3/16 | 24 |

Both QD3 and HD3 use the same count proportions,64 teacher-forced targets per
prompt, but independently authored task instances. No category reweighting
after observing exposure. Do not copy pilot prompts into either set. Do not
create the holdout by merely changing numbers/names in calibration templates.
Record family/topic metadata and review semantic template overlap as well as
normalized text and input-token equality. Exact dedupe alone is insufficient.

FD3:8 new prompts (2 English,2 Chinese,2 math,1 structured,1 code). Exclude all
historical corpora, pilot, QD3 and HD3. This set evaluates true free generation,
not forced outputs. Downstream quality retains the original frozen manifest.

All source text, token IDs, source hashes, tokenizer identity, order and family
metadata must be frozen before each formal mode's output. Defer0-generated
histories are frozen before both forced replays. HD3 histories and replays are
not generated before candidate/runtime prerequisites. No append-until-coverage,
difficult-prompt deletion, exposure-conditioned filtering, or holdout fitting.

Statistical confidence remains explicitly conditional on independent,
exchangeable prompt blocks. Authored task diversity is not a proof of this
assumption or an unconditional guarantee for arbitrary production traffic.
