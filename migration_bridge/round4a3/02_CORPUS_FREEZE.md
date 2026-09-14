# Round 4A.3 Corpus Freeze

## Corpus Q — qualification only

- prompts: 9
- positions: 64 matched-history positions per prompt (576 total)
- source: the frozen Round 4A validation corpus
- categories: English, Chinese, math, structured output
- canonical SHA256: `7617682ccd1d004f6221d35b9e484465ebb6c2a40359481f047f3fbb298b439a`

Q is the only corpus allowed to derive `epsilon_logit`, evaluate C in
`{1,2,3,4}`, and construct the candidate acceptance contract.

## Corpus H — held-out numerical validation

- prompts: 12
- positions: 64 matched-history positions per prompt (768 total)
- categories: English factual 3, Chinese factual 3, math/numeric 2,
  structured/JSON 2, code/reasoning 2
- no prompt duplicates with the recorded S/Q/T corpora
- canonical SHA256: `dc137b4c9531029f58eb2daa4746760840a301e6930ee6b98da12f5edc06c2cc`

H was frozen before any Q numerical result was calculated. H is not available
to epsilon fitting, C selection, or tie-set design. If it informs a revised
hypothesis, it becomes development evidence and a new H2 is required.

## Corpus D — downstream quality

- GSM8K test: first 32 rows from revision
  `740312add88f781978c0658806c59bc2815b9866`
- C-Eval validation: 32 rows, eight each from Chinese language/literature,
  high-school history, high-school geography, and basic medicine, revision
  `617524a00b307ff6f9933702f724131fe12ca7ce`
- canonical manifest SHA256: `7dfb4badd385e26ad43a9bb7ebd03ca20b75198f82b4b2ef4750c7245d9f0499`

The exact prompt, answer, protocol, repository revision, and source-file hash
are embedded in `corpora/d_manifest.json`. Dataset fetching happened once on B
at pinned revisions. A3 evaluation has no network dependency.

## Tokenizer and protocol

All Q/H `input_ids` were frozen with the tokenizer distributed in
`/workspace/models/DeepSeek-V2-Lite-604d5664`. Teacher forcing always uses the
All-NPU greedy history, `temperature=0`, one requested token per position, and
top-16 plus full-vocabulary logits. The SGLang API version does not accept a
per-request `seed`; seed 0 remains protocol metadata while greedy decoding and
same-path repeat tests provide the determinism gate.
