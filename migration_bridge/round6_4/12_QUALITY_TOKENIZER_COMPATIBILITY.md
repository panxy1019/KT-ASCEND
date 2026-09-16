# Quality tokenizer compatibility investigation

State: `A3_VERIFIED` tokenizer-only and serving parity. Primary quality
outputs have **not** been produced. The approved 2pp margin is unchanged.

## First dev pilot (preserved, not a primary quality result)

`/workspace/results/r64-quality-dev-v1` completed 8 GSM8K train + 8 CMMLU dev
requests, OFF only. All reached their token limit and the original strict
parser returned invalid. Raw token IDs remain preserved. Offline decoding
showed readable text, including CMMLU option letters followed by a new question;
several GSM8K answers degenerated into numbered-heading repetition.

Do not call the 16 invalid responses a Deferred quality regression: Deferred
was OFF, the test was developmental, and a detokenizer defect was reproduced.
Do not claim fixing decoding alone repairs the underlying repetitive generation.

## Independent tokenizer reproduction

A3 Transformers `5.3.0.dev0` resolves the frozen model's
`LlamaTokenizerFast` declaration to `LlamaTokenizer`, replacing its ByteLevel
pipeline. Examples include `hello world` encoding as `[18678,322,3982]`
instead of `[31539,1843]`; Chinese text can lose most of its content.
Decoding valid raw IDs yields visible `Ġ`/`Ċ` instead of spaces/newlines.

This agrees with the documented upstream mechanism in
[Transformers issue 45488](https://github.com/huggingface/transformers/issues/45488).
The issue is independent support, not proof of our serving/model qualification.

`prepare_tokenizer_compat.py` creates a separate disposable artifact
`/workspace/results/r64-tokenizer-compat-v1`, changing only the loader class in
tokenizer_config to `PreTrainedTokenizerFast`. The generic class is named
`TokenizersBackend` in the installed Transformers version. It verifies the
model, normalizer, pre-tokenizer, post-processor, decoder and added-token tables
against the original JSON and probes English, Chinese, numeric and code strings.

| Artifact | SHA256 |
|---|---|
| Unchanged tokenizer.json | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| Original tokenizer_config.json | `31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8` |
| Overlay tokenizer_config.json | `d5f57d20f87e97e94dac06c5f91871bc819fe3fff41b2b0c96c04c2622dc95bb` |

Neither the read-only model nor host packages are edited. No global Transformers
upgrade or production-code patch is made. A test launcher explicitly selects the
hashed overlay; it remains opt-in and is recorded in resolved argv.

## BOS and protocol boundaries

The generic loader preserves tokenizer.json literally, but does **not** apply
the model config's `add_bos_token=true`. JSON-only parity must not be described
as complete restoration of all historical LlamaTokenizer behavior.
[Transformers v4 LlamaTokenizerFast source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/tokenization_llama_fast.py)
applies that flag via a post-processor. New quality draft v2 explicitly prefixes
the configured BOS ID `100000` once. Existing S6/Q/HD inputs remain unchanged.

The v2 dev protocol uses zero-shot GSM8K `Question: ... Answer: Let's think step
by step.`, normal EOS plus next-question boundaries, and conventional final
numeric extraction (explicit #### is strict when present). CMMLU retains its
zero-shot option prompt and stops at newline, accepting a single option letter.
Budgets remain GSM8K 512 / CMMLU 32. No grammar, examples, repetition penalty,
sampling change, label-based selection or truncated-answer rescue is introduced.

The same 1319/2048 primary IDs, labels and stratified selection are retained.
Draft v2 is not an executable primary contract. Train/dev validation, source
freeze and paired statistical protocol freeze still precede any primary ON run.

## Historical evidence boundary

Frozen input-ID runtime gates are not silently retokenized. A separate serving
probe requires old/new loader token trajectories to match within each mode and
checks new text ingestion against frozen-JSON IDs. Text-based historical MC
scores used the old loader and cannot be equated to scores from the corrected
loader without labeling the changed evaluation interface. Preserve historical
smoke as compatibility evidence and report the limitation explicitly.

## Serving result

`r64-tokenizer-serving-v1/probe-report.json` is
`TOKENIZER_COMPAT_SERVING_PASS`, SHA256
`8870d3ee5b6a0dfdd1bcb89439701513ef2d92d8fa95e7acd075719c5a34beba`.
Four frozen S6 prompts × old/new loader × OFF/ON = 16 captured 64-token
responses. Within each execution mode all old/new trajectories are exact;
new-loader response text equals raw frozen-JSON decoding. Four additional
text-vs-ID pairs (English/Chinese, OFF/ON) passed executable assertions;
their HTTP responses were not separately serialized, so raw-response evidence
is restricted to the 16 main parity responses. All four owned servers stopped
normally through the guard and reaped their own descendants.

This validates the loader-interface repair only. It does not establish quality
noninferiority, does not qualify batch4, and does not retroactively re-encode any
historical input IDs. The BOS+prompt+stopping draft is validated separately.
