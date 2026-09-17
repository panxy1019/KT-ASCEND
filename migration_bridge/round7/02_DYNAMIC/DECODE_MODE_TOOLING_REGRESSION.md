# Future Dynamic decode-mode tooling repair

Date: 2026-09-17. Status: **LOCAL_VERIFIED** repair; deployed mode definition
**CODE_INSPECTED**. Dynamic full-model A3 qualification has not started.

## Symptom and cause

Prepared `dynamic_route_summary.py` filtered rows using the substring `DECODE`
in `str(batch.forward_mode)`. The actual A3 Python3.11 enum is an `IntEnum`:
EXTEND stringifies to `"1"`, DECODE to `"2"`. The prepared filter would therefore
miss every real decode row and fail with `NO_ATTRIBUTED_DECODE_FORWARD`.
This is a test/analysis classification defect, not a production routing or
arithmetic defect and not a failure of the currently running Static campaign.

The pinned official SGLang defines EXTEND and DECODE with `auto()` in
`ForwardMode(IntEnum)` and provides `is_decode()`. Python3.11 documents that
`IntEnum.__str__` uses `int.__str__`.
[Pinned SGLang mode definition](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/model_executor/forward_batch_info.py),
[Python3.11 IntEnum documentation](https://docs.python.org/3.11/library/enum.html#enum.IntEnum).

Actual deployed file:
`/workspace/kt-src/third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py`.
SHA256: `223c3119001df570eb13b8532f3337aaf4cd8ed2617b2a796ff9149c9563847d`.
Lines74/77/79/127 contain the class, EXTEND, DECODE and semantic predicate.

## Minimal repair

1. Exact token-history association now returns prefill and useful decode indices
   in addition to the unchanged included/unused-tail attribution.
2. Dynamic HOLDOUT passes independently proven decode keys to the summary.
   Unknown numeric mode is never guessed or hardcoded to value2.
3. Future debug observer also records enum `.name` and `.is_decode()`. If present,
   semantic annotations must agree with independent history classification.
4. Missing/duplicate/layer-incomplete decode forwards fail explicitly. Static
   offline summaries omit Dynamic counterfactual fields rather than imply an
   initial-mask counterfactual was measured.

## Regression and deployment boundary

Local tests cover actual numeric mode with history, numeric mode without proof,
semantic boolean annotation, contradictory annotation, missing expected forward,
and absence of false Dynamic counterfactual claims in Static analysis. Additional
prefill tests cover exact-key filtering, duplicate/missing layers and malformed
route shape. The complete local pure-Python suite is **65/65 PASS**;
`git diff --check` also passes. Local fixtures do not establish A3 qualification.

Lossless re-analysis of the two closed Static archives proves stage attribution
against actual captured numeric metadata; see the prefill/decode checkpoint.
No new Dynamic model run has been claimed.

The healthy Static driver continues on immutable release-v6. Native extension,
actual deployed wrapper, frequency mask and acceptance policy are unchanged.
Deploy this repair only into a new versioned tool release before future Dynamic
probes; never replace active v6 or older staged v9 in place.
