# Primary quality protocol v2 — developmental review, not yet frozen

Authority: approved zero-shot primary benchmark scope and 2pp absolute margin.
No primary OFF or ON outputs have been observed. This document is not permission
for a runner to bypass `FROZEN_BEFORE_PRIMARY_OUTPUT` validation.

## Corpus unchanged, serialization versioned

- GSM8K official test: all1319, native free-response, max512 generated tokens.
- CMMLU: same2048 selected IDs/67subjects/seed6404 Hamilton allocation, max32.
- Model, weights, placement, expert selection, greedy temperature0, batch1 unchanged.
- Same original tokenizer.json SHA; explicit model-config BOS100000 prefix.
- New primary bytes SHA: `779d1318137ee1a3573bafe20e265f2aa31170660c08b49ada0507fdb179a3ba`.
- New dev bytes SHA: `f125f737c890450dd7fd3d00ecdb6cbb429fd6cc2866757489b01d6012c56ac2`.
- New inputmax203/335; total required contextmax715, within tested1024capacity.

Reproduce using `revise_quality_draft.py`, the v1 draft, pinned sources and the
unchanged tokenizer config. The v2 manifest is in fixtures; exact primary/dev
bytes are retained in the local project artifact directory and A3 results.

## Evaluation semantics

The tokenizer compatibility overlay is explicit, opt-in and serving-parity
verified. Input IDs are supplied directly and contain exactly one configured
BOS. Stop on ordinary EOS or the predeclared task boundaries. No repetition
penalty, few-shot examples, grammar, adaptive budget or label leakage.

GSM8K uses `Question: ...\nAnswer: Let's think step by step.` and stops at a
new `Question:`/`Q:`. The answer extractor uses a strict #### suffix when the
model emits one; otherwise the final numeric substring. This is a **simple
zero-shot answer extraction metric**, not semantic grading of the explanation.
For example, an explanation that gives a correct dollar amount and then
restates a different number of minutes can be counted wrong. Neither gold
answers nor test results are used to select which number to extract.

CMMLU retains its zero-shot Chinese prompt, stops on newline, and accepts one
option letter (optional final period). Full outputs, finish reasons, parsed
answers, invalid counts and truncation counts are retained. Any length-truncated
answer is wrong even if it contains the correct numeric value earlier.

## Statistical proposal retained

Paired gain/loss marginal Clopper-Pearson bounds with within-endpoint Bonferroni,
one-sided97.5% per endpoint. Across the two endpoints, **family alpha=.05**;
the **quality margin=.02** is a distinct, approved product tolerance.
The 20,000 paired bootstraps are supplemental and not the zero-discordance proof.

The exact-bound statement is conditional on the paired trinomial/binomial
working model. It is **not** an exact finite-population survey interval for
CMMLU's fixed subject allocation. Report the stratified bootstrap and each
subject's counts separately; the benchmark claims apply to this fixed protocol,
not arbitrary production domains. These assumptions and prospective power
limitations must remain visible in the final frozen contract and ADR.

Lower bound >−.02 on both benchmarks permits numerical noninferiority acceptance,
subject to the independent runtime/invalid-output review. Interval crossing is
INCONCLUSIVE, not a reason to add easier examples or adjust the margin.

## Required next evidence

1. Complete v2 OFF-only8+8 development pilot and examine stopping/text/parser.
2. Complete OFF-only32train+32dev under the identical v2 protocol.
3. Freeze evaluator files, corpus bytes, actual source hashes, environment and
   the stated statistical method before any primary ON output.
4. Run complete paired3367 OFF/ON, preserve every row including errors.

No quality pass is asserted here. The v1 interface failure remains archived.

## v2 small development pilot completed

`r64-quality-dev-v2` completed OFF-only8+8, no request/system failure.
GSM8K:1/8correct,1invalid,4length-truncated;354.878s total generation latency.
CMMLU:5/8correct,0invalid,0truncated; all eight responses stopped after two
generated tokens (option plus newline);4.442s total latency.

This is enough to exercise the functioning evaluation interface, not evidence
of model or Deferred quality. In particular GSM8K's weak zero-shot performance
and repetitive continuation are now visible as baseline limitations. Do not
rescue truncated answers, alter the parser after test results, or hide this
limitation behind a paired delta. The unchanged32+32development validation is
running as `r64-quality-dev32-v2` before the final freeze decision.
