# Round6.4 quality protocol preparation — not quality results

Status: **PROTOCOL_PREPARATION / LOCAL_VERIFIED**. No primary model ON outputs
have been generated or inspected. The approved margin remains absolute2pp per
primary benchmark. Dataset/parser/capacity preparation is not a quality PASS.

## Sources and exact input draft

Downloaded with installed `hf`1.28.0, pinned revisions, without remote dataset
script execution. A3 evaluation will read offline JSON input IDs.

| Dataset | Revision | Primary sample |
|---|---|---|
| openai/gsm8k | 740312add88f781978c0658806c59bc2815b9866 | main/test all1319 |
| lmlmcat/cmmlu | efcc940752ea4a1ea94d2727f11f83858d64fc8e | 2048 of11582;67subjects |

The requested historical CMMLU Hub name `haonan-li/cmmlu` redirects to
`lmlmcat/cmmlu`; this is recorded, not silently treated as a different corpus.
GSM8K source is MIT. CMMLU card metadata says BY-NC-4.0 and its license paragraph
links BY-NC-SA-4.0. Retain both notices, attribution and noncommercial restrictions;
do not infer commercial deployment rights from this research evaluation.

References: [GSM8K original repository](https://github.com/openai/grade-school-math),
[CMMLU original repository](https://github.com/haonan-li/CMMLU),
[pinned CMMLU card](https://huggingface.co/datasets/lmlmcat/cmmlu/blob/efcc940752ea4a1ea94d2727f11f83858d64fc8e/README.md).

Input preparation uses Hamilton proportional subject quotas, sorted subject
names, Python Random(6404) sampling without replacement, sorted selected row
indices. No score-driven selection. Generator `tools/prepare_quality_inputs.py`.

Local sources: `/home/admin/Documents/ChatGPT/KT/r64-quality-sources`.
Local input draft: `/home/admin/Documents/ChatGPT/KT/r64-quality-draft-v1`.

- Primary JSON SHA: `7d501ab27d76ec19ff64aebf1fc5d5c0e70ee64aa829faf99261ec5c111b3d4d`.
- Dev JSON SHA: `77b9fd2d80c2f5882f9710c00c644628b2995f00bc629ff758c03521e4fe1769`.
- Tokenizer SHA: `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`.
- GSM8K rawtest SHA: `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- CMMLU ZIP SHA: `22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6`.

These are drafts pending dev-only model/template validation and capacity smoke;
`primary_execution_allowed=false` remains enforced until formal freeze.

## Overlap, not sample deletion

`evidence/quality-preparation-v1/overlap-audit.json` inventories normalized literal
question equality/containment across historical corpus JSON, including authoring
drafts conservatively. It is not proof of semantic novelty or absence in training.

- GSM8K128/1319 questions overlap the historical MC benchmark. Retain and report.
- CMMLU2/2048 overlap historical C-Eval questions. Retain and report.
- Another2 CMMLU test questions overlap dev. Do not use those matching dev
  questions to tune the template/parser; retain primary test rows unchanged.
- Final quality table must include overlapping/unobserved subsets separately.

## Parser and capacity

0-shot templates stored with exact input IDs; GSM8K asks for reasoning followed
by a final `#### <number>` line. CMMLU asks only for A/B/C/D. Parser accepts no
generated code and never evaluates expressions. Truncated outputs count wrong
even if they happen to contain a parseable answer. Invalid and truncated counts
remain explicit. All3367 gold answers parse;3 synthetic parser tests PASS.

Maximum input lengths: GSM223,CMMLU334. Required input+output maximum735.
Old context512 is insufficient. Test-only `capacity_python_adapter.py` changes
only context1024,total4096,prefill/chunk1024,and explicit batch1/2/4 cap. The
original launcher and all production code remain unchanged. The adapter records
resolved argv and rejects unexpected/duplicate original options;2tests PASS.
Equal OFF/ON capacity and a dev-only smoke are required before quality.

## Paired interval implementation candidate

Let G count OFFwrong/ONcorrect and L count OFFcorrect/ONwrong, n paired items.
Delta=(G-L)/n. `tools/paired_quality_interval.py` constructs a conservative exact
bound under the IID paired-trinomial benchmark model:

1. Exact binomial marginal lower bound on gain probability, alpha=.0125.
2. Exact binomial marginal upper bound on loss probability, alpha=.0125.
3. LowerDelta=lowerGain-upperLoss. Union bound gives one-sided coverage>=.975.

Gain/loss are mutually exclusive, not independent arms; union bound does not
assume their independence. Across the two primary endpoints, the pre-approved
Bonferroni family error budget is .05. Both lowerDelta values must exceed-.02.
An upper bound is reported for classifying clear regression. Crossing the
boundary is INCONCLUSIVE, never automatically PASS/FAIL or permission to extend n.

This is a deliberately conservative exact-method candidate permitted by the
approved plan, not an additional point-accuracy threshold. Five tests include
finite-n outcome enumeration over a probability grid and zero-discordance
closed-form checks. Reference for the marginal exact method:
[SciPy binomial confidence-interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

Important statistical limitation: coverage is exact for the stated paired
sampling model, not an assertion about arbitrary non-IID future tasks. CMMLU
uses fixed subject quotas; report strata and stratified bootstrap sensitivity
and resolve the sampling-model interpretation before final protocol freeze.
No method selection may depend on primary ON outcomes.

Paired bootstrap20000/seed6404 remains supplemental. Zero discordance yields
bootstrap[0,0] but a nonzero exact uncertainty bound (approximately-.00332 at
n1319), so no fabricated certainty.

Prospective simulation `prospective-power.json` is assumptions only. At true
Delta0, assumeddiscordance5%, this conservative candidate passes approximately
49.7% atn1319 and79.9% atn2048. This warns that approved sample sizes need not
prove2pp noninferiority; it does not authorize changing margin or picking easier
examples. No model results entered these simulations.

Local analysis runtime: numpy2.5.3,scipy1.18.1,pyarrow25.0.1,tokenizers0.23.2.
Before formal freeze, record complete evaluator hashes and runtime dependency
versions, validate dev-only behavior, then freeze the machine-readable protocol.
