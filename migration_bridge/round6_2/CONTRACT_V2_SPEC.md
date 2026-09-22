# Round 6.2 Contract V2 — preregistered method

Status: METHOD_PREREGISTERED_BEFORE_QD2_OUTPUT. This is not an approved
numerical contract. Corpus manifests, telemetry implementation and A3 capture
integrity are separate required prerequisites. No QD2/HD2 model output has
been generated or inspected at this registration.

## Frozen scope

DeepSeek-V2-Lite revision 604d5664dddd88a0433dbae533b7fe9472482de0;
TP1, BF16, LLAMAFILE F32 GGUF, D1 layers 7/8, producer 7, terminal 8;
same placement for defer=0 and defer=1. No D2/D3, placement changes, Graph,
Dynamic Placement, quantization changes or new backend. V1 remains rejected.
Shared NPU0 is authorized; all interference is recorded. Performance under
contention is descriptive, not an isolated speedup estimate.

## Workload and independence boundary

QD2 and HD2 each contain 128 new prompts, 64 matched-history positions per
prompt (8192 positions per corpus), frozen before QD2 execution. Both require
at least 500 positions with actual current-row or prefix Deferred exposure.
Insufficient exposure is STOP, not permission to add favorable prompts.

Domains include English prose, Chinese prose, JSON/structured, code and math,
with short/medium/longer-prefix inputs. No reuse of old QD/HD, no cross-set
prompt or template duplicates. Corpus construction and sampling provenance
must be reviewed before signing their freeze. Repeated numerical variants of
one template are NOT 128 independent prompts.

Statistical sampling unit: a prompt block containing all 64 positions. The
binomial model assumes independent, exchangeable prompt blocks in the declared
workload population. Within-prompt positions and candidate pairs need not be
independent. Grouping alone does not establish the between-prompt assumption.
If the corpus design cannot justify that assumption, mark
STATISTICAL_DESIGN_NOT_QUALIFIED and stop before claiming held-out confidence.
Confidence statements do not establish guarantees for arbitrary user prompts.

## Baseline history and capture identity

Generate defer=0 greedy history once, freeze its bytes and SHA, then replay
identical input token IDs/history for both modes. Greedy temperature 0,
ignore_eos true, 64 positions. No token reduction or early position exclusion.
Every record binds request ID, prompt hash, full request input hash, history
hash, forward ID, target index, logits shape and output bytes. Never infer
association from a legacy Pass filename. Capture failure invalidates the run.
API-selected tokens must agree with captured maxima; tied maxima must preserve
the serving token policy rather than an invented tie breaker.

## Decision metrics and hard gates

K=32. U is baseline top32 plus the serving baseline top1 (include it explicitly
in case of ties). Sort equal-logit candidates by ascending token ID for
reproducible reporting; do not change actual serving selection.

For baseline anchor a and i in U:

    m0_i = z0[a] - z0[i]
    mD_i = zD[a] - zD[i]
    d_i = mD_i - m0_i
    c_i = max(0, -d_i)
    b_i = max(0, d_i)
    C_t = max_i c_i

Compute metrics in float64 from identical captured float logits. Stable means
all non-anchor baseline margins exceed R_D strictly. The ambiguity set is
{i in U: m0_i <= R_D}, including a. Require:

- stable top1 exact at every position;
- ambiguous top1 belongs to the ambiguity set at every position;
- Deferred top1 outside U: zero;
- for every ordered candidate pair (i,j) in U with z0[i]-z0[j] > R_D,
  zD[i]-zD[j] must remain strictly positive (empirical additional hard gate,
  not a mathematical consequence of the anchor-only contraction bound);
- finite full logits, metrics and routing weights;
- when no Deferred event occurred anywhere in the evaluated request prefix,
  including the current row, full logits must be byte-identical;
- exact runtime semantics, same-process/restart determinism, pending drain,
  terminal no-defer, exactly-once and whole-request no-hit regression.

Ambiguity cardinality and beneficial expansion are diagnostic, not arbitrary
cardinality failures. No decision hard failure is excused by a statistical pass.

## Calibration and statistical gates

Three separate risks at each position: C_t, full-logit max absolute difference,
and full-logit relative L2 (denominator max(norm(z0), 1e-12)). Cosine and layer
amplification are descriptive; zero-norm cases must be explicit, never NaN.
For each metric, take the maximum over the 64 positions of each prompt.

QD2 calibration: one-sided order-statistic tolerance limit, target coverage
0.95 of prompt-block maxima, family confidence at least 0.95 across the three
metrics. Use Bonferroni alpha=0.05/3 per metric. For n=128, select the smallest
rank k satisfying P[Binomial(n,0.95) <= k-1] >= 1-alpha: k=127.
Use the observed kth value without multiplier, interpolation or clipping.
R_D is the threshold for C_t. The other two thresholds are global-sanity
reserves. Neither old failure margins nor HD2 outputs enter derivation.

This is a 5% **prompt-block tail-risk budget**, NOT permission for 5% token
errors, and NOT a 99.5%/99.9% certification. The latter would need materially
more independent prompt blocks. Any claim must retain this distinction.

HD2: count prompt blocks whose maximum strictly exceeds each frozen reserve.
Compute one-sided Clopper-Pearson upper bound at confidence 1-0.05/3.
All three upper bounds must be <=0.05. Record position exceedance counts too,
but do not use correlated positions as the binomial N. Some nonzero tail
exceedances can pass; decision hard gates still require zero failures.
Use `tools/tail_statistics.py`; freeze its SHA with the candidate contract.

QD2 decision hard gates must pass before candidate freeze. HD2 is one-shot
after candidate commit. Missing output, runtime failure or capture corruption
is not a numerical pass. Infrastructure-only retry requires an explicit incident
record with unchanged inputs/rules; numerical failure forbids refitting on HD2.
Any revised hypothesis needs new held-out data (HD3).

## Quality, free generation and performance

Only after numerical held-out pass: frozen C-Eval128 and GSM8K-MC128, same
evaluator and manifest c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce.
Report paired correctness and 20000 paired bootstrap replicates (seed 6202),
95% percentile CI. A wholly negative delta CI is QUALITY_REGRESSION; otherwise
report NO_DETECTED_SIGNIFICANT_REGRESSION, not proof of quality equivalence.
No invalid outputs, NaN or parser failures are accepted. A formal noninferiority
claim is not made without a separate justified effect-size margin.

Free generation uses the separately frozen FD workload, 64 tokens, no prompt
selection after outputs. A stable first divergence or ambiguity escape is a
hard failure. Runtime traces, RSS/HBM, TTFT, latency p50/p95, decode throughput,
queue/worker timing and overlap are recorded after correctness/quality gates.
Shared-NPU telemetry is explicitly labeled; no isolated performance claim.

## Stop and freeze discipline

All rules above are fixed before QD2. Corpus, telemetry, executable hash and
execution deadlines must be frozen before their corresponding model runs.
No placeholder numerical values may appear as an approved candidate.
Any insufficiency, hard-gate failure or statistical failure keeps D1 blocked.
Only the final ADR can approve V2; D2/D3 require a separate round.
