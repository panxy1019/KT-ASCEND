# All-NPU quality limitation, before Hybrid quality completion

The complete frozen All-NPU run returned GSM8K3/32,18/32 truncated,0 invalid;
CMMLU145/256,8 invalid,0 truncated. Its process/lifecycle completed normally.
This note does not alter the running frozen comparison or select replacement
questions. Hybrid answers must still be collected under exactly the same protocol.

## Primary-source comparison

The [DeepSeek-V2 paper, section3.2 and AppendixG](https://arxiv.org/html/2405.04434v5)
reports an8-shot GSM8K generation protocol and illustrates worked Q/A exemplars.
Its multiple-choice evaluations use perplexity-based scoring;CMMLU uses5shots.
The paper distinguishes the pretrained Lite base model from its SFT chat variant.
The [official model repository](https://github.com/deepseek-ai/DeepSeek-V2) reports
Lite base GSM8K41.1 and directs readers to the paper for evaluation details.

Our frozen screen is not a reproduction of that protocol: GSM8K uses zero worked
examples and `Answer: Let's think step by step.`,max512;CMMLU is generated-letter
scoring. Therefore the official score cannot be used as a numerical equality
target for this32-item screen. A protocol effect is a plausible hypothesis for
repetitive All-NPU answers,not a demonstrated root cause or proof of a serving bug.

## Required interpretation

- Only3 correct baseline GSM8K items provide weak regression discrimination.
- Equal low scores,if observed,cannot prove broad reasoning quality equivalence.
-18truncations must remain errors under the current frozen rule.
- Do not extend budgets,add examples,rescore,or replace samples in the running
  comparison and then call it the same test.
- First finish and audit the paired screen. If its result or baseline limitations
  prevent meaningful qualification, a separately versioned base-model evaluation
  study must freeze its protocol and independent inputs before Hybrid outputs.
  Worked examples must come from train/dev,not the final evaluation questions.

No new threshold,quality rule,model checkpoint or runtime patch is introduced by
this source review. The unchanged32+256 experiment remains the source of its own
result,whether passing or failing its predeclared engineering screen.
