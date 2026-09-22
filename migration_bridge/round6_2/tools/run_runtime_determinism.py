"""D1 production-overlap same-process/restart regression on frozen S6.

No complete-forward observer is required. Every API response is retained;
count, finiteness, repeat equality and prefix equality are fail-closed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request
from verify_reserved_inputs import checked
from freeze_corpora import canonical_sha


def validate(answer, request_id, token_count):
    if answer["meta_info"]["id"] != request_id:
        raise ValueError("RUNTIME_REQUEST_ID_MISMATCH")
    tokens = answer["output_ids"]
    if len(tokens) != token_count or any(type(t) is not int or not 0 <= t < 102400 for t in tokens):
        raise ValueError("RUNTIME_TOKEN_COUNT_OR_ID_INVALID")
    probabilities = answer["meta_info"].get("output_token_logprobs", [])
    if len(probabilities) != token_count or any(row[0] is None or not math.isfinite(float(row[0])) for row in probabilities):
        raise ValueError("RUNTIME_MISSING_OR_NONFINITE_LOGPROBS")
    if [row[1] for row in probabilities] != tokens:
        raise ValueError("RUNTIME_LOGPROB_TOKEN_MISMATCH")
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart-reference", type=Path)
    p.add_argument("--base-url", default="http://127.0.0.1:31000")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse runtime output overwrite")
    corpus, candidate = checked(args.corpus), checked(args.candidate)
    if corpus["corpus"] != "S6" or len(corpus["prompts"]) != 12:
        raise ValueError("FROZEN_RUNTIME_CORPUS_MISMATCH")
    if candidate["version"] != "round6.2-v2-candidate":
        raise ValueError("CANDIDATE_NOT_FROZEN")
    references = None
    if args.restart_reference:
        previous = checked(args.restart_reference)
        if previous["status"] != "PASS" or previous["candidate_sha256"] != candidate["sha256"] or previous["corpus_sha256"] != corpus["sha256"]:
            raise ValueError("RUNTIME_RESTART_REFERENCE_MISMATCH")
        references = {row["prompt_id"]: row["reference64"] for row in previous["rows"]}
    args.output.mkdir(parents=True)
    rows, request_count = [], 0
    with (args.output / "requests.jsonl").open("x") as stream:
        for prompt in corpus["prompts"]:
            baseline = references[prompt["id"]] if references else None
            runs, short = [], {}
            budgets = [64] if references else [64] * 10 + [8, 16, 32]
            for index, budget in enumerate(budgets):
                rid = f'r62-runtime-{"restart" if references else "repeat"}-{prompt["id"]}-{index}'
                payload = dict(rid=rid, input_ids=prompt["input_ids"],
                               sampling_params=dict(temperature=0, max_new_tokens=budget, ignore_eos=True),
                               return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                request = urllib.request.Request(args.base_url + "/generate", data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"}, method="POST")
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=600) as response:
                    answer = json.load(response)
                stream.write(json.dumps(dict(payload=payload, response=answer,
                                             elapsed_seconds=time.perf_counter() - started)) + "\n")
                stream.flush()
                tokens = validate(answer, rid, budget)
                request_count += 1
                if baseline is None:
                    baseline = tokens
                if tokens != baseline[:budget]:
                    failure = dict(status="FAIL", classification="CLEAN_RESTART_MISMATCH" if references else "SAME_PATH_NONDETERMINISM",
                                   prompt_id=prompt["id"], request_id=rid, token_budget=budget,
                                   expected=baseline[:budget], actual=tokens)
                    (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                    raise ValueError(failure["classification"])
                if budget == 64:
                    runs.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                else:
                    short[str(budget)] = tokens
                print(json.dumps(dict(prompt_id=prompt["id"], completed_requests=request_count, budget=budget, exact=True)), flush=True)
            rows.append(dict(prompt_id=prompt["id"], reference64=baseline, hashes64=runs, short_outputs=short))
    report = dict(status="PASS", kind="clean_restart" if references else "same_process",
                  candidate_sha256=candidate["sha256"], corpus_sha256=corpus["sha256"],
                  request_count=request_count, all_finite=True, all_exact=True, rows=rows)
    report["sha256"] = canonical_sha(report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
