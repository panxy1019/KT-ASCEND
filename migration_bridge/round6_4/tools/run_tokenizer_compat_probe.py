"""Same frozen input IDs: old/new tokenizer loader must not change generation."""
import argparse
import json
from pathlib import Path
import signal
import time
import urllib.request
from tokenizers import Tokenizer
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, PROMPTS, validate


def generate(payload):
    req = urllib.request.Request("http://127.0.0.1:31000/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    corpus = checked(args.corpus)
    assert corpus["sha256"] == S6_SHA
    prompts = {r["id"]:r for r in corpus["prompts"]}
    raw = Tokenizer.from_file("/workspace/results/r64-tokenizer-compat-v1/tokenizer.json")
    args.output.mkdir(exist_ok=False)
    report = dict(status="RUNNING", rows=[], started_unix=time.time())
    def cancel(signum, frame):
        raise InterruptedError("OWNED_TOKENIZER_PROBE_CANCELLED")
    signal.signal(signal.SIGTERM, cancel)
    try:
        for profile in ("D1_NODEFER", "D1"):
            refs = {}
            for compat in (False, True):
                root = args.output / f"{profile}-compat{int(compat)}"
                with OwnedServer(root, profile, tokenizer_compat=compat):
                    with (root/"requests.jsonl").open("x") as log:
                        for pid in PROMPTS:
                            rid = f"r64-tokenizer-{profile}-{int(compat)}-{pid}"
                            payload = dict(rid=rid, input_ids=prompts[pid]["input_ids"], sampling_params=dict(
                                temperature=0,max_new_tokens=64,ignore_eos=True), return_logprob=True,
                                logprob_start_len=-1, top_logprobs_num=0)
                            answer = generate(payload)
                            log.write(json.dumps(dict(payload=payload,response=answer))+"\n")
                            log.flush()
                            tokens = validate(answer,rid,64)
                            if compat:
                                assert tokens == refs[pid], "TOKENIZER_LOADER_CHANGED_FIXED_ID_TRAJECTORY"
                                assert answer["text"] == raw.decode(tokens), "SERVING_DECODE_DIFFERS_FROM_FROZEN_JSON"
                            else:
                                refs[pid] = tokens
                            report["rows"].append(dict(profile=profile, compat=compat, prompt=pid, verified=True))
                            print(json.dumps(report["rows"][-1]), flush=True)
                        if compat:
                            # Exercise actual text ingestion, independently of fixed-ID generation.
                            for n,text in enumerate(("hello world", "中文 空格与换行\n答案：A")):
                                common = dict(sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True),
                                              return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                                via_text = generate(dict(common,rid=f"r64-text-{profile}-{n}",text=text))
                                via_ids = generate(dict(common,rid=f"r64-ids-{profile}-{n}",input_ids=raw.encode(text).ids))
                                assert via_text["meta_info"]["prompt_tokens"] == len(raw.encode(text).ids)
                                assert via_text["output_ids"] == via_ids["output_ids"], "TEXT_INPUT_ID_PARITY"
        report["status"] = "TOKENIZER_COMPAT_SERVING_PASS"
    except BaseException as exc:
        report.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output/"probe-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
