"""Prequalification only: actual one-prefill/continuous-decode requests.

Reads an append-only capture index instead of rescanning the tensor tree per
step. Retains full captures in this initial observer. Not a QD3 evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round6_2/tools"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:31000")
    args = p.parse_args()
    import torch
    from audit_deferred_transitions import validate_transition
    torch.set_num_threads(1)
    frozen = json.loads(args.requests.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.index.read_text().strip():
        raise ValueError("R63_NONEMPTY_CAPTURE_INDEX")
    count, rows = 0, []
    started = time.time()
    with args.index.open() as index:
        for request in frozen["requests"]:
            rid, prompt = request["rid"], request["input_ids"]
            budget = len(request["baseline_output_ids"]) if args.trace else frozen["token_budget"]
            payload = dict(rid=rid, input_ids=prompt, sampling_params=dict(
                temperature=0, max_new_tokens=budget, ignore_eos=True),
                return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
            begin = time.monotonic()
            http = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(http, timeout=600) as response:
                answer = json.load(response)
            with (args.output / f"{rid}.response.json").open("x") as out:
                json.dump(dict(request=payload, response=answer), out)
            tokens = answer["output_ids"]
            if answer["meta_info"]["id"] != rid or len(tokens) != budget:
                raise ValueError("R63_HTTP_IDENTITY_OR_BUDGET")
            if args.trace and tokens != request["baseline_output_ids"]:
                raise ValueError("R63_SCHEDULER_DID_NOT_RETURN_FORCED_TOKENS")
            steps, vectors, exposed = [], [], False
            for step in range(budget + 1):
                deadline = time.monotonic() + 30
                while True:
                    offset = index.tell()
                    line = index.readline()
                    if line.endswith("\n"):
                        break
                    index.seek(offset)
                    if time.monotonic() > deadline:
                        raise TimeoutError("R63_INDEX_COMMIT")
                    time.sleep(.05)
                entry = json.loads(line)
                if entry["sequence"] != count or entry["batch"]["request_ids"] != [rid]:
                    raise ValueError("R63_INDEX_SEQUENCE_OR_RID")
                count += 1
                data = torch.load(entry["file"], map_location="cpu", weights_only=False, mmap=True)
                tx = data["capture.transaction"]
                if tx["pid"] != entry["pid"] or tx["forward_sequence_id"] != entry["sequence"]:
                    raise ValueError("R63_INDEX_CAPTURE_JOIN")
                for key, sha in tx["tensor_byte_sha256"].items():
                    value = data[key].contiguous()
                    if hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() != sha:
                        raise ValueError("R63_CAPTURE_HASH")
                expected = prompt if step == 0 else [tokens[step-1]]
                positions = list(range(len(prompt))) if step == 0 else [len(prompt) + step - 1]
                if (data["capture.forward_batch_info.input_ids"].tolist() != expected or
                    data["capture.forward_batch_info.positions"].tolist() != positions or
                    data["capture.forward_batch_info.seq_lens"].tolist() != [len(prompt) + step]):
                    raise ValueError("R63_KV_HISTORY_OR_POSITION")
                logits = data["capture.next_token_logits"]
                if tuple(logits.shape) != (1, 102400) or not torch.isfinite(logits).all():
                    raise ValueError("R63_LOGITS_SHAPE_OR_FINITE")
                if not args.trace and step < budget and logits[0, tokens[step]] != logits.max():
                    raise ValueError("R63_NATURAL_TOKEN_NOT_MAX")
                events = 0
                for layer, owned in ((7, (16,38,41,50)), (8, (8,25,33,55))):
                    ids = data[f"r62.ktep.layer{layer}.topk_ids"]
                    weights = data[f"r62.ktep.layer{layer}.routing_weights"]
                    immediate = data[f"r62.cpu.layer{layer}.immediate_ids"]
                    delayed = data.get(f"r62.cpu.layer{layer}.deferred_ids")
                    if ids.shape != (len(expected), 6) or not torch.isfinite(weights).all():
                        raise ValueError("R63_ROUTING_SHAPE_OR_FINITE")
                    for expert in owned:
                        selected = ids == expert
                        d = torch.zeros_like(selected) if delayed is None else delayed == expert
                        if not torch.all(((immediate == expert) ^ d)[selected]):
                            raise ValueError("R63_ROUTE_EXACT_ONCE")
                        if layer == 8 and d.any():
                            raise ValueError("R63_TERMINAL_DEFER")
                        events += int((selected & d).sum())
                validate_transition(tx["deferred_events"], events)
                exposed |= events > 0
                steps.append(dict(step=step, file=entry["file"], sequence=entry["sequence"],
                    pid=entry["pid"], target=step < budget, current_deferred=events,
                    cumulative_exposure=exposed, logits_sha256=tx["tensor_byte_sha256"]["capture.next_token_logits"]))
                if step < budget:
                    vectors.append(logits.clone())
            torch.save(torch.cat(vectors), args.output / f"{rid}.logits.pt")
            rows.append(dict(rid=rid, input_ids=prompt, baseline_output_ids=tokens,
                steps=steps, seconds=time.monotonic()-begin))
            print(json.dumps(dict(rid=rid, targets=budget, forwards=len(steps),
                exposed_targets=sum(s["cumulative_exposure"] for s in steps if s["target"]),
                seconds=rows[-1]["seconds"])), flush=True)
        if index.read().strip():
            raise ValueError("R63_EXTRA_INDEX_ENTRIES")
    if args.trace:
        traced = [json.loads(line) for line in args.trace.read_text().splitlines()]
        all_steps = [(row, s) for row in rows for s in row["steps"]]
        if len(traced) != len(all_steps):
            raise ValueError("R63_TRACE_COUNT")
        for record, (row, step) in zip(traced, all_steps):
            if (record["rid"] != row["rid"] or record["step"] != step["step"] or
                record["sequence"] != step["sequence"] or record["pid"] != step["pid"] or
                record["sampled_logits_sha256"] != step["logits_sha256"]):
                raise ValueError("R63_SAMPLE_CAPTURE_JOIN")
    result = dict(status="PREQUALIFICATION_CAPTURE_PASS", forced=bool(args.trace),
        started_unix=started, ended_unix=time.time(), complete_forwards=count, rows=rows)
    with (args.output / "result.json").open("x") as out:
        json.dump(result, out, indent=2)


if __name__ == "__main__":
    main()
