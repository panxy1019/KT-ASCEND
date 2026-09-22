"""Frozen FD free generation with logits from the actual generating forwards.

Does not substitute a full-prefix prefill replay for an original decode step.
At first divergence the two free-running histories are still identical.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from verify_reserved_inputs import checked
from run_runtime_determinism import validate
from freeze_corpora import canonical_sha


def expected_inputs(prompt_ids, output_ids, step):
    if step == 0:
        return prompt_ids, list(range(len(prompt_ids)))
    return [output_ids[step - 1]], [len(prompt_ids) + step - 1]


def inspect_capture(path, prompt, tokens, step, rid, metadata, mode):
    import torch
    from assemble_capture import byte_sha
    from audit_deferred_transitions import validate_transition
    data = torch.load(path, map_location="cpu", weights_only=False)
    tx = data["capture.transaction"]
    if tx["batch"]["request_ids"] != [rid] or tx["provenance"] != metadata:
        raise ValueError("FD_CAPTURE_IDENTITY_MISMATCH")
    for name, sha in tx["tensor_byte_sha256"].items():
        if byte_sha(data[name]) != sha:
            raise ValueError("FD_CAPTURE_HASH_MISMATCH")
    expected_ids, expected_positions = expected_inputs(prompt["input_ids"], tokens, step)
    ids = data["capture.forward_batch_info.input_ids"].reshape(-1).tolist()
    positions = data["capture.forward_batch_info.positions"].reshape(-1).tolist()
    seq_lens = data["capture.forward_batch_info.seq_lens"].reshape(-1).tolist()
    if ids != expected_ids or positions != expected_positions or seq_lens != [len(prompt["input_ids"]) + step]:
        raise ValueError("FD_ACTUAL_FORWARD_HISTORY_MISMATCH")
    logits = data["capture.next_token_logits"][-1].contiguous()
    if logits.shape != (102400,) or not torch.isfinite(logits).all():
        raise ValueError("FD_INVALID_LOGITS")
    if step < len(tokens) and logits[tokens[step]] != logits.max():
        raise ValueError("FD_SERVING_TOKEN_NOT_ACTUAL_CAPTURE_MAXIMUM")
    routes = []
    cpu_hits = 0
    for layer, owned in ((7, {16, 38, 41, 50}), (8, {8, 25, 33, 55})):
        top = data[f"r62.model.layer{layer}.router_topk_ids"]
        immediate = data[f"r62.cpu.layer{layer}.immediate_ids"]
        deferred = data.get(f"r62.cpu.layer{layer}.deferred_ids")
        weights = data[f"r62.cpu.layer{layer}.routing_weights"]
        if top.shape != (len(ids), 6) or immediate.shape != top.shape or weights.shape != top.shape:
            raise ValueError("FD_ROUTE_SHAPE_MISMATCH")
        if deferred is not None and deferred.shape != top.shape:
            raise ValueError("FD_DEFERRED_SHAPE_MISMATCH")
        if not torch.isfinite(weights).all():
            raise ValueError("FD_NONFINITE_ROUTING_WEIGHT")
        for row in range(len(ids)):
            for rank in range(6):
                expert = int(top[row, rank])
                if expert not in owned:
                    continue
                cpu_hits += 1
                delayed = deferred is not None and int(deferred[row, rank]) == expert
                if delayed == (int(immediate[row, rank]) == expert):
                    raise ValueError("FD_OWNERSHIP_NOT_EXACT_ONCE")
                if delayed:
                    if mode != "defer1" or layer != 7:
                        raise ValueError("FD_ILLEGAL_DEFERRED_ROUTE")
                    routes.append(dict(producer=7, consumer=8, expert=expert, rank=rank,
                                       position=positions[row], route_weight=float(weights[row, rank])))
    validate_transition(tx["deferred_events"], len(routes))
    return dict(step=step, capture_file=str(path), forward_id=[tx["pid"], tx["forward_sequence_id"]],
                actual_forward_input_ids=ids, actual_positions=positions,
                full_history_sha256=canonical_sha(prompt["input_ids"] + tokens[:step]),
                logits_sha256=byte_sha(logits), cpu_hits=cpu_hits, deferred_routes=routes), logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--mode", choices=["defer0", "defer1"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:31000")
    args = p.parse_args()
    import torch
    if args.output.exists():
        raise ValueError("refuse FD overwrite")
    corpus, candidate = checked(args.corpus), checked(args.candidate)
    if corpus["sha256"] != "197f8a9a4bca0349a2984d603e71c1722cea1828fdbff30be00fe7441a4aa1a6":
        raise ValueError("FD_CORPUS_DRIFT")
    if json.loads(args.quality_report.read_text())["status"] != "PASS":
        raise ValueError("QUALITY_NOT_PASSED")
    metadata = json.loads(args.provenance.read_text())
    if metadata.get("mode") != args.mode:
        raise ValueError("FD_PROVENANCE_MODE_MISMATCH")
    if list(args.captures.rglob("Pass*.pt")):
        raise ValueError("FD_REQUIRES_FRESH_CAPTURE_SERVER")
    args.output.mkdir(parents=True)
    rows, sequences = [], []
    for prompt in corpus["prompts"]:
        rid = f'r62-fd-{args.mode}-{prompt["id"]}'
        payload = dict(rid=rid, input_ids=prompt["input_ids"],
                       sampling_params=dict(temperature=0, max_new_tokens=64, ignore_eos=True),
                       return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
        before = set(args.captures.rglob("Pass*.pt"))
        request = urllib.request.Request(args.base_url + "/generate", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.load(response)
        (args.output / f'{prompt["id"]}.response.json').write_text(json.dumps(dict(payload=payload, response=answer)) + "\n")
        tokens = validate(answer, rid, 64)
        deadline = time.monotonic() + 30
        while True:
            files = set(args.captures.rglob("Pass*.pt")) - before
            if len(files) >= 65:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("FD_INCOMPLETE_65_FORWARD_CAPTURE")
            time.sleep(.1)
        if len(files) != 65:
            raise ValueError("FD_UNEXPECTED_FORWARD_COUNT")
        ordered = []
        for path in files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            ordered.append((data["capture.transaction"]["forward_sequence_id"], path))
        steps, logits = [], []
        for step, (_, path) in enumerate(sorted(ordered)):
            record, tensor = inspect_capture(path, prompt, tokens, step, rid, metadata, args.mode)
            sequences.append(record["forward_id"])
            steps.append(record)
            if step < 64:
                logits.append(tensor)
        torch.save(torch.stack(logits), args.output / f'{prompt["id"]}.logits.pt')
        rows.append(dict(prompt_id=prompt["id"], output_ids=tokens, steps=steps))
        print(json.dumps(dict(mode=args.mode, prompt_id=prompt["id"], complete_forwards=65)), flush=True)
    if len({s[0] for s in sequences}) != 1 or [s[1] for s in sequences] != list(range(len(sequences))):
        raise ValueError("FD_FORWARD_SEQUENCE_GAP_OR_RESTART")
    if list(args.captures.rglob("*.incomplete")):
        raise ValueError("FD_UNCOMMITTED_CAPTURE")
    report = dict(status="CAPTURE_COMPLETE", mode=args.mode, corpus_sha256=corpus["sha256"],
                  candidate_sha256=candidate["sha256"], capture_provenance=metadata,
                  rows=rows, complete_forwards=len(sequences))
    report["sha256"] = canonical_sha(report)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
