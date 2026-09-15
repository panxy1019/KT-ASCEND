"""Frozen 128x64 baseline histories and complete-forward matched-history sweeps.

No retries, filtering, corpus edits, resume merging, or threshold fitting.
Run under the owned-process supervisor; a failure invalidates this run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request
import torch
from assemble_capture import assemble, canonical_sha


def frozen(path):
    value = json.loads(path.read_text())
    actual = canonical_sha({k: v for k, v in value.items() if k != "sha256"})
    if actual != value["sha256"]:
        raise ValueError("FROZEN_INPUT_HASH_MISMATCH")
    return value


def write(path, value):
    if path.exists():
        raise ValueError(f"refuse evidence overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def request(url, rid, ids, count):
    body = dict(rid=rid, input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True),
                return_logprob=True, logprob_start_len=-1, top_logprobs_num=16)
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    if result["meta_info"]["id"] != rid or len(result["output_ids"]) != count:
        raise ValueError("REQUEST_ID_OR_TOKEN_COUNT_MISMATCH")
    probabilities = result["meta_info"].get("output_token_logprobs", [])
    if len(probabilities) != count or any(not math.isfinite(float(row[0])) for row in probabilities):
        raise ValueError("NONFINITE_OR_MISSING_OUTPUT_LOGPROBS")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["history", "sweep"])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:31000")
    parser.add_argument("--mode", choices=["defer0", "defer1"], required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--heldout-entry", type=Path,
                        help="committed candidate and PASS runtime evidence entry manifest")
    args = parser.parse_args()
    corpus = frozen(args.corpus)
    if corpus["corpus"] == "HD2":
        if args.heldout_entry is None:
            raise ValueError("HD2_REQUIRES_CANDIDATE_AND_RUNTIME_GATE")
        entry = frozen(args.heldout_entry)
        if entry.get("runtime_regression") != "PASS" or not entry.get("candidate_commit"):
            raise ValueError("HD2_PREDECESSOR_GATE_NOT_CLOSED")
    if corpus["count"] != 128 or len(corpus["prompts"]) != 128 or corpus["protocol"]["positions"] != 64:
        raise ValueError("CORPUS_PROTOCOL_DRIFT")
    if args.root.exists():
        raise ValueError("output directory already exists; no implicit resume")
    if args.action == "history" and args.mode != "defer0":
        raise ValueError("HISTORY_MUST_BE_BASELINE")
    provenance = json.loads(args.provenance.read_text())
    if provenance["mode"] != args.mode:
        raise ValueError("PROVENANCE_MODE_MISMATCH")
    args.root.mkdir(parents=True)
    write(args.root / "run_spec.json", dict(action=args.action, mode=args.mode,
          corpus_sha256=corpus["sha256"], provenance=provenance,
          http_timeout_seconds=600, capture_timeout_seconds=30,
          tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows = []
    history = frozen(args.history) if args.action == "sweep" else None
    if history is not None and history["corpus_sha256"] != corpus["sha256"]:
        raise ValueError("HISTORY_CORPUS_MISMATCH")
    history_rows = {r["prompt_id"]: r for r in history["rows"]} if history else {}
    for prompt in corpus["prompts"]:
        if args.action == "history":
            answer = request(args.url, f'r62-history-{prompt["id"]}', prompt["input_ids"], 64)
            row = dict(prompt_id=prompt["id"], output_ids=answer["output_ids"], response=answer)
            write(args.root / "responses" / f'{prompt["id"]}.json', row)
            rows.append(row)
        else:
            generated = history_rows[prompt["id"]]["output_ids"]
            if len(generated) != 64:
                raise ValueError("HISTORY_LENGTH_MISMATCH")
            vectors = []
            for index in range(64):
                rid = f'r62-{corpus["corpus"].lower()}-{args.mode}-{prompt["id"]}-{index:02d}'
                ids = prompt["input_ids"] + generated[:index]
                before = set(args.captures.rglob("Pass*.pt"))
                answer = request(args.url, rid, ids, 1)
                write(args.root / "responses" / f'{prompt["id"]}-{index:02d}.json', answer)
                deadline = time.monotonic() + 30
                matches = []
                while not matches:
                    for path in sorted(set(args.captures.rglob("Pass*.pt")) - before):
                        data = torch.load(path, map_location="cpu", weights_only=False)
                        tx = data["capture.transaction"]
                        if tx["batch"]["request_ids"] != [rid]:
                            continue
                        if data["capture.forward_batch_info.input_ids"].reshape(-1).tolist() == ids:
                            matches.append(path)
                    if matches:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("complete-forward commit timeout")
                    time.sleep(.1)
                if len(matches) != 1:
                    raise ValueError("DUPLICATE_REQUEST_TARGET")
                record = dict(request_id=rid, prompt_id=prompt["id"], prompt_hash=prompt["prompt_sha256"],
                              input_ids=ids, history_ids=generated[:index],
                              baseline_history_sha256=history["sha256"], token_index=index,
                              output_token=int(answer["output_ids"][0]))
                row, logits = assemble(matches[0], record, args.mode)
                if row["capture_provenance"] != provenance:
                    raise ValueError("CAPTURE_PROVENANCE_MISMATCH")
                row.update(baseline_token=int(generated[index]), serving_token=record["output_token"],
                           category=prompt["category"])
                rows.append(row)
                vectors.append(logits)
                write(args.root / "positions" / f'{prompt["id"]}-{index:02d}.json', row)
            (args.root / "logits").mkdir(exist_ok=True)
            torch.save(torch.stack(vectors), args.root / "logits" / f'{prompt["id"]}.pt')
        print(json.dumps(dict(action=args.action, mode=args.mode, prompt_id=prompt["id"],
                              completed_rows=len(rows))), flush=True)
    result = dict(action=args.action, mode=args.mode, corpus_sha256=corpus["sha256"],
                  positions_per_prompt=64, row_count=len(rows), rows=rows,
                  baseline_history_sha256=history["sha256"] if history else None)
    result["sha256"] = canonical_sha(result)
    write(args.root / "manifest.json", result)


if __name__ == "__main__":
    main()
