"""Export causal per-position route records from an audited stateful session.

No cross-mode logit metrics or reserve derivation. Full logits remain referenced
by immutable capture hashes. This is a schema adapter, not an acceptance gate.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round6_2/tools"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--forcing", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from assemble_capture import canonical_sha, byte_sha
    from audit_deferred_transitions import validate_transition
    from validate_position import validate
    torch.set_num_threads(1)
    session = json.loads((args.session / "session.json").read_text())
    audit = json.loads((args.session / "final-integrity.json").read_text())
    capture = json.loads((args.session / "capture/result.json").read_text())
    if session["status"] != "PREQUALIFICATION_SESSION_PASS" or audit["status"] != "PASS":
        raise ValueError("STATEFUL_FINAL_INTEGRITY_NOT_PASSED")
    if not capture["forced"]:
        raise ValueError("FORMAL_POSITION_EXPORT_REQUIRES_REPLAY")
    corpus = json.loads(args.corpus.read_text())
    forcing_raw = args.forcing.read_bytes()
    forcing = json.loads(forcing_raw)
    force_by_rid = {r["rid"]: r for r in forcing["requests"]}
    corpus_by_rid = {r["rid"]: r for r in corpus["requests"]}
    if len(force_by_rid) != len(forcing["requests"]) or set(force_by_rid) != set(corpus_by_rid):
        raise ValueError("CORPUS_FORCE_COVERAGE")
    if [r["rid"] for r in capture["rows"]] != list(corpus_by_rid):
        raise ValueError("CORPUS_ORDER_OR_COVERAGE")
    history_sha = __import__("hashlib").sha256(forcing_raw).hexdigest()
    natural_records = [json.loads(line) for line in (Path(session["server_root"]) / "forcing.jsonl").read_text().splitlines()]
    natural = {(r["rid"], r["step"]): r for r in natural_records}
    if len(natural) != len(natural_records):
        raise ValueError("DUPLICATE_SAMPLE_STEP")
    mode = "defer1" if session["profile"] == "D1" else "defer0"
    rows = []
    for request in capture["rows"]:
        rid, prompt = request["rid"], request["input_ids"]
        force = force_by_rid[rid]
        if prompt != force["input_ids"] or request["baseline_output_ids"] != force["baseline_output_ids"]:
            raise ValueError("FROZEN_HISTORY_DRIFT")
        if len(force["baseline_output_ids"]) != 64:
            raise ValueError("FORMAL_BUDGET_NOT_64")
        history_routes, prior_cpu = [], 0
        for s in request["steps"]:
            if not s["target"]:
                continue
            t = s["step"]
            data = torch.load(s["file"], map_location="cpu", weights_only=False, mmap=True)
            tx = data["capture.transaction"]
            if tx["provenance"]["mode"] != mode or tx["batch"]["request_ids"] != [rid]:
                raise ValueError("CAPTURE_MODE_OR_RID")
            for key, sha in tx["tensor_byte_sha256"].items():
                if byte_sha(data[key]) != sha:
                    raise ValueError("CAPTURE_HASH_DRIFT")
            full_history = prompt + force["baseline_output_ids"][:t]
            ids = data["capture.forward_batch_info.input_ids"].tolist()
            positions = data["capture.forward_batch_info.positions"].tolist()
            if ids != (prompt if t == 0 else [full_history[-1]]) or positions != (list(range(len(prompt))) if t == 0 else [len(full_history)-1]):
                raise ValueError("STATEFUL_INPUT_DRIFT")
            current_pos = len(full_history)-1
            logits = data["capture.next_token_logits"][0]
            sample = natural[(rid, t)]
            if sample["sampled_logits_sha256"] != byte_sha(logits) or sample["manifest_sha256"] != history_sha:
                raise ValueError("SAMPLE_HASH_DRIFT")
            cpu_counts = [prior_cpu, 0]
            new_routes = []
            for layer, owned in ((7, (16,38,41,50)), (8, (8,25,33,55))):
                top = data[f"r62.ktep.layer{layer}.topk_ids"]
                weights = data[f"r62.ktep.layer{layer}.routing_weights"]
                router = data[f"r62.model.layer{layer}.router_logits"].float().softmax(-1)
                immediate = data[f"r62.cpu.layer{layer}.immediate_ids"]
                delayed = data.get(f"r62.cpu.layer{layer}.deferred_ids")
                for r, position in enumerate(positions):
                    for rank in range(6):
                        expert = int(top[r, rank])
                        if expert not in owned:
                            continue
                        cpu_counts[int(position == current_pos)] += 1
                        is_delayed = delayed is not None and int(delayed[r, rank]) == expert
                        if is_delayed == (int(immediate[r, rank]) == expert):
                            raise ValueError("STATEFUL_ROUTE_EXACT_ONCE")
                        if is_delayed:
                            if layer != 7 or mode != "defer1":
                                raise ValueError("STATEFUL_ILLEGAL_DEFERRED_ROUTE")
                            new_routes.append(dict(event_id=f"{rid}:{t}:{layer}:{position}:{rank}",
                                request_id=rid, producer_layer=7, consumer_layer=8,
                                expert_logical_id=expert, topk_rank=rank, token_position=position,
                                route_score=float(router[r, expert]), normalized_route_weight=float(weights[r, rank])))
            validate_transition(tx["deferred_events"], len(new_routes))
            history_routes.extend(new_routes)
            route_records = [dict(r, scope="current" if r["token_position"] == current_pos else "prefix") for r in history_routes]
            current = sum(r["scope"] == "current" for r in route_records)
            prefix = len(route_records)-current
            label = "CURRENT_AND_PREFIX_EXPOSURE" if current and prefix else "CURRENT_EXPOSURE" if current else "PREFIX_ONLY_EXPOSURE" if prefix else "NO_EXPOSURE"
            row = dict(position_id=f"{rid}:{t}", request_id=rid, prompt_id=rid,
                prompt_hash=canonical_sha(prompt), history_hash=canonical_sha(force["baseline_output_ids"][:t]),
                request_input_hash=canonical_sha(full_history), token_index=t,
                current_input_position=current_pos, forward_id=f"{tx['pid']}:{tx['forward_sequence_id']}",
                logits_sha256=byte_sha(logits), logits_shape=[102400], mode=mode,
                baseline_history_sha256=history_sha, trace_complete=True, pending_drained=True,
                deferred_event_count_current_row=current, deferred_event_count_prefix=prefix,
                cpu_route_count_current_row=cpu_counts[1], cpu_route_count_prefix=cpu_counts[0],
                deferred_route_records=route_records, deferred_exposure=label,
                serving_token=sample["natural_token"], forced_token=sample["returned_token"],
                category=corpus_by_rid[rid].get("category", "engineering"), capture_file=s["file"],
                actual_input_ids=ids, actual_positions=positions)
            validate(row)
            rows.append(row)
            prior_cpu = sum(cpu_counts)
    report = dict(version="r63-stateful-position-export-v1", mode=mode,
        corpus_sha256=canonical_sha(corpus), baseline_history_sha256=history_sha,
        row_count=len(rows), rows=rows, numerical_metrics_read=False)
    report["sha256"] = canonical_sha(report)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps(dict(status="PASS", positions=len(rows), mode=mode,
        exposed=sum(r["deferred_exposure"] != "NO_EXPOSURE" for r in rows))))


if __name__ == "__main__":
    main()
