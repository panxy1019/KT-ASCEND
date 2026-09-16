"""Join all compact physical forwards to per-RID native request histories."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys


def expected_row(request, step):
    prompt, generated = request["input_ids"], request["output_ids"]
    if not 0 <= step <= len(generated):
        raise ValueError("EXCESS_LOOKAHEAD")
    return (prompt, list(range(len(prompt))), len(prompt)) if step == 0 else (
        [generated[step-1]], [len(prompt)+step-1], len(prompt)+step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, "/workspace/results/round6_2-prequalification/tools")
    from audit_deferred_transitions import validate_transition
    root = args.root
    run = json.loads((root/"batch-report.json").read_text())
    assert json.loads((root/"lifecycle.json").read_text())["status"] == "PLANNED_STOP"
    requests = {r["rid"]:r for r in run["rows"]}
    assert len(requests) == len(run["rows"]) == 16*run["repeats"]
    entries = [json.loads(s) for s in (root/"index.jsonl").read_text().splitlines()]
    files = set((root/"tensors").rglob("Pass*.pt"))
    assert len({e["pid"] for e in entries}) == 1
    assert files == {Path(e["file"]) for e in entries} and len(files) == len(entries)
    assert not list((root/"tensors").rglob("*.incomplete"))
    count = collections.Counter()
    actual_batch = collections.Counter()
    signatures = collections.defaultdict(list)
    report = dict(status="RUNNING", physical_forwards=len(entries), known_requests=len(requests),
                  cpu_owned_layer_forwards=0, real_cpu_tasks_with_selected_routes=0,
                  deferred_forwards=0, deferred_routes=0, expert_route_hits={},
                  worker_sync_complete=0, all_finite=True, failures=[])
    try:
        for sequence, entry in enumerate(entries):
            assert entry["sequence"] == sequence
            data = torch.load(entry["file"], map_location="cpu", weights_only=False, mmap=True)
            tx = data["capture.transaction"]
            assert (tx["pid"], tx["forward_sequence_id"]) == (entry["pid"], sequence)
            rids = entry["batch"]["request_ids"]
            assert rids == tx["batch"]["request_ids"] and len(set(rids)) == len(rids)
            assert 0 < len(rids) <= run["batch"] and set(rids) <= set(requests)
            lengths, inputs, positions = [], [], []
            steps = []
            for rid in rids:
                step = count[rid]
                ids, pos, seq_len = expected_row(requests[rid], step)
                inputs.extend(ids)
                positions.extend(pos)
                lengths.append(seq_len)
                steps.append(step)
                count[rid] += 1
            assert data["capture.forward_batch_info.input_ids"].tolist() == inputs, "BATCH_RID_TOKEN_HISTORY"
            assert data["capture.forward_batch_info.positions"].tolist() == positions, "BATCH_RID_POSITION"
            assert data["capture.forward_batch_info.seq_lens"].tolist() == lengths, "BATCH_RID_KV_LENGTH"
            logits = data["capture.next_token_logits"]
            assert logits.shape == (len(rids), 102400) and torch.isfinite(logits).all(), "BATCH_LOGITS"
            for i, rid in enumerate(rids):
                if steps[i] < 64:
                    assert logits[i, requests[rid]["output_ids"][steps[i]]] == logits[i].max(), "BATCH_ROW_OUTPUT_TOKEN"
            actual_batch[len(rids)] += 1
            repeats = {requests[rid]["repeat"] for rid in rids}
            assert len(repeats) == 1, "CROSS_REPEAT_SCHEDULE_OVERLAP"
            signatures[next(iter(repeats))].append(dict(prompts=[requests[rid]["prompt_id"] for rid in rids],
                steps=steps, lengths=lengths, forward_mode=tx["batch"]["forward_mode"]))
            total_deferred = 0
            for layer, owned in ((7, (16,38,41,50)), (8, (8,25,33,55))):
                ids = data[f"r62.ktep.layer{layer}.topk_ids"]
                weights = data[f"r62.ktep.layer{layer}.routing_weights"]
                immediate = data[f"r62.cpu.layer{layer}.immediate_ids"]
                delayed = data.get(f"r62.cpu.layer{layer}.deferred_ids", torch.full_like(ids, -1))
                assert ids.shape == immediate.shape == delayed.shape == weights.shape == (len(inputs), 6)
                assert torch.isfinite(weights).all() and torch.isfinite(data[f"r62.cpu.layer{layer}.cpu_output_after_sync"]).all()
                hits = immediate_hits = delayed_hits = 0
                for expert in owned:
                    selected = ids == expert
                    is_delayed = delayed == expert
                    assert torch.all(((immediate == expert) ^ is_delayed)[selected]), "ROUTE_EXACT_ONCE"
                    assert not torch.any(is_delayed & ~selected), "UNSELECTED_DEFERRED_ROUTE"
                    if layer == 8 or run["profile"] == "D1_NODEFER":
                        assert not is_delayed.any(), "TERMINAL_OR_OFF_DEFERRED"
                    n = int(selected.sum())
                    key = f"L{layer}:E{expert}"
                    report["expert_route_hits"][key] = report["expert_route_hits"].get(key,0)+n
                    hits += n
                    immediate_hits += int((immediate == expert).sum())
                    delayed_hits += int(is_delayed.sum())
                    total_deferred += int(is_delayed.sum())
                if layer == 7:
                    protected = set(ids[:, :5].flatten().tolist())
                    assert not protected.intersection(int(e) for e in delayed.flatten().tolist() if e >= 0), "GLOBAL_PROTECTED_UNION"
                report["cpu_owned_layer_forwards"] += int(hits > 0)
                report["real_cpu_tasks_with_selected_routes"] += int(immediate_hits > 0)+int(delayed_hits > 0)
            validate_transition(tx["deferred_events"], total_deferred)
            report["worker_sync_complete"] += sum(e["event"] == "sync_complete" for e in tx["deferred_events"])
            report["deferred_forwards"] += int(total_deferred > 0)
            report["deferred_routes"] += total_deferred
            for key, sha in tx["tensor_byte_sha256"].items():
                assert hashlib.sha256(data[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == sha
        assert set(count) == set(requests) and all(n in (64,65) for n in count.values()), "REQUEST_TARGET_OR_LOOKAHEAD_COUNT"
        assert actual_batch[run["batch"]] > 0, "REQUESTED_MODEL_BATCH_NOT_OBSERVED"
        report["actual_batch_histogram"] = dict(actual_batch)
        report["forward_counts_per_rid"] = dict(count)
        report["schedule_sha256_by_repeat"] = {str(r):hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest() for r,s in signatures.items()}
        assert len(set(report["schedule_sha256_by_repeat"].values())) == 1, "ACTUAL_SCHEDULE_VARIABILITY"
        assert run["status"] == "BATCH_CAPTURE_GENERATION_PASS"
        report["status"] = "BATCH_TRANSACTION_AUDIT_PASS"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        with (root/"batch-audit.json").open("x") as out:
            json.dump(report, out, indent=2)
        print(json.dumps({k:v for k,v in report.items() if k != "forward_counts_per_rid"}), flush=True)


if __name__ == "__main__":
    main()
