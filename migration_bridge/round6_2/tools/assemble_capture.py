"""Bind complete-forward D1 captures to a request with independent API identity."""
import hashlib
import json
from pathlib import Path
import torch
from validate_position import validate


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def byte_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def assemble(path: Path, request: dict, mode: str):
    payload = torch.load(path,map_location="cpu",weights_only=False)
    tx = payload["capture.transaction"]
    for name, expected in tx["tensor_byte_sha256"].items():
        if byte_sha(payload[name]) != expected:
            raise ValueError("CAPTURE_BYTE_HASH_MISMATCH")
    ids = payload["capture.forward_batch_info.input_ids"].reshape(-1).tolist()
    positions = payload["capture.forward_batch_info.positions"].reshape(-1).tolist()
    if ids != request["input_ids"] or positions != list(range(len(ids))):
        raise ValueError("FULL_REQUEST_INPUT_OR_POSITION_MISMATCH")
    if tx["batch"]["request_ids"] != [request["request_id"]]:
        raise ValueError("REQUEST_ID_MISMATCH")
    logits = payload["capture.next_token_logits"][-1].contiguous()
    if logits.shape != (102400,) or not torch.isfinite(logits).all():
        raise ValueError("INVALID_LOGITS")
    if float(logits.max()-logits[request["output_token"]]) != 0:
        raise ValueError("API_TOKEN_NOT_CAPTURE_MAXIMUM")
    routes = []
    cpu_counts = [0,0]  # prefix, current
    defer_counts = [0,0]
    for layer, owned in ((7,{16,38,41,50}),(8,{8,25,33,55})):
        topids = payload[f"r62.model.layer{layer}.router_topk_ids"]
        router = payload[f"r62.model.layer{layer}.router_logits"].float()
        weights = payload[f"r62.cpu.layer{layer}.routing_weights"]
        immediate = payload[f"r62.cpu.layer{layer}.immediate_ids"]
        deferred = payload.get(f"r62.cpu.layer{layer}.deferred_ids")
        if topids.shape != (len(ids),6) or router.shape != (len(ids),64) or weights.shape != topids.shape or immediate.shape != topids.shape:
            raise ValueError("ROUTE_SHAPE_MISMATCH")
        if deferred is not None and deferred.shape != topids.shape:
            raise ValueError("DEFERRED_SHAPE_MISMATCH")
        if not torch.isfinite(router).all() or not torch.isfinite(weights).all():
            raise ValueError("NONFINITE_ROUTING")
        # Frozen model config: softmax, norm_topk_prob=false, routed scaling=1.
        scores = router.softmax(dim=-1)
        for token in range(len(ids)):
            current = int(token == len(ids)-1)
            for rank in range(6):
                expert = int(topids[token,rank])
                if expert not in owned:
                    continue
                cpu_counts[current] += 1
                is_deferred = deferred is not None and int(deferred[token,rank]) == expert
                is_immediate = int(immediate[token,rank]) == expert
                if is_deferred == is_immediate:
                    raise ValueError("CPU_OWNERSHIP_SPLIT_NOT_EXACT_ONCE")
                if not is_deferred:
                    continue
                defer_counts[current] += 1
                if layer != 7 or mode != "defer1":
                    raise ValueError("ILLEGAL_DEFERRED_ROUTE")
                routes.append(dict(event_id=f'{request["request_id"]}:{layer}:{token}:{rank}',
                    request_id=request["request_id"],producer_layer=7,consumer_layer=8,
                    expert_logical_id=expert,topk_rank=rank,token_position=token,
                    scope="current" if current else "prefix",
                    route_score=float(scores[token,expert]),normalized_route_weight=float(weights[token,rank])))
    events = tx["deferred_events"]
    pending = [e for e in events if e["event"]=="pending_state"]
    sync = [e for e in events if e["event"]=="sync_complete" and e.get("layer_idx")==8]
    drained = bool(pending and pending[-1]["pending_layers"]==[] and sync and sync[-1]["allow_pending"]==0)
    current,prefix = defer_counts[1],defer_counts[0]
    exposure = ("CURRENT_AND_PREFIX_EXPOSURE" if current and prefix else "CURRENT_EXPOSURE" if current else "PREFIX_ONLY_EXPOSURE" if prefix else "NO_EXPOSURE")
    row = dict(position_id=f'{request["prompt_id"]}:{request["token_index"]}',
        request_id=request["request_id"],prompt_id=request["prompt_id"],
        prompt_hash=request["prompt_hash"],history_hash=canonical_sha(request["history_ids"]),
        request_input_hash=canonical_sha(ids),token_index=request["token_index"],
        current_input_position=len(ids)-1,forward_id=f'{tx["pid"]}:{tx["forward_sequence_id"]}',
        logits_sha256=byte_sha(logits),logits_shape=list(logits.shape),mode=mode,
        baseline_history_sha256=request["baseline_history_sha256"],
        trace_complete=True,pending_drained=drained,
        deferred_event_count_current_row=current,deferred_event_count_prefix=prefix,
        cpu_route_count_current_row=cpu_counts[1],cpu_route_count_prefix=cpu_counts[0],
        deferred_route_records=routes,deferred_exposure=exposure,
        capture_file=str(path),capture_provenance=tx["provenance"],
        optional_deferred_partial_norm="NOT_CAPTURED_PENDING_BUFFER_UNSAFE_TO_READ")
    validate(row)
    return row,logits
