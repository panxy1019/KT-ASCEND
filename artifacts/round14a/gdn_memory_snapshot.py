"""Small, dependency-free helpers for torch_npu allocator snapshots.

The device probe imports this module inside the frozen runtime image.  Keeping
the analysis pure makes the attribution rules testable without an NPU.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


ACTIVE_STATES = frozenset(("active_allocated", "active_awaiting_free"))


def _segments(snapshot: Mapping[str, Any] | list[Mapping[str, Any]]):
    if isinstance(snapshot, list):
        return snapshot
    return snapshot.get("segments", [])


def _frame_key(frames: Iterable[Mapping[str, Any]]) -> str:
    frames = list(frames or ())
    if not frames:
        return "UNKNOWN"
    # The allocator records innermost-to-outermost frames.  Prefer the first
    # Python frame outside torch internals, then fall back to the first frame.
    chosen = frames[0]
    for frame in frames:
        filename = str(frame.get("filename", ""))
        if filename and "/torch/" not in filename and "/torch_npu/" not in filename:
            chosen = frame
            break
    return "{}:{}:{}".format(
        chosen.get("filename", ""),
        int(chosen.get("line", 0) or 0),
        chosen.get("name", ""),
    )


def snapshot_totals(snapshot: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, int]:
    totals = defaultdict(int)
    for segment in _segments(snapshot):
        totals["segment_bytes"] += int(segment.get("total_size", 0))
        totals["segment_allocated_bytes"] += int(segment.get("allocated_size", 0))
        totals["segment_active_bytes"] += int(segment.get("active_size", 0))
        for item in segment.get("blocks", ()): 
            state = str(item.get("state", "unknown"))
            size = int(item.get("size", 0))
            requested = int(item.get("requested_size", size))
            totals[f"{state}_bytes"] += size
            totals[f"{state}_requested_bytes"] += requested
            if state in ACTIVE_STATES:
                totals["active_bytes"] += size
                totals["active_requested_bytes"] += requested
    return dict(sorted(totals.items()))


def _block_index(snapshot):
    result = {}
    for segment in _segments(snapshot):
        stream = str(segment.get("stream", "UNKNOWN"))
        for block in segment.get("blocks", ()):
            row = dict(block)
            row["segment_stream"] = stream
            row["segment_address"] = int(segment.get("address", 0))
            result[int(block.get("address", 0))] = row
    return result


def diff_active_blocks(before, after) -> dict[str, Any]:
    before_blocks = _block_index(before)
    states = defaultdict(int)
    streams = defaultdict(int)
    owners = defaultdict(int)
    selected = []
    requested_total = 0
    size_total = 0
    for address, block in sorted(_block_index(after).items()):
        state = str(block.get("state", "unknown"))
        if state not in ACTIVE_STATES:
            continue
        old = before_blocks.get(address)
        if old is not None and str(old.get("state")) in ACTIVE_STATES:
            continue
        size = int(block.get("size", 0))
        requested = int(block.get("requested_size", size))
        owner = _frame_key(block.get("frames", ()))
        row = dict(block)
        row.update(address=address, owner=owner)
        selected.append(row)
        size_total += size
        requested_total += requested
        states[state] += size
        streams[str(block.get("segment_stream", "UNKNOWN"))] += size
        owners[owner] += size
    return {
        "new_active_bytes": size_total,
        "new_active_requested_bytes": requested_total,
        "states": dict(sorted(states.items())),
        "streams": dict(sorted(streams.items())),
        "owners": dict(sorted(owners.items(), key=lambda item: (-item[1], item[0]))),
        "blocks": selected,
    }


def match_live_tensors(blocks, live_tensors):
    rows = []
    for block in blocks:
        start = int(block.get("address", 0))
        end = start + int(block.get("size", 0))
        names = []
        tensors = []
        for tensor in live_tensors:
            pointer = int(tensor.get("data_ptr", 0))
            if start <= pointer < end:
                names.append(str(tensor.get("name", "UNKNOWN")))
                tensors.append(dict(tensor))
        row = dict(block)
        row["tensor_names"] = sorted(names)
        row["tensors"] = sorted(tensors, key=lambda item: str(item.get("name", "")))
        rows.append(row)
    return rows


def plateau_verdict(values, *, warmup: int, tolerance_bytes: int) -> dict[str, Any]:
    values = [int(value) for value in values]
    if warmup < 0 or warmup >= len(values):
        raise ValueError("warmup must leave at least one measured value")
    measured = values[warmup:]
    low = min(measured)
    high = max(measured)
    growth = measured[-1] - measured[0]
    return {
        "pass": high - low <= int(tolerance_bytes),
        "warmup": warmup,
        "samples": len(values),
        "measured_samples": len(measured),
        "min_after_warmup_bytes": low,
        "max_after_warmup_bytes": high,
        "span_after_warmup_bytes": high - low,
        "growth_after_warmup_bytes": growth,
    }
