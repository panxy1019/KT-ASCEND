"""Summarize bounded Round9 working-set traces without performance claims."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(documents):
    if not documents:
        raise ValueError("at least one timeline document is required")
    phases = defaultdict(lambda: {"count": 0, "total_ns": 0, "max_ns": 0,
                                  "bytes": 0})
    clocks = Counter()
    categories = set()
    scheduler_batches = []
    shadow_identities = Counter()
    dropped = 0
    independent_work_before_ple_wait = False
    h2d_compute_overlap = False
    requests = []

    for document in documents:
        if document.get("schema") != 1:
            raise ValueError("unsupported timeline schema")
        events = document.get("events")
        if not isinstance(events, list) or document.get("event_count") != len(events):
            raise ValueError("timeline event count mismatch")
        if [event.get("sequence") for event in events] != list(range(len(events))):
            raise ValueError("timeline sequence is not contiguous")
        document_dropped = document.get("dropped_events", 0)
        if type(document_dropped) is not int or document_dropped < 0:
            raise ValueError("timeline dropped count is invalid")
        dropped += document_dropped
        requests.append(document.get("identity"))
        begins = []
        gather_submits = []
        h2d_enqueues = []
        for index, event in enumerate(events):
            category = event.get("category")
            phase = event.get("phase")
            clock = event.get("clock_domain")
            if not all(isinstance(value, str) and value
                       for value in (category, phase, clock)):
                raise ValueError("timeline event identity is malformed")
            categories.add(category)
            clocks[clock] += 1
            metric = phases[f"{category}:{phase}"]
            metric["count"] += 1
            duration = event.get("duration_ns", 0)
            byte_count = event.get("bytes", 0)
            if (type(duration) is not int or duration < 0
                    or type(byte_count) is not int or byte_count < 0):
                raise ValueError("timeline duration/bytes must be nonnegative integers")
            metric["total_ns"] += duration
            metric["max_ns"] = max(metric["max_ns"], duration)
            metric["bytes"] += byte_count
            if category == "scheduler" and phase == "batch_begin":
                begins.append(event)
            elif category == "scheduler" and phase == "batch_end":
                if not begins:
                    raise ValueError("scheduler batch end lacks begin")
                begin = begins.pop(0)
                scheduler_batches.append({
                    "mode": begin.get("mode"),
                    "duration_ns": event["host_monotonic_ns"] - begin["host_monotonic_ns"],
                })
            if category == "ple" and phase == "gather_submit":
                gather_submits.append(index)
            elif (category == "ple" and phase == "gather_wait_and_h2d_dependency"
                  and gather_submits):
                start = gather_submits.pop(0)
                independent_work_before_ple_wait = independent_work_before_ple_wait or any(
                    item.get("category") == "expert" for item in events[start + 1:index]
                )
            elif category == "ple" and phase == "h2d_enqueue":
                h2d_enqueues.append(index)
            elif (category == "ple" and phase == "transfer_consumer_retire_wait"
                  and h2d_enqueues):
                start = h2d_enqueues.pop(0)
                h2d_compute_overlap = h2d_compute_overlap or any(
                    item.get("category") == "expert" for item in events[start + 1:index]
                )
            if category == "expert" and phase == "shadow_prepare":
                key = (event.get("layer"), tuple(event.get("logical_experts", ())))
                shadow_identities[key] += 1
        if begins and document_dropped == 0:
            raise ValueError("scheduler batch begin lacks end")

    required = {"scheduler", "ple", "expert"}
    complete = dropped == 0 and required <= categories
    return {
        "schema": 1,
        "status": ("PASS_BOUNDED_TIMELINE_PROFILE" if complete
                   else "INCOMPLETE_BOUNDED_TIMELINE_PROFILE"),
        "scope": "PROFILE_ONLY_NOT_PERFORMANCE_QUALIFICATION",
        "requests": requests,
        "event_count": sum(metric["count"] for metric in phases.values()),
        "dropped_events": dropped,
        "clock_domains": dict(sorted(clocks.items())),
        "phase_metrics": dict(sorted(phases.items())),
        "scheduler_batches": scheduler_batches,
        "independent_work_before_ple_wait_observed": independent_work_before_ple_wait,
        "h2d_expert_compute_overlap_observed": h2d_compute_overlap,
        "repeated_shadow_identity_candidates": sum(
            count - 1 for count in shadow_identities.values() if count > 1),
        "limitations": [
            "device operations use host enqueue/wait observations; no cross-domain timestamp subtraction",
            "work before the PLE wait can overlap CPU gather but does not by itself prove H2D overlap",
            "profile does not establish product quality or performance qualification",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [json.loads(line) for line in args.input.read_text().splitlines()
                 if line.strip()]
    args.output.write_text(json.dumps(summarize(documents), indent=2) + "\n")


if __name__ == "__main__":
    main()
