"""Same-mode/same-history full versus compact capture byte parity."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    a, b = json.loads(args.full.read_text()), json.loads(args.compact.read_text())
    if len(a["rows"]) != len(b["rows"]):
        raise ValueError("OBSERVER_REQUEST_COUNT")
    counts = dict(forwards=0, retained_tensor_byte_comparisons=0, full_bytes=0, compact_bytes=0)
    def compare_value(value, other, label):
        if isinstance(value, torch.Tensor):
            if not isinstance(other, torch.Tensor) or value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value.contiguous().view(torch.uint8), other.contiguous().view(torch.uint8)):
                raise ValueError(f"OBSERVER_TENSOR_MISMATCH:{label}")
            counts["retained_tensor_byte_comparisons"] += 1
        elif isinstance(value, (list, tuple)):
            if type(value) is not type(other) or len(value) != len(other):
                raise ValueError("OBSERVER_SEQUENCE_SHAPE")
            for i, (v, w) in enumerate(zip(value, other, strict=True)):
                compare_value(v, w, f"{label}:{i}")
        elif value != other:
            raise ValueError("OBSERVER_RETAINED_VALUE")
    for x, y in zip(a["rows"], b["rows"], strict=True):
        if any(x[k] != y[k] for k in ("rid", "input_ids", "baseline_output_ids")):
            raise ValueError("OBSERVER_HISTORY_MISMATCH")
        for s, t in zip(x["steps"], y["steps"], strict=True):
            left = torch.load(s["file"], map_location="cpu", weights_only=False, mmap=True)
            right = torch.load(t["file"], map_location="cpu", weights_only=False, mmap=True)
            if left["capture.transaction"]["provenance"]["mode"] != right["capture.transaction"]["provenance"]["mode"]:
                raise ValueError("OBSERVER_MODE_MISMATCH")
            if right["capture.transaction"]["r63_observer"] != "compact":
                raise ValueError("COMPACT_NOT_ENABLED")
            if right["capture.transaction"]["r63_full_diagnostic"] and set(left) != set(right):
                raise ValueError("OBSERVER_FULL_DIAGNOSTIC_KEYS")
            for key, value in right.items():
                if key != "capture.transaction":
                    compare_value(value, left[key], f"{x['rid']}:{s['step']}:{key}")
            # Device pointer / wall-clock telemetry can differ; compare only
            # established semantic event records through each run's validator.
            if (s["current_deferred"], s["cumulative_exposure"]) != (t["current_deferred"], t["cumulative_exposure"]):
                raise ValueError("OBSERVER_EXPOSURE_MISMATCH")
            counts["forwards"] += 1
            counts["full_bytes"] += Path(s["file"]).stat().st_size
            counts["compact_bytes"] += Path(t["file"]).stat().st_size
    report = dict(status="PASS", **counts,
        not_claimed="dropped tensors are diagnostic-only, not silently validated in compact mode")
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
