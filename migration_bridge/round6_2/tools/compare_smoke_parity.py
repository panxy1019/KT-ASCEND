"""Fail closed on byte-level OFF/reference versus complete-telemetry differences."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse parity evidence overwrite")
    runs = [json.loads((root / "smoke_result.json").read_text())
            for root in (args.on, args.reference)]
    if runs[0]["mode"] != runs[1]["mode"]:
        raise ValueError("MODE_MISMATCH")
    tensors = [torch.load(root / "smoke_logits.pt", weights_only=True, map_location="cpu")
               for root in (args.on, args.reference)]
    if any(t.shape != (20, 102400) for t in tensors):
        raise ValueError("PARITY_SHAPE_MISMATCH")
    pairs = []
    for index, (on, ref) in enumerate(zip(runs[0]["rows"], runs[1]["rows"])):
        if on["request"]["input_ids"] != ref["request"]["input_ids"]:
            raise ValueError("PARITY_INPUT_MISMATCH")
        delta = (tensors[0][index].double() - tensors[1][index].double()).abs()
        pairs.append(dict(index=index, token_exact=on["output_token"] == ref["output_token"],
                          logits_byte_exact=sha(tensors[0][index]) == sha(tensors[1][index]),
                          max_abs=float(delta.max()), num_diff=int((delta != 0).sum())))
    exact = all(row["token_exact"] and row["logits_byte_exact"] for row in pairs)
    result = dict(mode=runs[0]["mode"], requests=20, pairs=pairs,
                  on_sha256=sha(tensors[0]), reference_sha256=sha(tensors[1]),
                  measurement="reference has final-logit observer only; full transaction hooks disabled",
                  status="PASS" if exact else "INSTRUMENTATION_PERTURBS_EXECUTION")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if exact else 1)


if __name__ == "__main__":
    main()
