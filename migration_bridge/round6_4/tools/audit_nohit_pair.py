"""Audit every captured forward of the frozen [1000] OFF/ON request."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    report = dict(status="RUNNING", definition="entire request including physical lookahead", forwards=[])
    try:
        results = []
        for profile in ("D1_NODEFER", "D1"):
            root = args.root / profile
            assert json.loads((root / "session.json").read_text())["status"] == "PREQUALIFICATION_SESSION_PASS"
            integrity = json.loads((root / "final-integrity.json").read_text())
            assert integrity["status"] == "PASS" and integrity["physical_captures"] == 2
            r = json.loads((root / "capture/result.json").read_text())
            assert len(r["rows"]) == 1 and r["rows"][0]["input_ids"] == [1000]
            assert len(r["rows"][0]["steps"]) == 2
            results.append(r["rows"][0])
        assert results[0]["baseline_output_ids"] == results[1]["baseline_output_ids"]
        for off_step, on_step in zip(results[0]["steps"], results[1]["steps"]):
            assert off_step["step"] == on_step["step"]
            captures = [torch.load(s["file"], map_location="cpu", weights_only=False, mmap=True)
                        for s in (off_step, on_step)]
            item = dict(step=off_step["step"], target=off_step["target"], layers=[], files=[])
            for profile, step, data in zip(("OFF", "ON"), (off_step, on_step), captures):
                item["files"].append(dict(profile=profile, path=step["file"]))
                for layer, owned in ((7, (16, 38, 41, 50)), (8, (8, 25, 33, 55))):
                    prefix = f"r62.ktep.layer{layer}."
                    ids = data[prefix + "topk_ids"]
                    hits = sum(int((ids == expert).sum()) for expert in owned)
                    cpu, npu, merged = [data[prefix + k] for k in ("cpu_partial", "npu_owned_partial", "routed_merge")]
                    finite = all(bool(torch.isfinite(t).all()) for t in (cpu, npu, merged))
                    zero = bool((cpu == 0).all())
                    merge_exact = exact(merged, npu.to(merged.dtype))
                    item["layers"].append(dict(profile=profile, layer=layer, cpu_routes=hits,
                                               cpu_zero=zero, merge_exact_after_visible_cast=merge_exact, finite=finite))
                    if hits:
                        raise ValueError(f"FROZEN_CONTROL_NOT_ELIGIBLE:{profile}:step{off_step['step']}:L{layer}:{hits}")
                    assert finite and zero and merge_exact, "CPU_NOT_HIT_CONTAMINATION"
                for key, expected_sha in data["capture.transaction"]["tensor_byte_sha256"].items():
                    actual = hashlib.sha256(data[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert actual == expected_sha, "CAPTURE_HASH_MISMATCH"
            item["full_logits_byte_exact"] = exact(captures[0]["capture.next_token_logits"], captures[1]["capture.next_token_logits"])
            assert item["full_logits_byte_exact"], "CPU_NOT_HIT_LOGIT_MISMATCH"
            report["forwards"].append(item)
        report["status"] = "WHOLE_REQUEST_NOHIT_PASS"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        target = args.root / "whole-request-audit.json"
        with target.open("x") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
