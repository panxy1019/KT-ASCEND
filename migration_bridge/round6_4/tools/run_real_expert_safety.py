"""Run the retained real-weight CPU/NPU probe for all frozen D1 experts.

Independent process per case avoids cross-case loader state. This is same-input
expert safety, not cross-layer Deferred output equivalence or quality evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROBE_SHA = "4be8d94090a7cc5e16e9597ed2759494795b003472e9b6f27e9f80004bf4f3e4"
EXPERTS = {7: [16, 38, 41, 50], 8: [8, 25, 33, 55]}


def validate(result, layer, expert, tokens):
    assert (result["layer"], result["logical_expert_id"], result["num_tokens"]) == (layer, expert, tokens)
    assert result["gate_shape"] == result["up_shape"] == [1408, 2048]
    assert result["down_shape"] == [2048, 1408]
    assert result["finite_cpu"] and result["finite_npu"]
    assert result["cpu_repeats"] >= 10
    assert len(result["cpu_repeat_unique_hashes"]) == 1 and result["cpu_repeat_max_abs"] == 0
    assert all(math.isfinite(v) for v in result["metrics"].values())
    assert 0 <= result["metrics"]["relative_l2"] <= 0.01, "EXPERT_NUMERICAL"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert hashlib.sha256(args.probe.read_bytes()).hexdigest() == PROBE_SHA
    args.output.mkdir(exist_ok=False, parents=True)
    report = dict(status="RUNNING", started_unix=time.time(), probe_sha256=PROBE_SHA,
                  safety_gate_relative_l2=0.01, cases=[])
    try:
        for layer, experts in EXPERTS.items():
            for expert in experts:
                for tokens in (1, 4):
                    name = f"L{layer}-E{expert}-T{tokens}"
                    target = args.output / f"{name}.json"
                    command = [sys.executable, str(args.probe),
                               "--model-dir", "/workspace/models/DeepSeek-V2-Lite-604d5664",
                               "--gguf", "/workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf",
                               "--layer", str(layer), "--expert", str(expert),
                               "--num-tokens", str(tokens), "--cpu-repeats", "10",
                               "--cpuinfer-threads", "16", "--output", str(target)]
                    with (args.output / f"{name}.log").open("x") as log:
                        subprocess.run(command, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
                    result = json.loads(target.read_text())
                    validate(result, layer, expert, tokens)
                    report["cases"].append(dict(case=name, metrics=result["metrics"],
                                                artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
                    (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(name, "PASS", result["metrics"], flush=True)
        report["status"] = "EXPERT_SAFETY_PASS"
    except BaseException as exc:
        report.update(status="FAILED", exception=repr(exc))
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
