#!/workspace/results/venv-r6/bin/python
"""Test-only interpreter adapter for the unchanged frozen serving launcher.

KT_R6_PYTHON points here. Change only token capacity and explicitly qualified
batch cap, write resolved argv, then exec the real interpreter. No code patches.
"""
import json
import hashlib
import os
from pathlib import Path
import sys


def adapt(argv, batch, performance=False):
    if batch not in (1, 2, 4) or argv[:2] != ["-m", "sglang.launch_server"]:
        raise ValueError("UNSUPPORTED_INVOCATION")
    expected = {"--context-length": "512", "--max-total-tokens": "512",
                "--chunked-prefill-size": "512", "--max-prefill-tokens": "512",
                "--max-running-requests": "1"}
    replacements = {"--context-length": "1024", "--max-total-tokens": "4096",
                    "--chunked-prefill-size": "1024", "--max-prefill-tokens": "1024",
                    "--max-running-requests": str(batch)}
    result = list(argv)
    for flag, value in expected.items():
        if result.count(flag) != 1 or result[result.index(flag)+1] != value:
            raise ValueError(f"FROZEN_LAUNCHER_DRIFT:{flag}")
        result[result.index(flag)+1] = replacements[flag]
    if "--disable-cuda-graph" not in result or result[result.index("--tp-size")+1] != "1":
        raise ValueError("SCOPE_DRIFT")
    if performance:
        if any(flag in result for flag in ("--expert-distribution-recorder-mode", "--debug-tensor-dump-output-folder", "--enable-return-routed-experts")):
            raise ValueError("PERFORMANCE_OBSERVER_ENABLED")
        # The old launcher couples cache-OFF to route recording. Keep cache-OFF
        # explicitly when recording is disabled so timing changes only observers.
        if "--disable-radix-cache" not in result:
            result.append("--disable-radix-cache")
    return result


def main():
    batch = int(os.environ["KT_R64_BATCH_CAP"])
    performance = os.environ.get("KT_R64_PERFORMANCE", "0") == "1"
    argv = adapt(sys.argv[1:], batch, performance=performance)
    compat = os.environ.get("KT_R64_TOKENIZER_COMPAT", "0") == "1"
    if compat:
        directory = Path("/workspace/results/r64-tokenizer-compat-v1")
        expected = {"tokenizer.json": "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
                    "tokenizer_config.json": "d5f57d20f87e97e94dac06c5f91871bc819fe3fff41b2b0c96c04c2622dc95bb"}
        for name, digest in expected.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError("TOKENIZER_COMPAT_ARTIFACT_DRIFT:"+name)
        if "--tokenizer-path" in argv:
            raise ValueError("UNEXPECTED_EXISTING_TOKENIZER_OVERRIDE")
        argv += ["--tokenizer-path", str(directory)]
    record = Path(os.environ["KT_R64_RESOLVED_COMMAND"])
    if not str(record).startswith("/workspace/results/r64-"):
        raise ValueError("DEDICATED_COMMAND_RECORD_REQUIRED")
    with record.open("x") as stream:
        json.dump(dict(profile="round6.4-capacity-v1", batch_cap=batch,
                       original=sys.argv[1:], resolved=argv,
                       capacity_only=not performance and not compat, performance_observer_off=performance,
                       tokenizer_compatibility_overlay=compat,
                       cache_disabled="--disable-radix-cache" in argv, production_code_changed=False), stream, indent=2)
    # CPython uses argv[0] to locate pyvenv.cfg. A bare "python" can select
    # system prefixes even when execv's executable is the virtualenv symlink.
    interpreter = "/workspace/results/venv-r6/bin/python"
    os.execv(interpreter, [interpreter, *argv])


if __name__ == "__main__":
    main()
