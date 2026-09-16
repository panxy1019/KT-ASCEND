#!/workspace/results/venv-r6/bin/python
"""Test-only interpreter adapter for the unchanged frozen serving launcher.

KT_R6_PYTHON points here. Change only token capacity and explicitly qualified
batch cap, write resolved argv, then exec the real interpreter. No code patches.
"""
import json
import os
from pathlib import Path
import sys


def adapt(argv, batch):
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
    return result


def main():
    batch = int(os.environ["KT_R64_BATCH_CAP"])
    argv = adapt(sys.argv[1:], batch)
    record = Path(os.environ["KT_R64_RESOLVED_COMMAND"])
    if not str(record).startswith("/workspace/results/r64-"):
        raise ValueError("DEDICATED_COMMAND_RECORD_REQUIRED")
    with record.open("x") as stream:
        json.dump(dict(profile="round6.4-capacity-v1", batch_cap=batch,
                       original=sys.argv[1:], resolved=argv,
                       capacity_only=True, production_code_changed=False), stream, indent=2)
    # CPython uses argv[0] to locate pyvenv.cfg. A bare "python" can select
    # system prefixes even when execv's executable is the virtualenv symlink.
    interpreter = "/workspace/results/venv-r6/bin/python"
    os.execv(interpreter, [interpreter, *argv])


if __name__ == "__main__":
    main()
