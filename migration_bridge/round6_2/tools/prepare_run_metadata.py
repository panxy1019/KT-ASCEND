"""Collect actual container source provenance; never infer deployed HEAD from local HEAD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/kt-src"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["defer0", "defer1"], required=True)
    p.add_argument("--artifact", type=Path, action="append", default=[])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse metadata overwrite")
    import torch
    import torch_npu
    import kt_kernel.experts_base as experts
    child = args.root / "third_party/sglang"
    paths = [Path(experts.__file__),
             child / "python/sglang/srt/debug_utils/tensor_dump_forward_hook.py",
             child / "python/sglang/srt/debug_utils/round6_1_deferred.py",
             args.root / "migration_bridge/round6/tools/launch_round6_profile.sh",
             *args.artifact]
    metadata = dict(
        mode=args.mode, provenance_type="actual_checkout_plus_explicit_source_overlay",
        parent_checkout_sha=git(args.root, "rev-parse", "HEAD"),
        sglang_checkout_sha=git(child, "rev-parse", "HEAD"),
        parent_status=git(args.root, "status", "--short"),
        sglang_status=git(child, "status", "--short"),
        file_sha256={str(path): sha(path) for path in paths},
        python=sys.version, torch=torch.__version__, torch_npu=torch_npu.__version__,
        platform=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        environment={name: os.environ.get(name) for name in (
            "ASCEND_RT_VISIBLE_DEVICES", "SGLANG_APPLY_CONFIG_BACKUP",
            "KT_R62_COMPLETE_FORWARD_DUMP", "KT_R62_REFERENCE_LOGITS_ONLY",
            "KT_R6_ENABLE_ROUTE_RECORDER")},
        model_revision="604d5664dddd88a0433dbae533b7fe9472482de0",
        shared_npu=True, host_libraries_modified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output))))


if __name__ == "__main__":
    main()
