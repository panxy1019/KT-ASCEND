"""One preregistered ABBA phase, observer-OFF and cache-OFF at fixed batch1."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
from owned_server import OwnedServer
from stream_timing import request
from run_compact_controls import checked

MANIFEST_RAW_SHA = "b383aa539815f8b6931446709163accec617fb809160223ee0e8b310dd0e8693"
FAST_MANIFEST_SHA = "257034644e2a3ef516722d354be4787965a5c937205bf73eb629ace57f42c19f"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--profile", choices=("D1","D1_NODEFER"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    assert digest in (MANIFEST_RAW_SHA, FAST_MANIFEST_SHA)
    manifest = checked(args.manifest)
    assert manifest["primary_batch"] == 1 and manifest["paired_blocks"] == (3 if digest == FAST_MANIFEST_SHA else 5)
    shapes = {r["id"]:r for r in manifest["shapes"]}
    if args.output.exists():
        raise ValueError("REFUSE_PERFORMANCE_OVERWRITE")
    result = dict(status="RUNNING", profile=args.profile, manifest_sha256=digest,
                  started_unix=time.time(), rows=[], warmups=[], observer=False, route_recorder=False, cache="OFF")
    def cancel(signum, frame):
        raise InterruptedError("OWNED_PERFORMANCE_CANCELLED")
    signal.signal(signal.SIGTERM,cancel)
    try:
        with OwnedServer(args.output,args.profile,batch=1,performance=True,tokenizer_compat=True):
            resolved = json.loads((args.output/"resolved-command.json").read_text())
            assert resolved["performance_observer_off"] and resolved["cache_disabled"]
            assert not list((args.output/"tensors").iterdir())
            warmup = shapes[manifest["warmup_shape"]]
            for i in range(manifest["warmup_requests_per_phase"]):
                row = request(f"r64-perf-warmup-{i}",warmup["input_ids"],warmup["output_tokens"])
                result["warmups"].append(dict(seconds=row["seconds"],included_in_timing=False))
            with (args.output/"requests.jsonl").open("x") as log:
                for shape in manifest["shapes"]:
                    row = request("r64-perf-"+shape["id"],shape["input_ids"],shape["output_tokens"])
                    assert row["response"]["meta_info"]["prompt_tokens"] == shape["input_tokens"]
                    log.write(json.dumps(dict(shape=shape["id"],**row))+"\n")
                    log.flush()
                    result["rows"].append(dict(shape=shape["id"],**{k:v for k,v in row.items() if k != "response"}))
                    print(json.dumps(dict(shape=shape["id"],seconds=row["seconds"],ttft=row["ttft_seconds"])),flush=True)
            assert not list((args.output/"tensors").iterdir()), "PERFORMANCE_CAPTURE_CONTAMINATION"
        result["status"] = "PERFORMANCE_PHASE_COMPLETE"
    except BaseException as exc:
        result.update(status="FAILED",exception=repr(exc))
        raise
    finally:
        result["ended_unix"] = time.time()
        if args.output.exists():
            (args.output/"performance-report.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
