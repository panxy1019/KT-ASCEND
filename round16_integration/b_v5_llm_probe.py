"""First full-model eager probe for the exact Round16-B Hybrid worker."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    try:
        # The engine core is forked by this frozen vLLM build.  Registering
        # before constructing LLM gives both parent and child an explicit,
        # non-terminating stack-dump signal for bounded loader diagnostics.
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        from vllm import LLM, SamplingParams

        print("ROUND16_B_V5_ENGINE_CONSTRUCT_BEGIN", flush=True)
        construct_start = time.monotonic()
        llm = LLM(
            model=args.model,
            worker_cls=(
                "round16_integration.b2_live_worker."
                "Round16BHybridWorker"
            ),
            tensor_parallel_size=1,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.82,
            # The target host/driver fails devmm page pinning for sufficiently
            # large H2D copies whose source is a lazy safetensors mmap
            # (Get_user_pages_fast ret=-14).  The frozen vLLM loader's eager
            # strategy materializes one checkpoint shard at a time in regular
            # Host storage and preserves the checkpoint/model semantics.
            safetensors_load_strategy="eager",
            disable_log_stats=True,
            seed=20260929,
        )
        construct_seconds = time.monotonic() - construct_start
        print(
            f"ROUND16_B_V5_ENGINE_CONSTRUCT_PASS seconds={construct_seconds:.3f}",
            flush=True,
        )
        request_start = time.monotonic()
        outputs = llm.generate(
            ["Reply with exactly ACK."],
            SamplingParams(
                temperature=0.0,
                max_tokens=4,
                ignore_eos=False,
                logprobs=1,
            ),
            use_tqdm=False,
        )
        request_seconds = time.monotonic() - request_start
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("full-model canary returned an unexpected result count")
        result = outputs[0].outputs[0]
        token_ids = list(result.token_ids)
        if not token_ids:
            raise RuntimeError("full-model canary produced no output token")
        payload = {
            "schema": 1,
            "status": "PASS_ROUND16_B_V5_FULL_MODEL_CANARY",
            "started_at": started,
            "ended_at": time.time(),
            "construct_seconds": construct_seconds,
            "request_seconds": request_seconds,
            "prompt": "Reply with exactly ACK.",
            "text": result.text,
            "token_ids": token_ids,
            "finish_reason": result.finish_reason,
            "worker_cls": (
                "round16_integration.b2_live_worker."
                "Round16BHybridWorker"
            ),
            "enable_b2": os.environ.get("ROUND16_B_ENABLE_B2"),
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "safetensors_load_strategy": "eager",
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        payload = {
            "schema": 1,
            "status": "FAIL_ROUND16_B_V5_FULL_MODEL_CANARY",
            "started_at": started,
            "ended_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
