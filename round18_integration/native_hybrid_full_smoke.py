"""Round18 native-runner full-model eager quality, decode, cancel, and reuse probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


EXPECTED_CONTRACT_SHA256 = (
    "176887de85e37c68443eeb6f245765353a33c7d7b4d636d4b796d8f9b52bec48"
)
EXPECTED_FIXTURES_SHA256 = (
    "14b0157098fb766eaaa5527c516513133fffc7c3f583331c7b5653749678cc5e"
)
EXPECTED_TOKENIZER_FILES = {
    "tokenizer.json": (
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    ),
    "tokenizer_config.json": (
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def validate_known_answer(text: str, validator: Mapping[str, object]) -> None:
    kind = validator.get("kind")
    expected = validator.get("expected")
    if kind == "exact_text":
        if text != expected:
            raise RuntimeError(
                f"exact_text mismatch: expected {expected!r}, observed {text!r}"
            )
        return
    if kind == "exact_json":
        try:
            observed = json.loads(text.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"exact_json parse failed: {error}") from error
        if observed != expected:
            raise RuntimeError(
                f"exact_json mismatch: expected {expected!r}, observed {observed!r}"
            )
        return
    raise RuntimeError(f"unsupported canary validator: {kind!r}")


def extract_finite_selected_logprobs(
    token_ids: Sequence[int], logprobs: Sequence[Mapping[int, object]] | None
) -> list[float]:
    if logprobs is None or len(logprobs) != len(token_ids):
        raise RuntimeError(
            "output token/logprob count mismatch: "
            f"tokens={len(token_ids)}, logprobs={0 if logprobs is None else len(logprobs)}"
        )
    selected: list[float] = []
    for index, (token_id, candidates) in enumerate(zip(token_ids, logprobs)):
        if token_id not in candidates:
            raise RuntimeError(
                f"selected token {token_id} is absent from logprobs at index {index}"
            )
        value = float(getattr(candidates[token_id], "logprob"))
        if not math.isfinite(value):
            raise RuntimeError(
                f"selected token logprob is not finite at index {index}: {value}"
            )
        selected.append(value)
    return selected


def freeze_canary_inputs(tokenizer: Any, contract: Mapping[str, object]) -> list[dict]:
    frozen: list[dict] = []
    for case in contract["canaries"]:  # type: ignore[index]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError(f"canary {case['id']} produced no rendered chat")
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not isinstance(input_ids, list) or not input_ids:
            raise RuntimeError(f"canary {case['id']} produced no frozen input IDs")
        frozen.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "validator": case["validator"],
                "input_ids": [int(value) for value in input_ids],
            }
        )
    return frozen


def _serialize_output(request_output: Any) -> dict[str, object]:
    if len(request_output.outputs) != 1:
        raise RuntimeError("request returned an unexpected completion count")
    completion = request_output.outputs[0]
    token_ids = [int(value) for value in completion.token_ids]
    if not token_ids:
        raise RuntimeError("request returned no output tokens")
    selected = extract_finite_selected_logprobs(token_ids, completion.logprobs)
    return {
        "request_id": str(request_output.request_id),
        "text": completion.text,
        "token_ids": token_ids,
        "selected_logprobs": selected,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
    }


def _generate_one(llm: Any, sampling_params: Any, input_ids: Sequence[int]) -> dict:
    started = time.monotonic()
    outputs = llm.generate(
        [{"prompt_token_ids": list(input_ids)}],
        sampling_params,
        use_tqdm=False,
    )
    if len(outputs) != 1:
        raise RuntimeError(f"request returned {len(outputs)} results instead of one")
    serialized = _serialize_output(outputs[0])
    serialized["request_seconds"] = time.monotonic() - started
    return serialized


def _run_cancel(engine: Any, sampling_params: Any, input_ids: Sequence[int]) -> dict:
    request_id = "round18-native-hybrid-abort"
    engine.add_request(
        request_id,
        {"prompt_token_ids": list(input_ids)},
        sampling_params,
    )
    step_count = 0
    visible_tokens: list[int] = []
    while not visible_tokens and engine.has_unfinished_requests():
        step_count += 1
        if step_count > 64:
            raise RuntimeError("abort control did not expose a decode token within 64 steps")
        for request_output in engine.step():
            if str(request_output.request_id) != request_id:
                raise RuntimeError(
                    f"abort control observed an unexpected request {request_output.request_id}"
                )
            if request_output.outputs:
                visible_tokens = [int(x) for x in request_output.outputs[0].token_ids]
    if not visible_tokens:
        raise RuntimeError("abort control completed without exercising visible decode")
    if len(visible_tokens) >= int(sampling_params.max_tokens):
        raise RuntimeError("abort control finished before abort_request was issued")
    engine.abort_request([request_id])
    if engine.has_unfinished_requests():
        late_outputs = [
            str(output.request_id)
            for output in engine.step()
            if str(output.request_id) == request_id
        ]
        if late_outputs or engine.has_unfinished_requests():
            raise RuntimeError(
                "aborted request remained active or published a late completion"
            )
    return {
        "request_id": request_id,
        "step_count_before_abort": step_count,
        "visible_token_ids_before_abort": visible_tokens,
        "unfinished_after_abort": bool(engine.has_unfinished_requests()),
    }


def _sampling(SamplingParams: Any, *, max_tokens: int, ignore_eos: bool) -> Any:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=20260920,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        logprobs=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--graph-mode",
        choices=("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"),
        default="NONE",
    )
    args = parser.parse_args()
    started = time.time()
    try:
        import faulthandler

        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        if _sha256(args.contract) != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError("frozen canary contract SHA-256 mismatch")
        if _sha256(args.fixtures) != EXPECTED_FIXTURES_SHA256:
            raise RuntimeError("frozen threshold fixtures SHA-256 mismatch")
        tokenizer_hashes = {
            name: _sha256(args.model / name) for name in EXPECTED_TOKENIZER_FILES
        }
        if tokenizer_hashes != EXPECTED_TOKENIZER_FILES:
            raise RuntimeError(
                f"tokenizer source identity mismatch: {tokenizer_hashes!r}"
            )

        from transformers import AutoTokenizer

        contract = json.loads(args.contract.read_text())
        fixtures = json.loads(args.fixtures.read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True
        )
        canaries = freeze_canary_inputs(tokenizer, contract)
        arithmetic = next(item for item in canaries if item["id"] == "arithmetic")
        expected_arithmetic = fixtures["cases"]["canary27"]["input_ids"]
        if arithmetic["input_ids"] != expected_arithmetic:
            raise RuntimeError(
                "runtime template arithmetic IDs differ from the frozen canary27 fixture"
            )
        boundary64 = [
            int(value) for value in fixtures["cases"]["boundary64"]["input_ids"]
        ]
        if len(boundary64) != 64:
            raise RuntimeError("boundary64 fixture does not contain exactly 64 tokens")
        if any(len(case["input_ids"]) > 128 for case in canaries):
            raise RuntimeError("a frozen canary exceeds the Round18 input scope")

        # All tokenizer and request inputs are frozen before engine construction.
        from vllm import LLM, SamplingParams
        from vllm.config.compilation import (
            CUDAGraphMode,
            CompilationConfig,
            CompilationMode,
        )

        graph_mode = CUDAGraphMode[args.graph_mode]
        enforce_eager = graph_mode is CUDAGraphMode.NONE
        compilation_config = CompilationConfig(
            mode=(
                CompilationMode.NONE
                if enforce_eager
                else CompilationMode.VLLM_COMPILE
            ),
            cudagraph_mode=graph_mode,
            # Round18 qualifies serial Decode.  Keep the capture pool bounded
            # to the only live batch size instead of inheriting a broad set.
            cudagraph_capture_sizes=[] if enforce_eager else [1],
            max_cudagraph_capture_size=0 if enforce_eager else 1,
            compile_sizes=[] if enforce_eager else [1, 32],
            compile_ranges_endpoints=[32],
        )

        print("ROUND18_NATIVE_HYBRID_ENGINE_CONSTRUCT_BEGIN", flush=True)
        construct_started = time.monotonic()
        llm = LLM(
            model=str(args.model),
            worker_cls="round18_integration.native_hybrid_live_worker.Round18NativeHybridWorker",
            tensor_parallel_size=1,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.82,
            safetensors_load_strategy="eager",
            disable_log_stats=True,
            seed=20260929,
            compilation_config=compilation_config,
        )
        construct_seconds = time.monotonic() - construct_started

        canary_results: list[dict] = []
        arithmetic_result: dict | None = None
        for case in canaries:
            observed = _generate_one(
                llm,
                _sampling(SamplingParams, max_tokens=32, ignore_eos=False),
                case["input_ids"],
            )
            validate_known_answer(str(observed["text"]), case["validator"])
            if observed["finish_reason"] == "length":
                raise RuntimeError(f"canary {case['id']} ended due to length")
            record = {
                "id": case["id"],
                "input_tokens": len(case["input_ids"]),
                **observed,
            }
            canary_results.append(record)
            if case["id"] == "arithmetic":
                arithmetic_result = record

        decode32 = _generate_one(
            llm,
            _sampling(SamplingParams, max_tokens=32, ignore_eos=True),
            boundary64,
        )
        if len(decode32["token_ids"]) != 32 or decode32["finish_reason"] != "length":
            raise RuntimeError(
                "multi-step decode did not return exactly 32 tokens with length finish"
            )

        cancel = _run_cancel(
            llm.llm_engine,
            _sampling(SamplingParams, max_tokens=64, ignore_eos=True),
            boundary64,
        )
        reuse = _generate_one(
            llm,
            _sampling(SamplingParams, max_tokens=32, ignore_eos=False),
            arithmetic["input_ids"],
        )
        validate_known_answer(str(reuse["text"]), arithmetic["validator"])
        if arithmetic_result is None:
            raise RuntimeError("arithmetic reference result was not captured")
        if (
            reuse["text"] != arithmetic_result["text"]
            or reuse["token_ids"] != arithmetic_result["token_ids"]
        ):
            raise RuntimeError("same-process reuse differs from the first arithmetic request")

        payload = {
            "schema": 1,
            "status": (
                "PASS_ROUND18_NATIVE_HYBRID_EAGER_SMOKE"
                if enforce_eager
                else "PASS_ROUND18_NATIVE_HYBRID_GRAPH_SMOKE"
            ),
            "started_at": started,
            "ended_at": time.time(),
            "construct_seconds": construct_seconds,
            "worker_cls": "round18_integration.native_hybrid_live_worker.Round18NativeHybridWorker",
            "enable_native_hybrid": os.environ.get("ROUND18_ENABLE_NATIVE_HYBRID"),
            "binding_receipt_path": os.environ.get("ROUND18_BINDING_RECEIPT_PATH"),
            "tensor_parallel_size": 1,
            "enforce_eager": enforce_eager,
            "requested_graph_mode": args.graph_mode,
            "model_revision": contract["model"]["revision"],
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "fixtures_sha256": EXPECTED_FIXTURES_SHA256,
            "tokenizer_files_sha256": tokenizer_hashes,
            "frozen_inputs_before_engine": True,
            "canaries": canary_results,
            "decode32": {"input_tokens": 64, **decode32},
            "cancel": cancel,
            "reuse": {"input_tokens": len(arithmetic["input_ids"]), **reuse},
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        payload = {
            "schema": 1,
            "status": (
                "FAIL_ROUND18_NATIVE_HYBRID_EAGER_SMOKE"
                if args.graph_mode == "NONE"
                else "FAIL_ROUND18_NATIVE_HYBRID_GRAPH_SMOKE"
            ),
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
