"""Run the frozen Plan10 M3 baseline request schedule after separate authorization."""

import argparse
import hashlib
import json
import math
import pathlib
import signal
import time
import urllib.error
import urllib.request


HERE = pathlib.Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "M3_CANARY_CONTRACT.json"
CONTRACT_FIXTURE = json.loads(CONTRACT_PATH.read_text())
PER_REQUEST_TIMEOUT_SECONDS = 1800
TOTAL_DRIVER_TIMEOUT_SECONDS = 21600


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def encode_prompt(tokenizer, prompt):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids or len(ids) > 256 or not all(isinstance(item, int) for item in ids):
        raise ValueError(f"invalid encoded input length/type: {len(ids)}")
    return {"prompt": prompt, "rendered_chat": rendered, "input_ids": ids}


def build_medium_input(tokenizer, specification):
    base = specification["base_prompt"].strip()
    words = specification["padding_unit"].strip().split()
    candidates = [encode_prompt(tokenizer, base)]
    current = base
    overflow = False
    for _ in range(32):
        for word in words:
            current += " " + word
            try:
                candidates.append(encode_prompt(tokenizer, current))
            except ValueError as error:
                if "invalid encoded input length/type" not in str(error):
                    raise
                overflow = True
                break
        if overflow:
            break
    allowed = specification["allowed_actual_input_tokens"]
    for target in allowed:
        for encoded in candidates:
            if len(encoded["input_ids"]) == target:
                encoded["target_input_tokens"] = target
                return encoded
    observed = sorted({len(encoded["input_ids"]) for encoded in candidates})
    raise ValueError(f"could not form legitimate medium input at {allowed}; observed={observed}")


def request_body(request_id, input_ids, max_new_tokens, ignore_eos):
    return {
        "rid": request_id,
        "input_ids": list(input_ids),
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def freeze_inputs(tokenizer, contract):
    canaries = []
    schedule = []
    for item in contract["canaries"]:
        frozen = encode_prompt(tokenizer, item["prompt"])
        frozen.update({"id": item["id"], "validator": item["validator"]})
        canaries.append(frozen)
        schedule.append(
            {
                "phase": "canary",
                "case_id": item["id"],
                "validator": item["validator"],
                "input_ids": frozen["input_ids"],
            }
        )
    short = encode_prompt(tokenizer, contract["workloads"]["short"]["prompt"])
    medium = build_medium_input(tokenizer, contract["workloads"]["medium"])
    workloads = {"short": short, "medium": medium}
    for workload in ("short", "medium"):
        schedule.append(
            {
                "phase": "warmup",
                "workload": workload,
                "repeat": 0,
                "input_ids": workloads[workload]["input_ids"],
            }
        )
    for workload in ("short", "medium"):
        for repeat in range(1, 4):
            schedule.append(
                {
                    "phase": "measurement",
                    "workload": workload,
                    "repeat": repeat,
                    "input_ids": workloads[workload]["input_ids"],
                }
            )
    if len(schedule) != contract["request_budget"]["generation_requests_per_mode"]:
        raise AssertionError("schedule does not match generation budget")
    for index, row in enumerate(schedule, start=1):
        suffix = row.get("case_id")
        if suffix is None:
            suffix = f'{row["workload"]}-{row["repeat"]}'
        row["request_id"] = f"r9-plan10-m3-baseline-{index:02d}-{suffix}"
        is_canary = row["phase"] == "canary"
        row["request"] = request_body(
            row["request_id"], row["input_ids"], 32 if is_canary else 4, not is_canary
        )
    frozen = {"canaries": canaries, "workloads": workloads, "schedule": schedule}
    frozen["encoded_inputs_sha256"] = sha256_bytes(
        canonical_bytes([row["request"]["input_ids"] for row in schedule])
    )
    return frozen


def output_ids(response, logs):
    ids = response.get("output_ids")
    if ids is None:
        ids = [item[1] for item in logs]
    return ids


def finish_type(finish_reason):
    return finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason


def validate_response(schedule_row, response):
    if not isinstance(response, dict):
        raise RuntimeError("response is not an object")
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("missing meta_info")
    logs = meta.get("output_token_logprobs")
    if not isinstance(logs, list) or not logs:
        raise RuntimeError("missing output token logprobs")
    try:
        finite = all(math.isfinite(float(item[0])) for item in logs)
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("malformed output token logprobs") from error
    if not finite:
        raise RuntimeError("nonfinite output token logprob")
    ids = output_ids(response, logs)
    completion_tokens = meta.get("completion_tokens")
    if not isinstance(ids, list) or not ids or len(ids) != len(logs):
        raise RuntimeError("output ids/logprobs are empty or different lengths")
    if completion_tokens != len(ids):
        raise RuntimeError("completion_tokens does not match output length")
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("empty output text")
    finish_reason = meta.get("finish_reason")
    kind = finish_type(finish_reason)
    if not kind or kind == "abort":
        raise RuntimeError(f"invalid terminal finish_reason: {finish_reason!r}")
    if schedule_row["phase"] == "canary":
        if kind == "length":
            raise RuntimeError("finish_reason length cannot pass a canary")
        validator = schedule_row["validator"]
        stripped = text.strip()
        if validator["kind"] == "exact_text":
            passed = text == validator["expected"]
        elif validator["kind"] == "exact_json":
            try:
                passed = json.loads(stripped) == validator["expected"]
            except json.JSONDecodeError:
                passed = False
        else:
            raise RuntimeError(f'unknown validator: {validator["kind"]}')
        if not passed:
            raise RuntimeError(f"frozen canary mismatch: {stripped!r}")
    else:
        if kind != "length" or completion_tokens != 4:
            raise RuntimeError("performance request did not terminate at exactly four tokens")
    return {
        "finite_output_logprobs": True,
        "output_ids": ids,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "canary_passed": True if schedule_row["phase"] == "canary" else None,
    }


def write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_exact_manifest(path, expected, label):
    raw = path.read_bytes()
    actual = json.loads(raw)
    if actual != expected:
        raise ValueError(f"{label} drift: expected exact contract object")
    return {"path": str(path.resolve()), "sha256": sha256_bytes(raw), "value": actual}


def tokenizer_evidence(model_path, tokenizer):
    files = {}
    for name in CONTRACT_FIXTURE["tokenization"]["required_hash_files"]:
        path = model_path / name
        files[name] = sha256_bytes(path.read_bytes())
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer has no actual chat_template")
    return {
        "files_sha256": files,
        "chat_template_sha256": sha256_bytes(template.encode()),
        "chat_template": template,
    }


def send_once(endpoint, body, timeout):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return raw


def timeout_handler(_signum, _frame):
    raise TimeoutError("21600 second total driver timeout")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--launch-profile", type=pathlib.Path, required=True)
    parser.add_argument("--authorization", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=["baseline"], default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")
    started_monotonic = time.monotonic()
    report = {
        "schema": "round9.plan10.m3-request-result.v1",
        "status": "PREFLIGHT",
        "scope": CONTRACT_FIXTURE["scope"],
        "mode": args.mode,
        "contract_path": str(CONTRACT_PATH),
        "contract_sha256": sha256_bytes(CONTRACT_PATH.read_bytes()),
        "started_unix": time.time(),
        "requests": [],
        "requests_recorded": 0,
        "requests_completed": 0,
        "request_budget": 13,
        "retries": 0,
        "hidden_requests": 0,
        "profiler": False,
        "claims": CONTRACT_FIXTURE["claims"],
    }
    write_report(args.output, report)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TOTAL_DRIVER_TIMEOUT_SECONDS)
    try:
        report["authorization"] = read_exact_manifest(
            args.authorization,
            CONTRACT_FIXTURE["execution_state"]["authorization_manifest"],
            "execution authorization",
        )
        report["launch_profile"] = read_exact_manifest(
            args.launch_profile, CONTRACT_FIXTURE["required_launch_profile"], "launch profile"
        )
        model_path = pathlib.Path(CONTRACT_FIXTURE["model"]["local_path"])
        if model_path.name != CONTRACT_FIXTURE["model"]["revision"]:
            raise ValueError("model revision/path drift")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        report["tokenizer"] = tokenizer_evidence(model_path, tokenizer)
        frozen = freeze_inputs(tokenizer, CONTRACT_FIXTURE)
        report["frozen_inputs"] = frozen
        report["status"] = "INPUTS_FROZEN_BEFORE_FIRST_GENERATION"
        write_report(args.output, report)

        for schedule_row in frozen["schedule"]:
            if report["requests_recorded"] >= report["request_budget"]:
                raise RuntimeError("generation request budget exhausted")
            record = {
                "request_id": schedule_row["request_id"],
                "phase": schedule_row["phase"],
                "workload": schedule_row.get("workload"),
                "repeat": schedule_row.get("repeat"),
                "case_id": schedule_row.get("case_id"),
                "state": "RECORDED_CONSUMES_BUDGET_BEFORE_SEND",
                "request": schedule_row["request"],
                "recorded_unix": time.time(),
                "raw_response": None,
                "parsed_response": None,
                "error": None,
                "client_e2e_seconds": None,
                "ttft_seconds": "UNKNOWN",
                "tpot_seconds": "UNKNOWN",
            }
            report["requests"].append(record)
            report["requests_recorded"] += 1
            write_report(args.output, report)
            remaining = TOTAL_DRIVER_TIMEOUT_SECONDS - (time.monotonic() - started_monotonic)
            if remaining <= 0:
                raise TimeoutError("total driver timeout before request send")
            request_started = time.monotonic()
            try:
                raw = send_once(
                    CONTRACT_FIXTURE["endpoint"],
                    schedule_row["request"],
                    min(PER_REQUEST_TIMEOUT_SECONDS, remaining),
                )
                record["raw_response"] = raw
                response = json.loads(raw)
                record["parsed_response"] = response
                record["validation"] = validate_response(schedule_row, response)
                record["state"] = "PASS"
                report["requests_completed"] += 1
            except BaseException as error:
                record["state"] = "FAIL"
                record["error"] = repr(error)
                if isinstance(error, urllib.error.HTTPError):
                    try:
                        record["raw_response"] = error.read().decode("utf-8", "replace")
                    except BaseException as read_error:
                        record["http_error_body_read_error"] = repr(read_error)
                raise
            finally:
                record["client_e2e_seconds"] = time.monotonic() - request_started
                write_report(args.output, report)

        measurements = {}
        for workload in ("short", "medium"):
            rows = [
                row
                for row in report["requests"]
                if row["phase"] == "measurement" and row["workload"] == workload
            ]
            measurements[workload] = {
                "actual_input_tokens": len(frozen["workloads"][workload]["input_ids"]),
                "client_e2e_seconds": [row["client_e2e_seconds"] for row in rows],
                "ttft_seconds": "UNKNOWN",
                "tpot_seconds": "UNKNOWN",
                "repetitions": 3,
            }
        report["measurements"] = measurements
        report["status"] = "PASS_NARROW_BASELINE_CANARIES_AND_RAW_REPETITIONS"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        raise
    finally:
        signal.alarm(0)
        report["ended_unix"] = time.time()
        report["client_total_seconds"] = time.monotonic() - started_monotonic
        write_report(args.output, report)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
