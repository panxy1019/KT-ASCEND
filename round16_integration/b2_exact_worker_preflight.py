"""Read-only exact donor worker preflight for the Round16-B B2 hook.

This script performs bounded source reads only.  It imports no torch, vLLM,
vLLM-Ascend, checkpoint, or model code and never touches an accelerator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from round16.b.exact_donor_worker_hook import (
    EXACT_WORKER_SOURCE_SHA256,
    validate_exact_worker_source_identity,
)
from round16_integration.p1_allgather_q32_probe import (
    load_source_preflight_token,
)


B2_WORKER_PREFLIGHT_API = "round16-integration:b2-exact-worker-preflight:v1"

SOURCE_PATHS = {
    "worker.py": "worker/worker.py",
    "model_runner_v1.py": "worker/model_runner_v1.py",
    "qwen4_exp/model.py": "models/qwen4_exp/model.py",
    "qwen4_exp/nvidia/model.py": "models/qwen4_exp/nvidia/model.py",
    "fused_moe.py": "ops/fused_moe/fused_moe.py",
    "ascend_forward_context.py": "ascend_forward_context.py",
}

REQUIRED_SOURCE_MARKERS = {
    "worker.py": (
        "class NPUWorker(WorkerBase):",
        "with context, set_current_vllm_config(self.vllm_config):",
        "self.model_runner.load_model()",
    ),
    "model_runner_v1.py": (
        "class NPUModelRunner(GPUModelRunner):",
        "self.model: nn.Module = get_model(vllm_config=self.vllm_config)",
        "with set_ascend_forward_context(",
    ),
    "qwen4_exp/model.py": (
        "class AscendQwen4ExpForConditionalGeneration(",
        "upstream_model.Qwen4ExpModel = AscendQwen4ExpModel",
    ),
    "qwen4_exp/nvidia/model.py": (
        "_qwen3_next.FusedMoE = _ascend_FusedMoE",
        "class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):",
        "loader = AutoWeightsLoader(",
    ),
    "fused_moe.py": (
        "class AscendMoERunner(MoERunner):",
        "def no_shared_forward_impl(",
        "prepare_output = _EXTRA_CTX.moe_comm_method.prepare(",
        "routed_out = _EXTRA_CTX.moe_comm_method.finalize(",
        "forward_context = get_forward_context()",
    ),
    "ascend_forward_context.py": (
        "def set_ascend_forward_context(",
        "forward_context.moe_comm_method = get_moe_comm_method(",
        "forward_context.moe_comm_type = moe_comm_type",
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_worker_api_shapes(name: str, source: str) -> None:
    if name not in REQUIRED_SOURCE_MARKERS:
        raise ValueError(f"unknown exact worker source {name}")
    try:
        ast.parse(source, filename=name)
    except SyntaxError as error:
        raise ValueError(f"exact worker source {name} is not valid Python") from error
    missing = tuple(
        marker for marker in REQUIRED_SOURCE_MARKERS[name] if marker not in source
    )
    if missing:
        raise ValueError(f"exact worker source {name} lacks API markers: {missing}")


def run_preflight(
    *,
    vllm_ascend_root: str | Path,
    base_source_receipt: str | Path,
) -> dict[str, Any]:
    root = Path(vllm_ascend_root).resolve()
    if not root.is_dir():
        raise ValueError("vllm_ascend source root is unavailable")
    observed: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, relative in SOURCE_PATHS.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("exact worker source path escapes donor root") from error
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"exact worker source {name} is missing or too large")
        data = path.read_bytes()
        observed[name] = _sha256(data)
        try:
            sources[name] = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"exact worker source {name} is not UTF-8") from error
    # Hash identity is checked before API claims are accepted.
    worker_token = validate_exact_worker_source_identity(observed)
    for name, source in sources.items():
        validate_worker_api_shapes(name, source)
    base_token = load_source_preflight_token(base_source_receipt)
    return {
        "schema": 1,
        "status": "PASS_B2_EXACT_WORKER_SOURCE",
        "passed": True,
        "live_eligible": False,
        "device_executed": False,
        "api_identity": B2_WORKER_PREFLIGHT_API,
        "observed_sha256": observed,
        "api_checks": list(SOURCE_PATHS),
        "worker_source_binding_sha256": worker_token.binding_sha256,
        "base_source_binding_sha256": base_token.binding_sha256,
        "resolved_entrypoints": [
            "NPUWorker.load_model -> NPUModelRunner.load_model under set_current_vllm_config",
            "NPUModelRunner.load_model -> get_model before post-load registration",
            "Qwen4ExpSparseMoeBlock -> Qwen3NextSparseMoeBlock -> module-global FusedMoE",
            "qwen3_next.FusedMoE exact binding -> _ascend_FusedMoE",
            "forward context -> current moe_comm_method; stream remains lazy runtime input",
        ],
        "required_failure_gate": "SELECTIVE_Q32_PREALLOCATION_REQUIRED_UNTIL_LIVE_OWNER_ACK",
        "not_run": [
            "NPU",
            "model load",
            "checkpoint read",
            "C128 bank",
            "device numerical parity",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-ascend-root", required=True, type=Path)
    parser.add_argument("--base-source-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    _write_json(
        parsed.output,
        run_preflight(
            vllm_ascend_root=parsed.vllm_ascend_root,
            base_source_receipt=parsed.base_source_receipt,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
