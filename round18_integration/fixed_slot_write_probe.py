"""Parameter gate for a later real Ascend fixed-slot write probe.

The offline command prints or validates the exact inputs required by the
device operator.  It deliberately performs no torch import, NPU call, model
start, allocation, or performance measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from round16.b.resident_packed_loader import RESIDENT_PACKED_LOADER_API
from round16.b.source_bound_q32_adapter import (
    SOURCE_BOUND_Q32_ROUTED_EXPERTS_API,
)
from round18.fixed_slot_write import (
    ELIGIBLE_BACKING_FORMAT,
    ELIGIBLE_SOURCE_FORMAT,
)


FIXED_SLOT_WRITE_PROBE_PARAMETERS_API = (
    "round18:fixed-slot-write-probe-parameters:v1"
)
_REQUIRED_KEYS = (
    "device",
    "logical_expert_id",
    "physical_slot",
    "generation",
    "source_dtype",
    "source_format",
    "destination_dtype",
    "destination_backing",
    "owner_api",
    "loader_api",
)
_BF16_DTYPES = frozenset(("bf16", "bfloat16", "torch.bfloat16"))


def device_probe_parameter_schema() -> dict[str, Any]:
    """Describe the minimum real-device binding without claiming a run."""

    return {
        "api": FIXED_SLOT_WRITE_PROBE_PARAMETERS_API,
        "logical_experts": 512,
        "physical_slots": 32,
        "source_projections": {
            "gate": [640, 2560],
            "up": [640, 2560],
            "down": [2560, 640],
        },
        "source_format": ELIGIBLE_SOURCE_FORMAT,
        "eligible_backing": ELIGIBLE_BACKING_FORMAT,
        "ineligible_backings": ["W8A8", "PREPROCESSED"],
        "owner_api": SOURCE_BOUND_Q32_ROUTED_EXPERTS_API,
        "loader_api": RESIDENT_PACKED_LOADER_API,
        "required_parameters": list(_REQUIRED_KEYS),
        "minimum_runtime_requirements": (
            "frozen Round18 Ascend image and source identities; one existing "
            "mutable BF16 Physical-Q32 owner; exact w13 gate/up subviews and "
            "w2 slot view; one complete BF16 checkpoint expert; native "
            "orientation/layout copy callbacks; torch.npu.synchronize (or an "
            "equivalent producer-to-consumer event wait) before generation "
            "publication; consumer-side exact-generation acquire gate"
        ),
        "required_correctness_cases": [
            "complete gate/up/down replacement",
            "injected second-projection failure remains unpublished",
            "stale and skipped generation rejection",
            "same-generation full rewrite after failure",
            "post-barrier readback against the source expert",
        ],
        "performance_claim": None,
    }


def validate_device_probe_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed before a future runner imports torch or touches the NPU."""

    if not isinstance(parameters, Mapping):
        raise TypeError("probe parameters must be a mapping")
    missing = [key for key in _REQUIRED_KEYS if key not in parameters]
    if missing:
        raise ValueError(f"probe parameters are missing: {', '.join(missing)}")
    result = dict(parameters)
    if type(result["device"]) is not str or not result["device"].startswith("npu:"):
        raise ValueError("real fixed-slot probe device must be an explicit npu:<id>")
    if type(result["logical_expert_id"]) is not int or not (
        0 <= result["logical_expert_id"] < 512
    ):
        raise ValueError("logical_expert_id must be in [0,512)")
    if type(result["physical_slot"]) is not int or not (
        0 <= result["physical_slot"] < 32
    ):
        raise ValueError("physical_slot must be in [0,32)")
    if type(result["generation"]) is not int or result["generation"] <= 0:
        raise ValueError("generation must be a positive integer")
    if str(result["source_dtype"]).lower() not in _BF16_DTYPES:
        raise ValueError("source_dtype must be BF16")
    if str(result["destination_dtype"]).lower() not in _BF16_DTYPES:
        raise ValueError("destination_dtype must be BF16")
    if result["source_format"] != ELIGIBLE_SOURCE_FORMAT:
        raise ValueError("only dense BF16 checkpoint rows are qualified")
    if result["destination_backing"] != ELIGIBLE_BACKING_FORMAT:
        raise ValueError("W8A8 and preprocessed destination backing are ineligible")
    if result["owner_api"] != SOURCE_BOUND_Q32_ROUTED_EXPERTS_API:
        raise ValueError("frozen Physical-Q32 owner API identity mismatch")
    if result["loader_api"] != RESIDENT_PACKED_LOADER_API:
        raise ValueError("frozen resident loader API identity mismatch")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print or validate the offline Round18 fixed-slot probe schema"
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        help="validate a JSON parameter file; this command never touches an NPU",
    )
    args = parser.parse_args(argv)
    if args.parameters is None:
        payload: Any = device_probe_parameter_schema()
    else:
        payload = validate_device_probe_parameters(
            json.loads(args.parameters.read_text(encoding="utf-8"))
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FIXED_SLOT_WRITE_PROBE_PARAMETERS_API",
    "device_probe_parameter_schema",
    "main",
    "validate_device_probe_parameters",
)
