#!/usr/bin/env python3
"""Same-image CPU semantic proof for the production OffloaderV1 tie helper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch.func import functional_call


CANDIDATE_SHA256 = (
    "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"
)


class Inner(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.register_buffer("scale", scale)


class TiedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        )
        self.register_buffer("scale", torch.tensor([11.0, 13.0]))
        self.inner = Inner(self.weight, self.scale)
        self.identity_observations = []

    def forward(self, value):
        self.identity_observations.append(
            {
                "parameter": self.weight is self.inner.weight,
                "buffer": self.scale is self.inner.scale,
            }
        )
        return (value @ self.weight + value @ self.inner.weight) * (
            self.scale + self.inner.scale
        )


def load_production_helper(source_path: Path):
    source = source_path.read_text()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != CANDIDATE_SHA256:
        raise RuntimeError(f"candidate source identity drifted: {digest}")
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_device_state_preserving_ties"
    ]
    if len(matches) != 1:
        raise RuntimeError("production tie helper count mismatch")
    module = ast.Module(body=matches, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_device_state_preserving_ties"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    helper = load_production_helper(args.source)
    module = TiedModule()
    value = torch.tensor([[17.0, 19.0]])
    reference = module(value)
    state = module.state_dict(keep_vars=True)

    full_state = {name: tensor.clone() for name, tensor in state.items()}
    default_full_error = None
    try:
        functional_call(module, full_state, (value,))
    except ValueError as error:
        default_full_error = str(error)
    if not default_full_error or "which are tied" not in default_full_error:
        raise RuntimeError("the fixture did not reproduce the tied-key failure")

    module.identity_observations.clear()
    tie_false = functional_call(
        module, full_state, (value,), tie_weights=False
    )
    tie_false_identity = list(module.identity_observations)
    if not torch.equal(tie_false, reference):
        raise RuntimeError("tie_weights=False changed this fixture's output")
    if all(
        row["parameter"] and row["buffer"] for row in tie_false_identity
    ):
        raise RuntimeError("tie_weights=False unexpectedly retained alias identity")

    module.identity_observations.clear()
    unique_state = helper(module, torch.device("cpu"))
    candidate = functional_call(module, unique_state, (value,))
    candidate_identity = list(module.identity_observations)
    if not torch.equal(candidate, reference):
        raise RuntimeError("tie-preserving candidate output mismatch")
    if not all(
        row["parameter"] and row["buffer"] for row in candidate_identity
    ):
        raise RuntimeError("tie-preserving candidate broke alias identity")
    if not (module.weight is module.inner.weight and module.scale is module.inner.scale):
        raise RuntimeError("functional_call did not restore module aliases")

    report = {
        "schema": 1,
        "status": "PASS_ROUND12A_TIED_SEMANTICS",
        "torch_version": torch.__version__,
        "functional_call_signature": str(inspect.signature(functional_call)),
        "candidate_source_sha256": CANDIDATE_SHA256,
        "state_keys": list(state),
        "state_object_ids": {name: id(tensor) for name, tensor in state.items()},
        "unique_state_keys": list(unique_state),
        "default_full_error": default_full_error,
        "tie_weights_false_output_exact": True,
        "tie_weights_false_identity": tie_false_identity,
        "candidate_output_exact": True,
        "candidate_identity": candidate_identity,
        "aliases_restored_after_call": True,
        "reference": reference.tolist(),
        "candidate": candidate.tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
