"""Default-off control and observer contract for a future fresh A1 profile.

This module is deliberately runtime-neutral.  It never imports torch, creates a
device event, starts a process, or executes a command.  Device code may later
feed a completed event duration into :func:`device_elapsed_record` after the
same deferred-query boundary used by the frozen Round15 observer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping


STAGE = "ROUND16_A1_FRESH_PROFILE"
DEFAULT_ENABLED = False
GENERATIONS_PER_BLOCK = 6
GENERATION_LIMIT_TOTAL = 12
RUNTIME_VERSION = "v003"
EVIDENCE_DATE = "20260929"

BLOCK_IDS: Mapping[str, str] = MappingProxyType(
    {
        "A": "round16-a1-fresh-profile-a-v001",
        "B": "round16-a1-fresh-profile-b-v001",
    }
)


def runtime_container_name(block: str) -> str:
    if block not in BLOCK_IDS:
        raise ValueError("block must be exactly 'A' or 'B'")
    return f"kt-round16-a1-{block.lower()}-{RUNTIME_VERSION}"


def runtime_evidence_name(block: str) -> str:
    if block not in BLOCK_IDS:
        raise ValueError("block must be exactly 'A' or 'B'")
    return (
        f"{EVIDENCE_DATE}-round16-a1-fresh-profile-"
        f"{block.lower()}-{RUNTIME_VERSION}"
    )

_SCHEDULE = (
    {
        "ordinal": 1,
        "role": "W1_WARMUP",
        "workload": "W1",
        "phase": "warmup",
        "input_tokens": 128,
        "output_tokens": 16,
    },
    {
        "ordinal": 2,
        "role": "W2_WARMUP",
        "workload": "W2",
        "phase": "warmup",
        "input_tokens": 128,
        "output_tokens": 128,
    },
    {
        "ordinal": 3,
        "role": "W1_MEASUREMENT",
        "workload": "W1",
        "phase": "measurement",
        "input_tokens": 128,
        "output_tokens": 16,
    },
    {
        "ordinal": 4,
        "role": "W2_MEASUREMENT",
        "workload": "W2",
        "phase": "measurement",
        "input_tokens": 128,
        "output_tokens": 128,
    },
    {
        "ordinal": 5,
        "role": "SHORT_DRAIN",
        "workload": "DRAIN",
        "phase": "drain",
        "fixture": "round10-t1-fixture.json#cases.canary_math391",
        "fixture_sha256": "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95",
        "driver_source": "artifacts/round15/p0-snapshots/v338/round14a_b_profile_driver.py",
        "driver_source_sha256": "325f9e06901cca49c4f92ca4f3fab9112d3500343af7864b297972f9c1d01e44",
        "input_tokens": 27,
        "output_tokens": 4,
        "ignore_eos": True,
    },
    {
        "ordinal": 6,
        "role": "FIXED_CANARY",
        "workload": "CANARY",
        "phase": "canary",
        "fixture": "round10-t1-fixture.json#cases.canary_math391",
        "fixture_sha256": "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95",
        "input_tokens": 27,
        "input_ids_sha256": "4d2584778859d8b34fa388d6fb09925c1da1a0197945de1298f09d199dce489d",
        "messages_sha256": "e5a27eaad7efa96e5ff9020ca99f901cf968238ac7ae7b662c7369923c5e6615",
        "temperature": 0,
        "top_p": 1,
        "seed": 20260920,
        "max_new_tokens": 32,
        "ignore_eos": False,
        "validator": {"kind": "exact_text", "expected": "391"},
        "logprob_requirement": "FINITE",
    },
)

OBSERVER_HOOKS: Mapping[str, str] = MappingProxyType(
    {
        "parent": "KTEPWrapperMethod.apply",
        "host_dispatch": "KTEPWrapperMethod._submit_cpu_forward",
        "host_sync": "KTEPWrapperMethod._sync_cpu_forward",
        "npu_shared": "SharedFullContext.gpu_method.apply",
        "npu_resident": "KTEPWrapperMethod.gpu_method.apply",
        "host_merge": "KTEPWrapperMethod.apply after main-stream wait_event",
        "scheduler_before": "round13_attribution.before",
        "scheduler_after": "round13_attribution.after",
    }
)


class BudgetExceededError(RuntimeError):
    """The immutable two-block profile budget would be exceeded."""


@dataclass(frozen=True)
class BlockContract:
    block: str
    block_id: str
    snapshot_id: str
    enabled: bool = DEFAULT_ENABLED
    model_process_limit: int = 1
    generation_limit: int = GENERATIONS_PER_BLOCK
    abort_limit: int = 0
    automatic_retry_limit: int = 0


def block_contract(block: str) -> BlockContract:
    """Return one frozen, non-executable block identity."""
    if block not in BLOCK_IDS:
        raise ValueError("block must be exactly 'A' or 'B'")
    block_id = BLOCK_IDS[block]
    return BlockContract(
        block=block,
        block_id=block_id,
        snapshot_id=f"{block_id}-snapshot",
    )


@dataclass(frozen=True)
class BudgetLedger:
    """Immutable generation-attempt accounting for the two fresh blocks."""

    used_by_block: tuple[tuple[str, int], ...]
    request_ids: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "BudgetLedger":
        return cls(tuple((block_id, 0) for block_id in BLOCK_IDS.values()))

    def used_for(self, block_id: str) -> int:
        return dict(self.used_by_block).get(block_id, 0)

    @property
    def total_used(self) -> int:
        return sum(count for _, count in self.used_by_block)


def spend_generation(
    ledger: BudgetLedger,
    *,
    block_id: str,
    request_id: str,
) -> BudgetLedger:
    """Return a new ledger after one unique attempt, or fail before mutation."""
    if block_id not in BLOCK_IDS.values():
        raise BudgetExceededError(f"unknown immutable block identity {block_id!r}")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    if request_id in ledger.request_ids:
        raise BudgetExceededError(f"duplicate request identity {request_id!r}")

    counts = dict(ledger.used_by_block)
    if counts[block_id] >= GENERATIONS_PER_BLOCK:
        raise BudgetExceededError(f"block generation budget exhausted for {block_id}")
    if ledger.total_used >= GENERATION_LIMIT_TOTAL:
        raise BudgetExceededError("total generation budget exhausted")
    counts[block_id] += 1
    ordered = tuple((item, counts[item]) for item in BLOCK_IDS.values())
    return BudgetLedger(ordered, ledger.request_ids + (request_id,))


@dataclass(frozen=True)
class SpanIdentity:
    """Identity shared by Host, NPU-event, and explicit UNKNOWN records."""

    component: str
    span_id: str
    parent_span_id: str
    stream_id: str
    start_event_id: str
    end_event_id: str
    correlation_id: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def with_component(self, component: str) -> "SpanIdentity":
        return replace(self, component=component)


def _identity_payload(identity: SpanIdentity) -> dict[str, str]:
    return asdict(identity)


def _exact_tick(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("observer clock must return an exact int")
    return value


class FreshA1Observer:
    """Host-call observer whose disabled path is a transparent direct call."""

    def __init__(
        self,
        *,
        enabled: bool = DEFAULT_ENABLED,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.enabled = enabled
        self._clock = clock
        self._rows: list[dict[str, Any]] = []

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._rows)

    def observe_call(
        self,
        identity: SpanIdentity,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call once and preserve its result or exception identity.

        Observer clock failures suppress the record instead of changing the
        wrapped operation's behavior.  No device synchronization is performed.
        """
        if not self.enabled:
            return operation(*args, **kwargs)
        try:
            started = _exact_tick(self._clock())
        except Exception:
            return operation(*args, **kwargs)

        outcome = "error"
        try:
            value = operation(*args, **kwargs)
            outcome = "return"
            return value
        finally:
            try:
                ended = _exact_tick(self._clock())
                if ended < started:
                    raise ValueError("observer clock moved backwards")
                self._rows.append(
                    {
                        "schema": 1,
                        **_identity_payload(identity),
                        "clock_domain": "HOST_MONOTONIC_NS",
                        "start_ns": started,
                        "end_ns": ended,
                        "duration_ns": ended - started,
                        "outcome": outcome,
                        "observer_status": "COMPLETE",
                    }
                )
            except Exception:
                pass


def device_elapsed_record(
    identity: SpanIdentity,
    *,
    elapsed_ms: float,
) -> dict[str, Any]:
    """Describe an already-completed deferred NPU event query.

    Event creation, recording, and querying belong to later device wiring.  The
    absence of absolute timestamps is explicit so Host/NPU clocks cannot be
    accidentally aligned.
    """
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)):
        raise TypeError("elapsed_ms must be a real number")
    value = float(elapsed_ms)
    if not math.isfinite(value) or value < 0:
        raise ValueError("elapsed_ms must be finite and non-negative")
    return {
        "schema": 1,
        **_identity_payload(identity),
        "clock_domain": "NPU_EVENT_ELAPSED_MS",
        "start_ns": None,
        "end_ns": None,
        "duration_ns": None,
        "elapsed_ms": value,
        "status": "COMPLETE",
        "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
    }


def cpu_expert_unknown(identity: SpanIdentity) -> dict[str, Any]:
    """Represent the native async CPU work without inventing a duration."""
    return {
        "schema": 1,
        **_identity_payload(identity),
        "clock_domain": "NATIVE_WORKER_CLOCK_UNAVAILABLE",
        "start_ns": None,
        "end_ns": None,
        "duration_ns": None,
        "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
        "reason": (
            "submit_forward/sync_forward expose no native worker start/end callback; "
            "reliable direct timing would require native ABI instrumentation"
        ),
        "derivation": "NOT_INFERRED_FROM_HOST_BOOKENDS",
    }


def controller_plan(block: str) -> dict[str, Any]:
    """Return non-executable snapshot/controller wiring for later review."""
    contract = block_contract(block)
    return {
        "schema": 1,
        "stage": STAGE,
        "enabled": False,
        "executable": False,
        "environment": {"ROUND16_A1_PROFILE_ENABLE": "0"},
        "snapshot": {
            "snapshot_id": contract.snapshot_id,
            "block": contract.block,
            "block_id": contract.block_id,
            "manifest": "round16-a1-source-sha256.json",
            "contract": "ROUND16_A1_FRESH_PROFILE_CONTRACT.json",
            "fresh_outputs_required": True,
        },
        "controller": {
            "model_process_limit": contract.model_process_limit,
            "generation_limit": contract.generation_limit,
            "abort_limit": contract.abort_limit,
            "automatic_retry_limit": contract.automatic_retry_limit,
            "executes_commands": False,
        },
        "observer_boundary_catalog": dict(OBSERVER_HOOKS),
        "observer_installation": {
            "round16_a1_host_layers": list(range(48)),
            "round13_direct_device_components": [
                "QSA_INDEXER",
                "GDN",
                "NPU_RESIDENT",
            ],
            "round13_device_sample_layers": [0, 24, 47],
            "a1_npu_shared_installed": False,
            "catalog_is_installation_claim": False,
        },
        "activation_requirements": [
            "separate device-stage authorization",
            "new immutable source manifest",
            "fresh output directory",
            "explicit 12-generation shared ledger",
            "ROUND16_A1_PROFILE_ENABLE=1 in the reviewed snapshot only",
        ],
    }


def manifest_payload() -> dict[str, Any]:
    """Return the JSON contract shared by tests and snapshot preparation."""
    blocks = [asdict(block_contract(block)) for block in ("A", "B")]
    return {
        "schema": 1,
        "status": "FROZEN_OFFLINE_NOT_RUN",
        "stage": STAGE,
        "default_enabled": DEFAULT_ENABLED,
        "source_basis": {
            "parent": "ROUND14A_V295_PROMOTED_GDN_S0",
            "round15_terminal_status": (
                "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION"
            ),
            "round15_final_status_sha256": (
                "f5f1e77a8972601b886dde67067ac845b511fd32a1268b9213d4eb44194db31f"
            ),
            "round15_p0_result_sha256": (
                "56de9c3d6eabff94b0b6f10f127c7d8767679da76d79e0f3659ad2f9683e6482"
            ),
            "observer_sha256": (
                "a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0"
            ),
            "ktep_wrapper_sha256": (
                "77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b"
            ),
            "scheduler_sha256": (
                "5b96f89ee24f49671b3c35960922d7f93b2133c62e087642d6bdde3ea20b1c6b"
            ),
            "launcher_sha256": (
                "daad3cf0ad99c870c4a884ca0440c4f88d5c887984c4735dff2fe72b5c291f1c"
            ),
            "cpu_worker_api": [
                "KTEPWrapperMethod._submit_cpu_forward",
                "KTEPWrapperMethod._sync_cpu_forward",
            ],
        },
        "snapshot_builder": {
            "implementation": "round16/a/fresh_a1_snapshot.py",
            "mutation_scope": "COPIED_TREE_ONLY",
            "parent_write_count": 0,
            "exact_sha_preconditions": True,
            "fresh_output_required": True,
            "default_enabled": False,
            "actual_boundaries": [
                "KTEPWrapperMethod.apply Host parent",
                "KTEPWrapperMethod._submit_cpu_forward",
                "KTEPWrapperMethod._sync_cpu_forward",
                "KTEPWrapperMethod.apply post-wait_event merge",
                "round13_attribution QSA_INDEXER/GDN/NPU_RESIDENT direct samples",
            ],
            "production_sync_changes": 0,
            "native_abi_changes": 0,
        },
        "blocks": blocks,
        "schedule": [dict(row) for row in _SCHEDULE],
        "budget": {
            "model_processes_total": 2,
            "model_processes_per_block": 1,
            "generation_attempts_total": GENERATION_LIMIT_TOTAL,
            "generation_attempts_per_block": GENERATIONS_PER_BLOCK,
            "abort_controls": 0,
            "automatic_retries": 0,
        },
        "observer": {
            "host_clock_domain": "HOST_MONOTONIC_NS",
            "device_clock_domain": "NPU_EVENT_ELAPSED_MS",
            "device_query": "DEFERRED_QUERY_NO_SYNCHRONIZE",
            "identity_fields": [
                "span_id",
                "parent_span_id",
                "stream_id",
                "start_event_id",
                "end_event_id",
                "correlation_id",
            ],
            "boundary_catalog": dict(OBSERVER_HOOKS),
            "boundary_catalog_is_installation_claim": False,
            "installed_wiring": {
                "round16_a1_host_layers": list(range(48)),
                "round13_direct_device_components": [
                    "QSA_INDEXER",
                    "GDN",
                    "NPU_RESIDENT",
                ],
                "round13_device_sample_layers": [0, 24, 47],
                "a1_npu_shared": False,
            },
            "cpu_expert": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
            "parent_child_arithmetic": "FORBIDDEN",
        },
        "candidate": {
            "enabled": False,
            "selected": None,
            "request_gate": "DECODE_W2_DEVICE_REQUIRED_NOT_RUN",
        },
        "execution": "DEVICE_REQUIRED_NOT_RUN",
    }


__all__ = (
    "BLOCK_IDS",
    "DEFAULT_ENABLED",
    "EVIDENCE_DATE",
    "RUNTIME_VERSION",
    "BudgetExceededError",
    "BudgetLedger",
    "FreshA1Observer",
    "SpanIdentity",
    "block_contract",
    "controller_plan",
    "cpu_expert_unknown",
    "device_elapsed_record",
    "manifest_payload",
    "runtime_container_name",
    "runtime_evidence_name",
    "spend_generation",
)
