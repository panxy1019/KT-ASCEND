"""Boundary that keeps CPU submission outside any device graph replay."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Sequence

from round16.common.contracts import SupportDecision

from .lifecycle_bridge import RequestLease, RequestLeaseManager, TaskIdentity
from .routing_bridge import HybridRoutePlan


@dataclass(frozen=True)
class ReplaySubmission:
    npu_output: Any
    cpu_handle: Any
    cpu_task: TaskIdentity | None
    replay_epoch: int
    input_fingerprint: str
    cpu_identity: tuple[Any, ...] | None


class HybridGraphBoundary:
    """Replay resident NPU work while issuing fresh CPU work per replay."""

    def __init__(self, lifecycle: RequestLeaseManager) -> None:
        self._lifecycle = lifecycle
        self._epoch_lock = Lock()
        self._replay_epoch = 0
        self.device_validation = SupportDecision.device_required(
            "NPUGraph capture/replay requires the frozen Ascend device image"
        )

    def replay(
        self,
        lease: RequestLease,
        route_plan: HybridRoutePlan,
        hidden_states: Sequence[Any],
        *,
        input_fingerprint: str,
        replay_npu: Callable[[HybridRoutePlan], Any],
        submit_cpu: Callable[[HybridRoutePlan, Sequence[Any], tuple[Any, ...]], Any],
    ) -> ReplaySubmission:
        if not isinstance(input_fingerprint, str) or not input_fingerprint.strip():
            raise ValueError("input_fingerprint must be non-empty")
        self._lifecycle.assert_active(lease)
        with self._epoch_lock:
            self._replay_epoch += 1
            replay_epoch = self._replay_epoch
        if not route_plan.cpu_routes:
            npu_output = replay_npu(route_plan)
            return ReplaySubmission(
                npu_output,
                None,
                None,
                replay_epoch,
                input_fingerprint,
                None,
            )
        task = self._lifecycle.begin_task(lease, "cpu-moe-graph-replay")
        cpu_identity = (
            *task.identity,
            "replay_epoch",
            replay_epoch,
            "input_fingerprint",
            input_fingerprint,
        )
        try:
            npu_output = replay_npu(route_plan)
            cpu_handle = submit_cpu(route_plan, hidden_states, cpu_identity)
        except BaseException:
            self._lifecycle.abort_task(task)
            raise
        return ReplaySubmission(
            npu_output,
            cpu_handle,
            task,
            replay_epoch,
            input_fingerprint,
            cpu_identity,
        )


__all__ = ("HybridGraphBoundary", "ReplaySubmission")
