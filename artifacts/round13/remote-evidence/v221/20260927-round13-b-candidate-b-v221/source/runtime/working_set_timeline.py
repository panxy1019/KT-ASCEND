"""Default-off bounded request timeline for Round9 working-set planes."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time


_CLOCK_DOMAINS = frozenset((
    "HOST_MONOTONIC",
    "HOST_OBSERVED_DEVICE_OPERATION",
))
_lock = threading.RLock()
_recorder = None


@dataclass(frozen=True, slots=True)
class TimelineIdentity:
    request_id: str
    placement_epoch: int
    allocation_epoch: int


def _configuration():
    raw = os.environ.get("ROUND9_TIMELINE_PATH")
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("ROUND9_TIMELINE_PATH must be absolute")
    try:
        maximum = int(os.environ.get("ROUND9_TIMELINE_MAX_EVENTS", "4096"))
    except ValueError as error:
        raise ValueError("ROUND9_TIMELINE_MAX_EVENTS must be an integer") from error
    if not 1 <= maximum <= 65536:
        raise ValueError("ROUND9_TIMELINE_MAX_EVENTS must be within 1..65536")
    return path, maximum


def enabled():
    """Return whether profiling is configured, validating its bounds when set."""
    return _configuration() is not None


class BoundedRequestTimeline:
    def __init__(self, path, maximum):
        self.path = path
        self.maximum = maximum
        self.identity = None
        self.events = []
        self.dropped = 0
        self.started_ns = None

    def begin(self, identity):
        if self.identity is None:
            self.identity = identity
            self.events = []
            self.dropped = 0
            self.started_ns = time.monotonic_ns()
            return
        if self.identity != identity:
            raise RuntimeError("timeline request changed before flush")

    def record(self, category, phase, clock_domain, fields):
        if self.identity is None:
            raise RuntimeError("timeline event has no request lease")
        if clock_domain not in _CLOCK_DOMAINS:
            raise ValueError("unsupported timeline clock domain")
        if len(self.events) >= self.maximum:
            self.dropped += 1
            return False
        item = {
            "sequence": len(self.events),
            "host_monotonic_ns": time.monotonic_ns(),
            "clock_domain": clock_domain,
            "category": category,
            "phase": phase,
            **fields,
        }
        json.dumps(item, allow_nan=False)
        self.events.append(item)
        return True

    def flush(self, reason):
        if self.identity is None:
            raise RuntimeError("timeline flush has no request lease")
        document = {
            "schema": 1,
            "status": "ROUND9_BOUNDED_WORKING_SET_TIMELINE",
            "identity": asdict(self.identity),
            "started_host_monotonic_ns": self.started_ns,
            "flushed_host_monotonic_ns": time.monotonic_ns(),
            "termination": reason,
            "event_count": len(self.events),
            "dropped_events": self.dropped,
            "events": self.events,
        }
        payload = json.dumps(document, sort_keys=True, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.identity = None
        self.events = []
        self.dropped = 0
        self.started_ns = None


def _get(create):
    global _recorder
    config = _configuration()
    if config is None:
        return None
    if _recorder is None and create:
        _recorder = BoundedRequestTimeline(*config)
    elif _recorder is not None and (
            _recorder.path != config[0] or _recorder.maximum != config[1]):
        raise RuntimeError("timeline configuration changed within one process")
    return _recorder


def begin_request(request_id, placement_epoch, allocation_epoch):
    config = _configuration()
    if config is None:
        return False
    if (not isinstance(request_id, str) or not request_id
            or type(placement_epoch) is not int or placement_epoch < 0
            or type(allocation_epoch) is not int or allocation_epoch < 0):
        raise ValueError("invalid timeline request identity")
    with _lock:
        _get(True).begin(TimelineIdentity(
            request_id, placement_epoch, allocation_epoch))
    return True


def event(category, phase, *, clock_domain="HOST_MONOTONIC", **fields):
    if _configuration() is None:
        return False
    if not isinstance(category, str) or not category or not isinstance(phase, str) or not phase:
        raise ValueError("timeline category and phase must be nonempty strings")
    with _lock:
        recorder = _get(True)
        if recorder.identity is None:
            return False
        return recorder.record(category, phase, clock_domain, fields)


def flush_request(reason):
    if _configuration() is None:
        return False
    if not isinstance(reason, str) or not reason:
        raise ValueError("timeline termination reason must be nonempty")
    with _lock:
        recorder = _get(False)
        # The process-level timeline can be enabled while a short request stays
        # below the profiled streaming threshold.  Its terminal path must remain
        # a no-op instead of manufacturing an empty trace or raising.
        if recorder is None or recorder.identity is None:
            return False
        recorder.flush(reason)
    return True


def snapshot():
    if _configuration() is None:
        return None
    with _lock:
        recorder = _get(False)
        if recorder is None:
            return {"active": False, "event_count": 0, "dropped_events": 0}
        return {
            "active": recorder.identity is not None,
            "identity": None if recorder.identity is None else asdict(recorder.identity),
            "event_count": len(recorder.events),
            "dropped_events": recorder.dropped,
        }


def reset_for_tests():
    global _recorder
    with _lock:
        _recorder = None


__all__ = (
    "TimelineIdentity", "BoundedRequestTimeline", "begin_request", "event",
    "flush_request", "snapshot", "enabled",
)
