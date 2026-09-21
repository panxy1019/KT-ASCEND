"""Default-off cancellation identity, trace and normal-submit fence.

The HTTP, tokenizer and scheduler processes may all append to one local file.
Each event is emitted with one ``O_APPEND`` write; high-frequency working-set
profiling uses a separate bounded in-memory recorder.
"""

from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
import threading
import time


_sequence_lock = threading.Lock()
_sequences = {}


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    request_id: str
    request_epoch: int
    allocation_epoch: int | None


def _trace_path():
    raw = os.environ.get("ROUND9_CANCEL_TRACE_PATH")
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("ROUND9_CANCEL_TRACE_PATH must be absolute")
    return path


def _emit(status, **fields):
    path = _trace_path()
    if path is None:
        return
    with _sequence_lock:
        sequence = _sequences.get(str(path), 0)
        _sequences[str(path)] = sequence + 1
    event = {
        "status": status,
        "event_seq": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(payload) > 4096:
        raise RuntimeError("cancellation event exceeds atomic append bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short cancellation event write")
    finally:
        os.close(descriptor)


class _SchedulerLifecycle:
    def __init__(self):
        self.next_epoch = itertools.count()
        self.by_object = {}
        self.cancelled = set()

    def admit(self, request, allocation_epoch):
        key = id(request)
        current = self.by_object.get(key)
        if current is not None:
            token, identity = current
            if token is not request:
                raise RuntimeError("request object identity was reused before retirement")
            if (allocation_epoch is not None and identity.allocation_epoch is not None
                    and allocation_epoch != identity.allocation_epoch):
                raise RuntimeError("request allocation epoch changed while live")
            if identity.allocation_epoch is None and allocation_epoch is not None:
                identity = RequestIdentity(
                    identity.request_id, identity.request_epoch, allocation_epoch)
                self.by_object[key] = (request, identity)
            return identity
        if not isinstance(request.rid, str) or not request.rid:
            raise ValueError("live request requires a nonempty rid")
        if allocation_epoch is not None and (
                type(allocation_epoch) is not int or allocation_epoch < 0):
            raise ValueError("allocation epoch must be a nonnegative integer")
        identity = RequestIdentity(request.rid, next(self.next_epoch), allocation_epoch)
        self.by_object[key] = (request, identity)
        _emit("REQUEST_ADMITTED", **asdict(identity))
        return identity

    def identity(self, request):
        current = self.by_object.get(id(request))
        if current is None or current[0] is not request:
            return None
        return current[1]

    def retire(self, request):
        identity = self.identity(request)
        if identity is None:
            raise ValueError("unknown or retired request identity")
        del self.by_object[id(request)]
        self.cancelled.discard(identity)
        _emit("REQUEST_RETIRED", **asdict(identity))
        return identity


def _lifecycle(scheduler, *, create):
    lifecycle = getattr(scheduler, "_round9_cancel_lifecycle", None)
    if lifecycle is None and create:
        lifecycle = _SchedulerLifecycle()
        scheduler._round9_cancel_lifecycle = lifecycle
    return lifecycle


def http_abort_received(request_id, abort_all):
    if _trace_path() is None:
        return
    _emit("HTTP_ABORT_RECEIVED", requested_id=request_id, abort_all=bool(abort_all))


def tokenizer_abort_requested(manager, request_id, abort_all):
    if _trace_path() is None:
        return bool(abort_all or request_id in manager.rid_to_state)
    live = bool(abort_all or request_id in manager.rid_to_state)
    _emit(
        "CONTROL_SENT" if live else "CONTROL_NOT_SENT",
        requested_id=request_id,
        abort_all=bool(abort_all),
        reason=None if live else "RID_NOT_LIVE",
    )
    return live


def scheduler_request_admitted(scheduler, request, allocation_epoch=None):
    if _trace_path() is None:
        return None
    return _lifecycle(scheduler, create=True).admit(request, allocation_epoch)


def _scheduler_requests(scheduler):
    seen = set()
    for owner_name in ("waiting_queue", "running_batch", "cur_batch"):
        owner = getattr(scheduler, owner_name, None)
        requests = owner if owner_name == "waiting_queue" else getattr(owner, "reqs", ())
        for request in requests or ():
            if id(request) not in seen:
                seen.add(id(request))
                yield request


def scheduler_abort_received(scheduler, recv_req):
    if _trace_path() is None:
        return ()
    lifecycle = _lifecycle(scheduler, create=True)
    matching = [
        request for request in _scheduler_requests(scheduler)
        if recv_req.abort_all or request.rid.startswith(recv_req.rid)
    ]
    _emit(
        "SCHEDULER_ABORT_RECEIVED",
        requested_id=recv_req.rid,
        abort_all=bool(recv_req.abort_all),
        matched_requests=len(matching),
    )
    identities = []
    for request in matching:
        identity = lifecycle.admit(request, None)
        identities.append(identity)
        if identity in lifecycle.cancelled:
            _emit("CANCEL_ALREADY_SEEN", **asdict(identity))
            continue
        lifecycle.cancelled.add(identity)
        _emit("CANCEL_SEEN", **asdict(identity))
        _emit("SUBMIT_FENCE_PUBLISHED", **asdict(identity))
    return tuple(identities)


def scheduler_before_normal_submit(scheduler, request, operation):
    if _trace_path() is None:
        return True
    lifecycle = _lifecycle(scheduler, create=True)
    identity = lifecycle.identity(request)
    if identity is None:
        identity = lifecycle.admit(request, None)
    if identity in lifecycle.cancelled:
        _emit("POST_CANCEL_SUBMIT_REJECTED", operation=operation, **asdict(identity))
        return False
    _emit("NORMAL_SUBMIT_ADMITTED", operation=operation, **asdict(identity))
    return True


def scheduler_request_retired(scheduler, request):
    if _trace_path() is None:
        return None
    return _lifecycle(scheduler, create=False).retire(request)
