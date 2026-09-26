"""Opt-in, bounded fixture capture around the real prefill group consumer.

The module deliberately imports no tensor runtime at import time.  Capturing is
diagnostic: device-to-host copies and their synchronization cost are part of
the capture and must not be interpreted as production-path performance.
"""
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile


_CAPTURE_PREFIX = "ROUND9_PREFILL_GROUP_CAPTURE"
_MAX_GROUPS = 2
_MAX_ROWS = 32


@dataclass(frozen=True)
class CapturePaths:
    tensor: Path
    receipt: Path
    routes: Path

    @classmethod
    def from_environment(cls, env):
        try:
            prefix = Path(env[_CAPTURE_PREFIX])
        except KeyError as error:
            raise RuntimeError(f"enabled prefill capture requires {error.args[0]}") from error
        if not prefix.is_absolute():
            raise ValueError("prefill capture path prefix must be absolute")
        tensor = Path(str(prefix) + ".pt")
        receipt = Path(str(prefix) + ".receipt.json")
        routes = Path(str(prefix) + ".routes.jsonl")
        for path in (tensor, receipt, routes):
            if not path.parent.is_dir():
                raise FileNotFoundError(f"capture parent directory is absent: {path.parent}")
            if path.exists():
                raise FileExistsError(f"capture output must be fresh: {path}")
        return cls(tensor=tensor, receipt=receipt, routes=routes)


def _plain_int(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def group_metadata(layer, group, identity, request_key, chunk):
    """Validate one real route group and return JSON-safe capture metadata."""
    if type(layer) is not int or layer != 0:
        raise ValueError("capture accepts only logical layer 0")
    if (not isinstance(request_key, tuple) or len(request_key) != 3 or
            not isinstance(chunk, tuple) or len(chunk) != 2):
        raise ValueError("capture requires an active request identity and prefill chunk")
    request_id, request_epoch, input_tokens = request_key
    chunk_start, chunk_count = chunk
    if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
        raise ValueError("request identity must have a scalar request_id")
    _plain_int(request_epoch, "request placement epoch")
    _plain_int(input_tokens, "request input tokens", minimum=1)
    _plain_int(chunk_start, "chunk start")
    _plain_int(chunk_count, "chunk count", minimum=1)
    if chunk_count > _MAX_ROWS:
        raise ValueError("capture chunk exceeds the 32-row bound")

    identity_request = getattr(identity, "request_id", None)
    identity_epoch = getattr(identity, "placement_epoch", None)
    allocation_epoch = getattr(identity, "allocation_epoch", None)
    if identity_request != request_id or identity_epoch != request_epoch:
        raise ValueError("group identity differs from the active request identity")
    _plain_int(allocation_epoch, "allocation epoch", minimum=1)

    try:
        rows = tuple(group.rows)
        logical = tuple(group.logical_experts)
        slots = tuple(tuple(row) for row in group.remapped_ids)
    except (AttributeError, TypeError) as error:
        raise ValueError("group is missing rows, logical experts, or slot IDs") from error
    if not 1 <= len(rows) <= _MAX_ROWS or len(rows) > chunk_count:
        raise ValueError("group must contain between 1 and 32 bounded rows")
    if any(type(row) is not int for row in rows):
        raise ValueError("group rows must be integers")
    if rows != tuple(range(rows[0], rows[0] + len(rows))):
        raise ValueError("group rows must be contiguous and increasing")
    if rows[0] < 0 or rows[-1] >= chunk_count:
        raise ValueError("group rows fall outside the active chunk")
    if (not 1 <= len(logical) <= 32 or len(set(logical)) != len(logical) or
            any(type(expert) is not int or expert < 0 for expert in logical)):
        raise ValueError("group logical experts must be distinct non-negative IDs")
    if len(slots) != len(rows) or not slots or not slots[0]:
        raise ValueError("group slot IDs must cover every captured row")
    width = len(slots[0])
    if any(len(slot_row) != width for slot_row in slots):
        raise ValueError("group slot ID rows must have one fixed TopK width")
    if any(type(slot) is not int or not 0 <= slot < len(logical)
           for slot_row in slots for slot in slot_row):
        raise ValueError("group slot ID is outside the logical expert mapping")
    logical_ids = [[logical[slot] for slot in slot_row] for slot_row in slots]
    return {
        "logical_layer": layer,
        "request_id": request_id,
        "placement_epoch": request_epoch,
        "allocation_epoch": allocation_epoch,
        "input_tokens": input_tokens,
        "chunk": [chunk_start, chunk_count],
        "rows": list(rows),
        "token_span": [rows[0], rows[-1] + 1],
        "request_token_span": [chunk_start + rows[0], chunk_start + rows[-1] + 1],
        "logical_experts": list(logical),
        "slot_ids": [list(row) for row in slots],
        "logical_ids": logical_ids,
    }


def route_group_metadata(layer, group, identity, request_key, chunk):
    """Return host-only route metadata for any real layerwise group call."""
    if not _integer_layer(layer):
        raise ValueError("route tape logical layer must be non-negative")
    if not isinstance(request_key, tuple) or len(request_key) != 3:
        raise ValueError("route tape requires an active request identity")
    request_id, request_epoch, input_tokens = request_key
    if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
        raise ValueError("route tape request_id must be scalar")
    _plain_int(request_epoch, "route tape placement epoch")
    _plain_int(input_tokens, "route tape input tokens", minimum=1)
    if (getattr(identity, "request_id", None) != request_id or
            getattr(identity, "placement_epoch", None) != request_epoch):
        raise ValueError("route tape group identity changed")
    allocation = getattr(identity, "allocation_epoch", None)
    _plain_int(allocation, "route tape allocation epoch", minimum=1)
    if chunk is None:
        chunk_value = None
    else:
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            raise ValueError("route tape chunk is malformed")
        start, count = chunk
        _plain_int(start, "route tape chunk start")
        _plain_int(count, "route tape chunk count", minimum=1)
        if count > _MAX_ROWS:
            raise ValueError("route tape chunk exceeds 32 rows")
        chunk_value = [start, count]
    try:
        rows = tuple(group.rows)
        logical = tuple(group.logical_experts)
        remapped = tuple(tuple(row) for row in group.remapped_ids)
    except (AttributeError, TypeError) as error:
        raise ValueError("route tape group fields are unavailable") from error
    if (not 1 <= len(rows) <= _MAX_ROWS or
            any(type(row) is not int for row in rows) or
            rows != tuple(range(rows[0], rows[0] + len(rows)))):
        raise ValueError("route tape rows must be bounded and contiguous")
    if chunk_value is not None and (rows[0] < 0 or rows[-1] >= chunk_value[1]):
        raise ValueError("route tape rows fall outside the chunk")
    if (not 1 <= len(logical) <= 32 or len(logical) != len(set(logical)) or
            any(type(expert) is not int or expert < 0 for expert in logical)):
        raise ValueError("route tape logical experts are invalid")
    if (len(remapped) != len(rows) or not remapped or not remapped[0] or
            any(len(row) != len(remapped[0]) for row in remapped) or
            any(type(slot) is not int or not 0 <= slot < len(logical)
                for row in remapped for slot in row)):
        raise ValueError("route tape remapped IDs are invalid")
    return {
        "request_id": request_id,
        "placement_epoch": request_epoch,
        "input_tokens": input_tokens,
        "chunk": chunk_value,
        "logical_layer": layer,
        "allocation_epoch": allocation,
        "rows": list(rows),
        "logical_experts": list(logical),
        "remapped_ids": [list(row) for row in remapped],
    }


def _integer_layer(value):
    return type(value) is int and value >= 0


class RouteTape:
    """Bounded, append-only host route order for later chunk/group analysis."""
    def __init__(self, path, *, max_groups=4096, max_bytes=8 * 1024**2):
        self.path = Path(path)
        if (type(max_groups) is not int or max_groups < 1 or
                type(max_bytes) is not int or max_bytes < 1024):
            raise ValueError("route tape bounds are invalid")
        self.max_groups = max_groups
        self.max_bytes = max_bytes
        self.count = 0
        self.failed = False
        self.failure = None
        header = {
            "schema": 1,
            "kind": "header",
            "status": "RECORDING_HOST_ONLY_ROUTE_TAPE",
            "identity_source": "request_cost_recorder.request_key_and_layerwise_BufferIdentity",
            "diagnostic_host_observer": True,
            "tensor_copy": False,
            "performance_evidence": False,
            "max_groups": max_groups,
            "max_bytes": max_bytes,
        }
        encoded = _json_line(header)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.bytes_written = len(encoded)

    def _append(self, row):
        encoded = _json_line(row)
        with self.path.open("ab", buffering=0) as stream:
            stream.write(encoded)
        self.bytes_written += len(encoded)

    def fail(self, reason):
        if self.failed:
            return
        self.failed = True
        self.failure = str(reason)[:300]
        row = {
            "schema": 1,
            "kind": "failure",
            "status": "FAILED_ROUTE_TAPE_LIMIT" if reason == "LIMIT" else "FAILED_ROUTE_TAPE_RECORDING",
            "reason": self.failure,
            "recorded_group_count": self.count,
            "bytes_before_failure": self.bytes_written,
        }
        try:
            encoded = _json_line(row)
            if self.bytes_written + len(encoded) <= self.max_bytes:
                self._append(row)
        except BaseException:
            pass

    def record(self, metadata):
        if self.failed:
            return
        row = {
            "schema": 1,
            "kind": "group",
            "sequence": self.count,
            "recorded_group_count": self.count + 1,
            **metadata,
        }
        encoded = _json_line(row)
        # Keep enough space for an explicit bounded failure record.
        if self.count >= self.max_groups or self.bytes_written + len(encoded) > self.max_bytes - 512:
            self.fail("LIMIT")
            return
        try:
            self._append(row)
            self.count += 1
        except BaseException as error:
            self.fail(f"{type(error).__name__}: {error}")


def _json_line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


class CaptureSequence:
    """Pure-Python admission state for exactly two adjacent real groups."""
    def __init__(self):
        self._first = None
        self._count = 0

    @property
    def complete(self):
        return self._count == _MAX_GROUPS

    def accept(self, metadata):
        if self.complete:
            return None
        if self._first is None:
            self._first = dict(metadata)
        else:
            first = self._first
            if (metadata["request_id"], metadata["placement_epoch"]) != (
                    first["request_id"], first["placement_epoch"]):
                raise ValueError("capture group identity changed")
            if metadata["chunk"] != first["chunk"]:
                raise ValueError("capture group chunk changed")
            if metadata["logical_layer"] != first["logical_layer"]:
                raise ValueError("capture logical layer changed")
            if metadata["allocation_epoch"] != first["allocation_epoch"] + 1:
                raise ValueError("capture groups are not adjacent allocations")
        index = self._count
        self._count += 1
        return index


class _DisabledCapture:
    enabled = False
    complete = False
    failed = False
    failure = None


class _PrefillGroupCapture:
    enabled = True

    def __init__(self, paths, recorder, route_tape):
        self.paths = paths
        self.recorder = recorder
        self.sequence = CaptureSequence()
        self.groups = []
        self.failed = False
        self.failure = None
        self.complete = False
        self.route_tape = route_tape

    def _fail(self, error):
        if not self.complete and not self.failed:
            self.failed = True
            self.failure = f"{type(error).__name__}: {error}"[:500]

    def before_group(self, layer, group, hidden_states, topk_weights, identity):
        if (self.complete or self.failed or layer != 0 or
                self.recorder.chunk is None):
            return None
        try:
            metadata = group_metadata(
                layer, group, identity, self.recorder.request_key,
                self.recorder.chunk,
            )
            index = self.sequence.accept(metadata)
            if index is None:
                return None
            # The default-OFF path returns before this deliberately lazy import.
            import torch
            device = getattr(hidden_states, "device", None)
            rows = torch.tensor(metadata["rows"], dtype=torch.int64, device=device)
            hidden = hidden_states.index_select(0, rows).detach().cpu().clone()
            weights = topk_weights.index_select(0, rows).detach().cpu().clone()
            if hidden.shape[0] != len(metadata["rows"]) or weights.shape[0] != len(metadata["rows"]):
                raise ValueError("selected tensor rows differ from group rows")
            if weights.ndim != 2 or weights.shape[1] != len(metadata["slot_ids"][0]):
                raise ValueError("selected TopK weights differ from group slot IDs")
            return {
                "index": index,
                "metadata": metadata,
                "hidden_states": hidden,
                "topk_weights": weights,
                "topk_slot_ids": torch.tensor(metadata["slot_ids"], dtype=torch.int32),
                "topk_logical_ids": torch.tensor(metadata["logical_ids"], dtype=torch.int64),
            }
        except BaseException as error:
            self._fail(error)
            return None

    def provider_failed(self, ticket, error):
        if ticket is not None:
            self._fail(RuntimeError(f"real consumer failed: {type(error).__name__}"))

    def after_group(self, ticket, result):
        if ticket is None or self.complete or self.failed:
            return
        try:
            import torch
            if torch.is_tensor(result):
                output = result
            else:
                output = torch.stack(tuple(result), dim=0)
            output = output.detach().float().cpu().clone()
            if output.shape[0] != len(ticket["metadata"]["rows"]):
                raise ValueError("real consumer output row count differs from group rows")
            ticket["reference_contribution_fp32"] = output
            if ticket["index"] != len(self.groups):
                raise RuntimeError("capture completion order changed")
            self.groups.append(ticket)
            if len(self.groups) == _MAX_GROUPS:
                self._publish(torch)
                self.complete = True
        except BaseException as error:
            self._fail(error)

    def _publish(self, torch):
        payload_groups = []
        receipt_groups = []
        for captured in self.groups:
            metadata = captured["metadata"]
            payload_groups.append({
                "metadata": metadata,
                "hidden_states": captured["hidden_states"],
                "topk_weights": captured["topk_weights"],
                "topk_slot_ids": captured["topk_slot_ids"],
                "topk_logical_ids": captured["topk_logical_ids"],
                "reference_contribution_fp32": captured["reference_contribution_fp32"],
            })
            receipt_groups.append({
                "group_index": captured["index"],
                **{key: metadata[key] for key in (
                    "logical_layer", "allocation_epoch", "rows", "token_span",
                    "request_token_span", "logical_experts", "slot_ids",
                    "logical_ids",
                )},
            })
        payload = {
            "schema": 1,
            "capture": "REAL_PREFILL_GROUP_CONSUMER",
            "diagnostic_device_copy_and_sync": True,
            "groups": payload_groups,
        }
        try:
            _atomic_torch_save(torch, payload, self.paths.tensor)
            raw = self.paths.tensor.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            first = self.groups[0]["metadata"]
            receipt = {
                "schema": 1,
                "status": "COMPLETE_REAL_CONSUMER_GROUP_CAPTURE",
                "diagnostic_device_copy_and_sync": True,
                "performance_evidence": False,
                "tensor_path": str(self.paths.tensor),
                "tensor_bytes": len(raw),
                "tensor_sha256": digest,
                "request": {
                    "request_id": first["request_id"],
                    "placement_epoch": first["placement_epoch"],
                    "input_tokens": first["input_tokens"],
                    "chunk": first["chunk"],
                },
                "groups": receipt_groups,
            }
            encoded = (json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n").encode()
            _atomic_bytes(encoded, self.paths.receipt)
        except BaseException:
            if not self.paths.receipt.exists():
                try:
                    self.paths.tensor.unlink()
                except FileNotFoundError:
                    pass
            raise


def _link_temp(temp_name, destination):
    try:
        # link() is an atomic create-without-overwrite completion step.
        os.link(temp_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _atomic_torch_save(torch, value, destination):
    with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False) as stream:
        temp_name = stream.name
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    _link_temp(temp_name, destination)


def _atomic_bytes(value, destination):
    with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False) as stream:
        temp_name = stream.name
        try:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    _link_temp(temp_name, destination)


def bind_capture(backend, recorder, env=None):
    """Wrap an existing cost-observer execute_group seam when explicitly ON."""
    environment = os.environ if env is None else env
    if not environment.get(_CAPTURE_PREFIX):
        return _DisabledCapture()
    if getattr(backend, "_round9_prefill_group_capture", None) is not None:
        raise RuntimeError("prefill group capture is already bound")
    execute = backend.execute_group
    if not callable(execute):
        raise TypeError("capture backend is missing execute_group()")
    paths = CapturePaths.from_environment(environment)
    route_tape = RouteTape(paths.routes)
    capture = _PrefillGroupCapture(paths, recorder, route_tape)

    @functools.wraps(execute)
    def execute_group(layer, group, hidden_states, topk_weights, identity):
        try:
            route_tape.record(route_group_metadata(
                layer, group, identity, recorder.request_key, recorder.chunk
            ))
        except BaseException as error:
            route_tape.fail(f"{type(error).__name__}: {error}")
        ticket = capture.before_group(
            layer, group, hidden_states, topk_weights, identity
        )
        try:
            result = execute(layer, group, hidden_states, topk_weights, identity)
        except BaseException as error:
            capture.provider_failed(ticket, error)
            raise
        capture.after_group(ticket, result)
        return result

    backend.execute_group = execute_group
    backend._round9_prefill_group_capture = capture
    return capture
