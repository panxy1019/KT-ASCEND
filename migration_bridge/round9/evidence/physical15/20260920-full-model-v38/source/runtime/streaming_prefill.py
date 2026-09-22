"""Cross-plane lifecycle for explicit bounded streaming prefill mode."""


class StreamingPrefillCoordinator:
    def __init__(self, threshold_tokens, *, chunk_tokens, event=None):
        if (type(threshold_tokens) is not int or threshold_tokens < 1
                or type(chunk_tokens) is not int or chunk_tokens < 1):
            raise ValueError("positive streaming threshold and chunk bound required")
        self.threshold = threshold_tokens
        self.chunk_tokens = chunk_tokens
        self.event = event or (lambda item: None)
        self.state = "READY"
        self.request_token = None
        self.identity = None
        self.total = self.next_start = self.chunks = 0
        self.active = self.complete = False
        self.planes = None

    def _ready(self):
        if self.state == "BROKEN_RELOAD_REQUIRED":
            raise RuntimeError(self.state)

    def _poison(self, error):
        self.state = "BROKEN_RELOAD_REQUIRED"
        self.active = False
        raise error

    def _emit(self, item):
        try:
            self.event(item)
        except BaseException as error:
            self._poison(error)

    def begin(self, request_id, request_token, *, placement_epoch, allocation_epoch,
              total, start, count):
        self._ready()
        values = (placement_epoch, allocation_epoch, total, start, count)
        if (not isinstance(request_id, str) or not request_id
                or any(type(value) is not int or value < 0 for value in values)
                or count < 1 or count > self.chunk_tokens or start + count > total):
            raise ValueError("invalid bounded streaming chunk")
        if total < self.threshold:
            if self.request_token is not None or self.active:
                raise RuntimeError("hybrid fallback cannot interrupt streaming state")
            return False
        if self.active or self.complete:
            raise RuntimeError("prior streaming chunk/request has not drained")
        identity = (request_id, placement_epoch, allocation_epoch)
        if self.request_token is None:
            if start != 0:
                raise ValueError("first streaming chunk must start at zero")
            self.request_token = request_token
            self.identity = identity
            self.total = total
            self.next_start = self.chunks = 0
            self.state = "STREAMING_ACTIVE"
        elif (request_token is not self.request_token or identity != self.identity
              or total != self.total):
            raise ValueError("streaming request or epoch changed between chunks")
        if start != self.next_start:
            raise ValueError("streaming chunks must be contiguous")
        self.active = True
        self.planes = {"expert": False, "ple": False, "context": False}
        self.current = (start, count)
        self._emit({
            "status": "STREAMING_CHUNK_BEGIN", "request_id": request_id,
            "placement_epoch": placement_epoch, "allocation_epoch": allocation_epoch,
            "start": start, "count": count, "total": total,
        })
        return True

    def _mark(self, name):
        self._ready()
        if not self.active or self.planes[name]:
            self._poison(RuntimeError(f"duplicate or unleased streaming {name} plane"))
        self.planes[name] = True

    def mark_expert(self, *, layers):
        if type(layers) is not int or layers != 48:
            self._poison(ValueError("streaming expert plane requires all48 layers"))
        self._mark("expert")

    def mark_ple(self, *, async_engaged, drained):
        if type(async_engaged) is not bool or drained is not True:
            self._poison(ValueError("streaming PLE plane must drain exactly"))
        self._mark("ple")

    def mark_context(self, *, layers, hbm_release):
        if type(layers) is not int or layers != 12 or hbm_release is not False:
            self._poison(ValueError("streaming context reference must drain12 layers without HBM claim"))
        self._mark("context")

    def finish_chunk(self):
        self._ready()
        if not self.active or not all(self.planes.values()):
            self._poison(RuntimeError("all streaming working-set planes must drain"))
        start, count = self.current
        self.active = False
        self.next_start = start + count
        self.chunks += 1
        self._emit({
            "status": "STREAMING_CHUNK_DRAINED", "request_id": self.identity[0],
            "placement_epoch": self.identity[1], "allocation_epoch": self.identity[2],
            "start": start, "count": count, "next_start": self.next_start,
            "planes": ["expert", "ple", "context"],
        })
        if self.next_start == self.total:
            self.complete = True
            self._emit({
                "status": "STREAMING_PREFILL_COMPLETE",
                "request_id": self.identity[0], "placement_epoch": self.identity[1],
                "allocation_epoch": self.identity[2], "chunks": self.chunks,
                "tokens": self.total,
            })

    def abort(self, request_token):
        self._ready()
        if request_token is not self.request_token:
            raise RuntimeError("streaming abort identity is stale")
        identity = self.identity
        self._emit({
            "status": "STREAMING_ABORTED", "request_id": identity[0],
            "placement_epoch": identity[1], "allocation_epoch": identity[2],
            "next_start": self.next_start,
        })
        self._clear()

    def request_end(self, request_token):
        self._ready()
        if request_token is not self.request_token or not self.complete or self.active:
            raise RuntimeError("streaming request cannot retire before complete drain")
        self._clear()

    def _clear(self):
        self.state = "READY"
        self.request_token = self.identity = self.planes = None
        self.total = self.next_start = self.chunks = 0
        self.active = self.complete = False
