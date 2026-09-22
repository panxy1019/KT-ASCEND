"""Synchronous request-local QSA page-tier skeleton with explicit ownership."""

from collections import Counter
from dataclasses import dataclass

from .context_page_store import PageKey, RequestLocalPageStore


SCOPE = "SYNCHRONOUS_REQUEST_LOCAL_TIER_NO_MODEL_HOOK"


@dataclass(frozen=True, slots=True)
class TierPage:
    key: PageKey
    state: str
    physical_page: int | None
    physical_generation: int | None
    host_generation: int | None


def _callback(owner, name):
    value = getattr(owner, name, None)
    if not callable(value):
        raise TypeError(f"{name} callback is required")
    return value


class RequestLocalContextTier:
    """One authority per page; transfers are synchronous and fail closed."""

    def __init__(self, store, *, allocator, mapping, exporter, importer):
        if not isinstance(store, RequestLocalPageStore):
            raise TypeError("RequestLocalPageStore is required")
        self.store = store
        self.allocator = allocator
        self.mapping = mapping
        self.exporter = exporter
        self.importer = importer
        for owner, names in ((allocator, ("allocate", "release")),
                             (mapping, ("publish", "clear"))):
            for name in names:
                _callback(owner, name)
        for callback, name in ((exporter, "exporter"), (importer, "importer")):
            if not callable(callback):
                raise TypeError(f"{name} callback is required")
        self.pages = {}
        self.physical_owners = {}
        self.state = "READY"

    def _ready(self):
        if self.state != "READY":
            raise RuntimeError(self.state)

    def _page(self, key):
        if not isinstance(key, PageKey) or key not in self.pages:
            raise ValueError("unknown request-local tier page")
        return self.pages[key]

    def _poison(self, error):
        self.state = "BROKEN_RELOAD_REQUIRED"
        raise error

    def register_resident(self, key, physical_page, physical_generation):
        self._ready()
        if (not isinstance(key, PageKey) or type(physical_page) is not int
                or physical_page < 0 or type(physical_generation) is not int
                or physical_generation < 0 or key in self.pages):
            raise ValueError("invalid or duplicate resident page")
        owner = (physical_page, physical_generation)
        if owner in self.physical_owners:
            raise ValueError("physical page generation is already owned")
        self.pages[key] = TierPage(
            key, "RESIDENT", physical_page, physical_generation, None)
        self.physical_owners[owner] = key
        return self.pages[key]

    def evict(self, key, *, current_logical_page):
        self._ready()
        page = self._page(key)
        if page.state != "RESIDENT":
            raise ValueError("only a resident page may evict")
        if type(current_logical_page) is not int or key.logical_page >= current_logical_page - 1:
            raise ValueError("active tail and preceding page must remain resident")
        owner = (page.physical_page, page.physical_generation)
        published = False
        try:
            layers = self.exporter(key, *owner)
            record = self.store.publish_export(key, layers)
            published = True
            self.mapping.clear(key, *owner)
            self.allocator.release(*owner)
        except BaseException as error:
            # Export failure leaves the original mapping untouched. Once a host
            # record publishes, this synchronous skeleton has no per-page rollback;
            # retain both artifacts for diagnosis and require a clean reload.
            if published:
                self._poison(error)
            raise
        del self.physical_owners[owner]
        self.pages[key] = TierPage(
            key, "SPILLED", None, None, record.page_generation)
        return self.pages[key]

    def prefetch(self, key):
        self._ready()
        page = self._page(key)
        if page.state != "SPILLED":
            raise ValueError("only a spilled page may prefetch")
        ticket = self.store.begin_reload(key)
        lease = self.store.complete_reload(ticket)
        record = self.store.read(lease)
        allocated = None
        try:
            allocated = self.allocator.allocate()
            if (not isinstance(allocated, tuple) or len(allocated) != 2
                    or any(type(value) is not int or value < 0 for value in allocated)
                    or allocated in self.physical_owners):
                raise ValueError("allocator returned an invalid physical generation")
            self.importer(record, *allocated)
            self.mapping.publish(key, *allocated)
        except BaseException:
            if allocated is not None:
                self.allocator.release(*allocated)
            self.store.release(lease)
            raise
        self.store.release(lease)
        self.physical_owners[allocated] = key
        self.pages[key] = TierPage(
            key, "RESIDENT", *allocated, page.host_generation)
        return self.pages[key]

    def invalidate_request(self, request_id, allocation_epoch):
        self._ready()
        keys = [key for key in self.pages
                if key.request_id == request_id and key.allocation_epoch == allocation_epoch]
        for key in keys:
            page = self.pages[key]
            if page.state == "RESIDENT":
                owner = (page.physical_page, page.physical_generation)
                try:
                    self.mapping.clear(key, *owner)
                    self.allocator.release(*owner)
                except BaseException as error:
                    self._poison(error)
                del self.physical_owners[owner]
        if not self.store.invalidate(request_id, allocation_epoch):
            raise RuntimeError("host records still have live leases or reloads")
        for key in keys:
            del self.pages[key]
        return True

    def snapshot(self):
        states = Counter(page.state for page in self.pages.values())
        return {
            "scope": SCOPE,
            "state": self.state,
            "pages": len(self.pages),
            "physical_owners": len(self.physical_owners),
            "states": dict(sorted(states.items())),
            "host": self.store.snapshot(),
        }
