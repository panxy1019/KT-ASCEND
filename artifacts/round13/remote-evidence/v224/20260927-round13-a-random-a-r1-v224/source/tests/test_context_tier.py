import unittest

from migration_bridge.round9.runtime import context_page_store as records
from migration_bridge.round9.runtime import context_tier as api


class Allocator:
    def __init__(self):
        self.available = [(9, 4), (10, 5)]
        self.live = {(7, 2), (8, 3)}
        self.released = []
    def allocate(self):
        item = self.available.pop(0); self.live.add(item); return item
    def release(self, physical, generation):
        item = (physical, generation)
        if item not in self.live: raise RuntimeError("double free")
        self.live.remove(item); self.released.append(item)


class Mapping:
    def __init__(self): self.entries = {}; self.events = []
    def publish(self, key, physical, generation):
        self.entries[key] = (physical, generation); self.events.append(("publish", key, physical, generation))
    def clear(self, key, physical, generation):
        if self.entries.get(key) != (physical, generation): raise RuntimeError("stale mapping")
        del self.entries[key]; self.events.append(("clear", key, physical, generation))


class ContextTierTests(unittest.TestCase):
    def setUp(self):
        self.store = records.RequestLocalPageStore(max_pages=8, max_host_bytes=1 << 24)
        self.allocator = Allocator(); self.mapping = Mapping(); self.imports = []
        self.payload = {}
        self.tier = api.RequestLocalContextTier(
            self.store, allocator=self.allocator, mapping=self.mapping,
            exporter=self.exporter, importer=self.importer)

    def key(self, page, request="r", epoch=1):
        return records.PageKey(request, epoch, page)

    def layers(self, marker):
        result = {}
        for layer in records.QSA_LAYER_IDS:
            plane = bytes([marker]) * (4 * 2 * 256 * 2)
            result[layer] = records.make_bf16_layer_payload(layer, 4, plane, plane)
        return result

    def exporter(self, key, physical, generation):
        return self.payload[(physical, generation)]

    def importer(self, record, physical, generation):
        self.imports.append((record.digest_sha256, physical, generation))
        self.payload[(physical, generation)] = {layer.layer_id: layer for layer in record.layers}

    def register(self, key, physical, generation, marker):
        self.payload[(physical, generation)] = self.layers(marker)
        self.mapping.publish(key, physical, generation)
        self.tier.register_resident(key, physical, generation)

    def test_spill_reload_reuses_another_physical_page_with_byte_identity(self):
        key = self.key(0); self.register(key, 7, 2, 11)
        original = tuple(item.digest_sha256 for item in self.payload[(7, 2)].values())
        spilled = self.tier.evict(key, current_logical_page=2)
        self.assertEqual(spilled.state, "SPILLED")
        self.assertNotIn(key, self.mapping.entries)
        self.assertIn((7, 2), self.allocator.released)
        resident = self.tier.prefetch(key)
        self.assertEqual((resident.physical_page, resident.physical_generation), (9, 4))
        self.assertEqual(self.mapping.entries[key], (9, 4))
        restored = tuple(item.digest_sha256 for item in self.payload[(9, 4)].values())
        self.assertEqual(restored, original)

    def test_tail_duplicate_owner_failure_and_request_isolation_fail_closed(self):
        first = self.key(0); second = self.key(1); other = self.key(0, "other")
        self.register(first, 7, 2, 1)
        self.register(second, 8, 3, 2)
        with self.assertRaisesRegex(ValueError, "tail"):
            self.tier.evict(second, current_logical_page=2)
        with self.assertRaisesRegex(ValueError, "owned"):
            self.tier.register_resident(other, 7, 2)
        self.tier.evict(first, current_logical_page=2)
        self.assertTrue(self.tier.invalidate_request("r", 1))
        self.assertEqual(self.mapping.entries, {})
        self.assertEqual(self.store.snapshot()["pages"], 0)
        self.assertEqual(self.tier.snapshot()["pages"], 0)

    def test_export_import_and_mapping_failures_preserve_one_authoritative_copy(self):
        key = self.key(0); self.register(key, 7, 2, 5)
        self.tier.exporter = lambda *args: (_ for _ in ()).throw(OSError("export"))
        with self.assertRaisesRegex(OSError, "export"):
            self.tier.evict(key, current_logical_page=2)
        self.assertEqual(self.mapping.entries[key], (7, 2))
        self.assertEqual(self.tier.snapshot()["states"], {"RESIDENT": 1})

        self.tier.exporter = self.exporter
        self.tier.evict(key, current_logical_page=2)
        self.tier.importer = lambda *args: (_ for _ in ()).throw(OSError("import"))
        with self.assertRaisesRegex(OSError, "import"):
            self.tier.prefetch(key)
        self.assertNotIn(key, self.mapping.entries)
        self.assertEqual(self.tier.snapshot()["states"], {"SPILLED": 1})
        self.assertIn((9, 4), self.allocator.released)

    def test_failure_after_host_publication_requires_reload(self):
        key = self.key(0); self.register(key, 7, 2, 9)
        self.mapping.clear = lambda *args: (_ for _ in ()).throw(OSError("clear"))
        with self.assertRaisesRegex(OSError, "clear"):
            self.tier.evict(key, current_logical_page=2)
        self.assertEqual(self.tier.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(self.mapping.entries[key], (7, 2))
        self.assertEqual(self.store.snapshot()["pages"], 1)


if __name__ == "__main__":
    unittest.main()
